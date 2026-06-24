// C++ shim: render a model's native chat template via llama.cpp's jinja engine
// (libllama-common), and — phase 2a — advertise tools and parse tool calls in
// the model's OWN trained format using llama.cpp's common_chat machinery. The
// rest of BASI is C; this is the one C++ translation unit, exposing extern "C".
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <utility>

#include "llama.h"
#include "chat.h"          // llama.cpp common/chat.h (common_chat_*)
#include "nlohmann/json.hpp"
#include "chat_tmpl.h"

// Single model, single-threaded: cache the parsed templates, the pinned `now`
// (so date-aware templates render a STABLE prefix across turns — the main
// loop's delta-prompt bookkeeping depends on that), the registered tools, and
// the most recent common_chat_params (its format/grammar drive the parser).
static const llama_model           *g_model = nullptr;
static common_chat_templates_ptr    g_tmpls;
static std::chrono::system_clock::time_point g_now = std::chrono::system_clock::now();
static std::vector<common_chat_tool> g_tools;
static common_chat_params           g_last_params;
static bool                         g_have_params = false;

static bool ensure_tmpls(const llama_model *model) {
    if (!model) return false;
    if (!g_tmpls || g_model != model) {
        g_tmpls = common_chat_templates_init(model, "");
        g_model = model;
    }
    return (bool)g_tmpls;
}

// Build the inputs shared by render and probe. Tools are attached when set.
static common_chat_templates_inputs make_inputs(
        const llama_chat_message *msgs, size_t n_msgs, bool add_gen_prompt) {
    common_chat_templates_inputs in;
    in.use_jinja = true;
    in.add_generation_prompt = add_gen_prompt;
    in.now = g_now;
    if (!g_tools.empty()) {
        in.tools = g_tools;
        in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;   // 2b will raise this to REQUIRED on turn 1
    }
    in.messages.reserve(n_msgs);
    for (size_t i = 0; i < n_msgs; i++) {
        const char *role    = msgs[i].role    ? msgs[i].role    : "";
        const char *content = msgs[i].content ? msgs[i].content : "";
        common_chat_msg m;
        if (strcmp(role, "tool_call") == 0) {
            /* content = {"name":..., "arguments":<json>} — rebuild a STRUCTURED
               assistant tool call so common_chat renders it in this model's own
               format (instead of replaying raw text the template can't pair). */
            m.role = "assistant";
            try {
                auto j = nlohmann::ordered_json::parse(content);
                common_chat_tool_call tc;
                if (j.contains("name")) tc.name = j["name"].get<std::string>();
                if (j.contains("arguments"))
                    tc.arguments = j["arguments"].is_string()
                        ? j["arguments"].get<std::string>()
                        : j["arguments"].dump();
                m.tool_calls.push_back(std::move(tc));
            } catch (...) { m.content = content; }
        } else if (strcmp(role, "tool_result") == 0) {
            /* content = {"name":..., "content":...} — a PROPER tool message, so
               the template emits its native tool-response slot. (A bare
               role:"tool" content message is silently dropped by e.g. Gemma.) */
            m.role = "tool";
            try {
                auto j = nlohmann::ordered_json::parse(content);
                if (j.contains("name"))    m.tool_name = j["name"].get<std::string>();
                if (j.contains("content")) m.content   = j["content"].get<std::string>();
            } catch (...) { m.content = content; }
        } else {
            m.role    = role;
            m.content = content;
        }
        in.messages.push_back(std::move(m));
    }
    return in;
}

extern "C" void basi_set_tools(const BasiToolDef *defs, int n) {
    g_tools.clear();
    if (!defs || n <= 0) return;
    g_tools.reserve(n);
    for (int i = 0; i < n; i++) {
        common_chat_tool t;
        t.name        = defs[i].name        ? defs[i].name        : "";
        t.description = defs[i].description  ? defs[i].description  : "";
        t.parameters  = defs[i].parameters   ? defs[i].parameters   : "{}";
        g_tools.push_back(std::move(t));
    }
}

extern "C" int basi_tools_active(const struct llama_model *model) {
    if (g_tools.empty()) return 0;
    try {
        if (!ensure_tmpls(model)) return 0;
        // Probe: a tiny render with tools tells us which format this template
        // resolves to. CONTENT_ONLY means it has no tool support.
        llama_chat_message probe[1] = { { "user", "hi" } };
        common_chat_templates_inputs in = make_inputs(probe, 1, true);
        common_chat_params params = common_chat_templates_apply(g_tmpls.get(), in);
        return params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" int basi_parse_tool_calls(const char *text, BasiToolCall **out) {
    *out = nullptr;
    if (!text || !g_have_params) return 0;
    try {
        // The base constructor only copies format + generation_prompt; the PEG
        // parser arena (the per-format tool grammar) lives in params.parser as
        // a serialized string and must be deserialized in, or common_chat_parse
        // falls back to a content-only parser and finds no tool calls.
        common_chat_parser_params pp(g_last_params);
        pp.parser.load(g_last_params.parser);
        common_chat_msg msg = common_chat_parse(std::string(text), /*is_partial*/ false, pp);
        if (msg.tool_calls.empty()) return 0;
        int n = (int)msg.tool_calls.size();
        BasiToolCall *arr = (BasiToolCall *)calloc(n, sizeof(BasiToolCall));
        if (!arr) return 0;
        for (int i = 0; i < n; i++) {
            arr[i].name      = strdup(msg.tool_calls[i].name.c_str());
            arr[i].arguments = strdup(msg.tool_calls[i].arguments.c_str());
        }
        *out = arr;
        return n;
    } catch (...) {
        return 0;   // malformed call → treated as a plain answer (2b grammar prevents this)
    }
}

extern "C" int basi_thinking_tags(const char **start, const char **end) {
    if (!g_have_params || !g_last_params.supports_thinking) return 0;
    if (g_last_params.thinking_start_tag.empty() ||
        g_last_params.thinking_end_tag.empty()) return 0;
    if (start) *start = g_last_params.thinking_start_tag.c_str();
    if (end)   *end   = g_last_params.thinking_end_tag.c_str();
    return 1;
}

extern "C" void basi_free_tool_calls(BasiToolCall *calls, int n) {
    if (!calls) return;
    for (int i = 0; i < n; i++) {
        free(calls[i].name);
        free(calls[i].arguments);
    }
    free(calls);
}

extern "C" char *basi_render_chat(const struct llama_model *model,
                                  const struct llama_chat_message *msgs, size_t n_msgs,
                                  bool add_gen_prompt) {
    if (!model || !msgs) return nullptr;
    try {
        if (!ensure_tmpls(model)) return nullptr;
        common_chat_templates_inputs in = make_inputs(msgs, n_msgs, add_gen_prompt);
        common_chat_params params = common_chat_templates_apply(g_tmpls.get(), in);
        g_last_params = params;     // cache for basi_parse_tool_calls (format/grammar)
        g_have_params = true;
        if (getenv("BASI_DEBUG_THINK")) {
            // Diagnostic: what reasoning format/tags does common_chat derive
            // for THIS model's template? (One-shot, stderr, opt-in.)
            static bool printed = false;
            if (!printed) {
                printed = true;
                fprintf(stderr,
                    "[think] format=%s supports_thinking=%d start=<<%s>> end=<<%s>>\n",
                    common_chat_format_name(params.format),
                    (int)params.supports_thinking,
                    params.thinking_start_tag.c_str(),
                    params.thinking_end_tag.c_str());
            }
        }
        if (getenv("BASI_DEBUG_PROMPT")) {
            fprintf(stderr, "\n===RENDERED PROMPT (%zu bytes)===\n%s\n===END===\n",
                    params.prompt.size(), params.prompt.c_str());
        }
        const std::string &p = params.prompt;
        char *out = static_cast<char *>(malloc(p.size() + 1));
        if (!out) return nullptr;
        memcpy(out, p.data(), p.size());
        out[p.size()] = '\0';
        return out;
    } catch (...) {
        return nullptr;   // any jinja/parse error → caller falls back
    }
}
