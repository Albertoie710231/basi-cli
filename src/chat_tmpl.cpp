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
#include "common.h"        // regex_escape, common_grammar_trigger
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

/* How many tools are currently advertised to the model (g_tools size). Lets a
   self-contained sub-generation (deepsearch, summary) clear the schemas and then
   restore the prior state exactly. */
extern "C" int basi_tools_registered(void) { return (int) g_tools.size(); }

// ── OpenAI-format serialization for the /v1/chat/completions path (item 6) ──
// These turn BASI's message array + registered tools into the JSON the server's
// chat endpoint expects, so the server can do templating + grammar + parsing.
// nlohmann-only (no common_chat/llama beyond the POD llama_chat_message), so they
// survive when the in-process templating engine is removed.

// Registered tools → OpenAI `tools` array (malloc'd JSON string, caller frees), or
// NULL if none. parameters is stored as a JSON string; parse it back to an object.
extern "C" char *basi_tools_to_json(void) {
    if (g_tools.empty()) return nullptr;
    try {
        nlohmann::ordered_json arr = nlohmann::ordered_json::array();
        for (const auto &t : g_tools) {
            nlohmann::ordered_json fn;
            fn["name"]        = t.name;
            fn["description"] = t.description;
            try { fn["parameters"] = nlohmann::ordered_json::parse(t.parameters); }
            catch (...) { fn["parameters"] = nlohmann::ordered_json::object(); }
            arr.push_back({ {"type", "function"}, {"function", std::move(fn)} });
        }
        return strdup(arr.dump().c_str());
    } catch (...) { return nullptr; }
}

// BASI messages → OpenAI `messages` array (malloc'd JSON string, caller frees).
// The custom roles map as: tool_call → assistant with tool_calls[] (synthetic
// call_N id); tool_result → tool with tool_call_id = the preceding call's id.
extern "C" char *basi_messages_to_json(const struct llama_chat_message *msgs, int n_msgs) {
    try {
        nlohmann::ordered_json arr = nlohmann::ordered_json::array();
        int call_seq = 0;                 // ids assigned to tool_calls in order
        std::string last_call_id;         // most recent call id, for pairing a result
        for (int i = 0; i < n_msgs; i++) {
            const char *role    = msgs[i].role    ? msgs[i].role    : "";
            const char *content = msgs[i].content ? msgs[i].content : "";
            if (strcmp(role, "tool_call") == 0) {
                // content = {"name":..., "arguments":<json>}
                nlohmann::ordered_json m;
                m["role"] = "assistant";
                m["content"] = nullptr;
                char idbuf[32]; snprintf(idbuf, sizeof idbuf, "call_%d", call_seq++);
                last_call_id = idbuf;
                std::string name, args = "{}";
                try {
                    auto j = nlohmann::ordered_json::parse(content);
                    if (j.contains("name")) name = j["name"].get<std::string>();
                    if (j.contains("arguments"))
                        args = j["arguments"].is_string() ? j["arguments"].get<std::string>()
                                                          : j["arguments"].dump();
                } catch (...) {}
                m["tool_calls"] = nlohmann::ordered_json::array({
                    { {"id", idbuf}, {"type", "function"},
                      {"function", { {"name", name}, {"arguments", args} }} } });
                arr.push_back(std::move(m));
            } else if (strcmp(role, "tool_result") == 0) {
                // content = {"name":..., "content":...}
                nlohmann::ordered_json m;
                m["role"] = "tool";
                if (!last_call_id.empty()) m["tool_call_id"] = last_call_id;
                std::string body = content;
                try {
                    auto j = nlohmann::ordered_json::parse(content);
                    if (j.contains("content")) body = j["content"].get<std::string>();
                } catch (...) {}
                m["content"] = body;
                arr.push_back(std::move(m));
            } else {
                arr.push_back({ {"role", role}, {"content", content} });
            }
        }
        return strdup(arr.dump().c_str());
    } catch (...) { return nullptr; }
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

// Phase 2b — tool-call grammar. common_chat already DERIVES a GBNF grammar that
// constrains the model's output to a valid tool call (chat.h: "incl. tool call
// grammar constraining"); without it a small model drifts off-format after a few
// rounds, emits malformed <tool_call> JSON, and the PEG parser rejects it. Build
// the matching llama grammar sampler so decoding is constrained. The grammar is
// determined by the tool set + format (session-constant), so a single probe
// render yields it. Mirrors the trigger conversion in common/sampling.cpp. The
// grammar is LAZY: it stays inactive until a trigger (e.g. "<tool_call>") so the
// model can still think and answer freely; it only forces valid JSON once a call
// begins. Returns a sampler the caller adds to its chain (and must reset between
// generations), or NULL when this format has no grammar / on any error.
extern "C" struct llama_sampler *basi_tool_grammar_sampler(const struct llama_model *model) {
    if (g_tools.empty() || !model) return nullptr;
    try {
        if (!ensure_tmpls(model)) return nullptr;
        llama_chat_message probe[1] = { { "user", "hi" } };
        common_chat_templates_inputs in = make_inputs(probe, 1, true);
        common_chat_params params = common_chat_templates_apply(g_tmpls.get(), in);
        const std::string &grammar = params.grammar;
        if (grammar.empty()) return nullptr;
        if (grammar.compare(0, 11, "%llguidance") == 0) return nullptr;  // not built here
        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (!vocab) return nullptr;

        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto &trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger_patterns.push_back(regex_escape(trigger.value));
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger_patterns.push_back(trigger.value);
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                    const auto &p = trigger.value;
                    std::string anchored = "^$";
                    if (!p.empty())
                        anchored = (p.front() != '^' ? "^" : "") + p + (p.back() != '$' ? "$" : "");
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger_tokens.push_back(trigger.token);
                    break;
                default:
                    break;
            }
        }
        std::vector<const char *> pats;
        pats.reserve(trigger_patterns.size());
        for (const auto &r : trigger_patterns) pats.push_back(r.c_str());

        struct llama_sampler *g;
        if (params.grammar_lazy) {
            g = llama_sampler_init_grammar_lazy_patterns(
                    vocab, grammar.c_str(), "root",
                    pats.data(), pats.size(),
                    trigger_tokens.data(), trigger_tokens.size());
        } else {
            g = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
        }
        if (getenv("BASI_DEBUG_GRAMMAR"))
            fprintf(stderr, "[grammar] format=%s lazy=%d patterns=%zu tokens=%zu bytes=%zu\n",
                    common_chat_format_name(params.format), (int)params.grammar_lazy,
                    trigger_patterns.size(), trigger_tokens.size(), grammar.size());
        return g;
    } catch (...) {
        return nullptr;
    }
}

// Server-backend (M2): the SAME tool-call grammar as basi_tool_grammar_sampler,
// but serialized as the llama-server /completion request fields — the inner
// object body (no braces) "grammar":...,"grammar_lazy":...,"grammar_triggers":[...]
// ready to splice into a request. NULL if this format has no grammar. Caller frees.
extern "C" char *basi_tool_grammar_json(const struct llama_model *model) {
    if (g_tools.empty() || !model) return nullptr;
    try {
        if (!ensure_tmpls(model)) return nullptr;
        llama_chat_message probe[1] = { { "user", "hi" } };
        common_chat_templates_inputs in = make_inputs(probe, 1, true);
        common_chat_params params = common_chat_templates_apply(g_tmpls.get(), in);
        if (params.grammar.empty()) return nullptr;
        if (params.grammar.compare(0, 11, "%llguidance") == 0) return nullptr;

        nlohmann::json triggers = nlohmann::json::array();
        for (const auto & t : params.grammar_triggers)
            triggers.push_back({ {"type", (int) t.type}, {"value", t.value}, {"token", t.token} });

        nlohmann::json obj;
        obj["grammar"]          = params.grammar;
        obj["grammar_lazy"]     = params.grammar_lazy;
        obj["grammar_triggers"] = triggers;
        std::string s = obj.dump();
        if (s.size() >= 2 && s.front() == '{' && s.back() == '}')
            s = s.substr(1, s.size() - 2);   // strip outer braces -> spliceable fragment
        return strdup(s.c_str());
    } catch (...) {
        return nullptr;
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
