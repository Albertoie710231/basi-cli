// Message/tool serialization for the /v1/chat/completions (pure-HTTP) path.
// Generation, templating, tool-call grammar and parsing all happen server-side
// now, so the in-process common_chat engine is gone — only the tool registry and
// these nlohmann serializers remain. No llama_/common_chat FUNCTIONS are called
// here (BasiMsg is a POD used for its fields only), so this translation
// unit links ZERO libllama symbols.
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <utility>

#include "basi_types.h"            // BasiMsg (POD {role, content}) — type only
#include "nlohmann/json.hpp"
#include "chat_tmpl.h"

namespace {
struct Tool { std::string name, description, parameters; };
std::vector<Tool> g_tools;   // the registered tool set (session-constant)

std::string b64(const std::string &in) {
    static const char *T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        unsigned v = ((unsigned char)in[i] << 16) | ((unsigned char)in[i+1] << 8) | (unsigned char)in[i+2];
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63];
        out += T[(v >> 6) & 63];  out += T[v & 63];
        i += 3;
    }
    if (i + 1 == in.size()) {
        unsigned v = (unsigned char)in[i] << 16;
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63]; out += "==";
    } else if (i + 2 == in.size()) {
        unsigned v = ((unsigned char)in[i] << 16) | ((unsigned char)in[i+1] << 8);
        out += T[(v >> 18) & 63]; out += T[(v >> 12) & 63]; out += T[(v >> 6) & 63]; out += '=';
    }
    return out;
}

// Images live on disk and are read HERE, at serialization time, rather than
// being carried inside the message. A 900x500 PNG is ~300KB of base64; holding
// that in BasiMsg.content would make every char-based context estimate and
// every compaction pass believe the conversation is ~70k tokens larger than it
// is, for an image the model bills at ~500. The message stays a path.
bool read_file_bytes(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return !out.empty();
}

// data: URI mime from the extension. The dispatcher normalizes everything to
// PNG before it gets here, so this is a fallback for a hand-written path.
const char *mime_of(const std::string &path) {
    size_t d = path.rfind('.');
    if (d == std::string::npos) return "image/png";
    std::string e = path.substr(d + 1);
    for (auto &c : e) c = (char) tolower((unsigned char) c);
    if (e == "jpg" || e == "jpeg") return "image/jpeg";
    if (e == "webp")               return "image/webp";
    if (e == "gif")                return "image/gif";
    return "image/png";
}
}

extern "C" void basi_set_tools(const BasiToolDef *defs, int n) {
    g_tools.clear();
    if (!defs || n <= 0) return;
    g_tools.reserve(n);
    for (int i = 0; i < n; i++)
        g_tools.push_back({ defs[i].name       ? defs[i].name       : "",
                            defs[i].description ? defs[i].description : "",
                            defs[i].parameters  ? defs[i].parameters  : "{}" });
}

// How many tools are currently advertised. Lets a self-contained sub-generation
// (deepsearch, summary) clear the schemas and restore the prior state exactly.
extern "C" int basi_tools_registered(void) { return (int) g_tools.size(); }

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
        /* error_handler_t::replace: never THROW on invalid UTF-8. Tool results
           carry arbitrary fetched bytes; a single bad sequence must degrade to
           U+FFFD, not abort serialization and blank out the whole turn. */
        return strdup(arr.dump(-1, ' ', false,
                               nlohmann::json::error_handler_t::replace).c_str());
    } catch (...) { return nullptr; }
}

// BASI messages → OpenAI `messages` array (malloc'd JSON string, caller frees).
// The custom roles map as: tool_call → assistant with tool_calls[] (synthetic
// call_N id); tool_result → tool with tool_call_id = the preceding call's id.
extern "C" char *basi_messages_to_json(const BasiMsg *msgs, int n_msgs) {
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
            } else if (strcmp(role, "image") == 0) {
                /* content = {"path":..., "text":...} — an image the agent asked
                 * to look at. OpenAI has no way to put an image in a role:"tool"
                 * result, so a view_image call lands as the text result plus
                 * THIS user message carrying the pixels. Kept under its own
                 * internal role so compaction's user-boundary walk does not
                 * mistake it for a real user turn. */
                std::string path, text;
                try {
                    auto j = nlohmann::ordered_json::parse(content);
                    if (j.contains("path") && j["path"].is_string()) path = j["path"].get<std::string>();
                    if (j.contains("text") && j["text"].is_string()) text = j["text"].get<std::string>();
                } catch (...) {}
                std::string bytes;
                if (path.empty() || !read_file_bytes(path, bytes)) {
                    /* The render was overwritten or cleaned up between the call
                     * and this turn. Say so as text rather than dropping the
                     * message: a silent disappearance would leave the model
                     * arguing about an image nobody can see. */
                    std::string note = text.empty() ? std::string() : text + "\n";
                    note += "[image at " + (path.empty() ? std::string("(no path)") : path) +
                            " could not be read — it may have been overwritten or deleted]";
                    arr.push_back({ {"role", "user"}, {"content", note} });
                } else {
                    nlohmann::ordered_json parts = nlohmann::ordered_json::array();
                    if (!text.empty())
                        parts.push_back({ {"type", "text"}, {"text", text} });
                    parts.push_back({ {"type", "image_url"},
                                      {"image_url", { {"url", std::string("data:") + mime_of(path) +
                                                              ";base64," + b64(bytes)} }} });
                    arr.push_back({ {"role", "user"}, {"content", std::move(parts)} });
                }
            } else {
                arr.push_back({ {"role", role}, {"content", content} });
            }
        }
        /* error_handler_t::replace: never THROW on invalid UTF-8. Tool results
           carry arbitrary fetched bytes; a single bad sequence must degrade to
           U+FFFD, not abort serialization and blank out the whole turn. */
        return strdup(arr.dump(-1, ' ', false,
                               nlohmann::json::error_handler_t::replace).c_str());
    } catch (...) { return nullptr; }
}

extern "C" void basi_free_tool_calls(BasiToolCall *calls, int n) {
    if (!calls) return;
    for (int i = 0; i < n; i++) {
        free(calls[i].name);
        free(calls[i].arguments);
    }
    free(calls);
}
