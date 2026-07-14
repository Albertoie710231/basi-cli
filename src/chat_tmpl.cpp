// Message/tool serialization for the /v1/chat/completions (pure-HTTP) path.
// Generation, templating, tool-call grammar and parsing all happen server-side
// now, so the in-process common_chat engine is gone — only the tool registry and
// these nlohmann serializers remain. No llama_/common_chat FUNCTIONS are called
// here (llama_chat_message is a POD used for its fields only), so this translation
// unit links ZERO libllama symbols.
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <utility>

#include "llama.h"            // llama_chat_message (POD {role, content}) — type only
#include "nlohmann/json.hpp"
#include "chat_tmpl.h"

namespace {
struct Tool { std::string name, description, parameters; };
std::vector<Tool> g_tools;   // the registered tool set (session-constant)
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

extern "C" void basi_free_tool_calls(BasiToolCall *calls, int n) {
    if (!calls) return;
    for (int i = 0; i < n; i++) {
        free(calls[i].name);
        free(calls[i].arguments);
    }
    free(calls);
}
