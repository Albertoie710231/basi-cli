/* /v1/chat/completions streaming client. See srvchat.h. Pure HTTP + nlohmann;
 * no libllama/ggml symbols (part of item 6 — dropping the in-process link). */
#include "srvchat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>
#include <unistd.h>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

static double srvchat_now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static char *dup_cstr(const std::string &s) {
    char *p = (char *) malloc(s.size() + 1);
    if (p) { memcpy(p, s.data(), s.size()); p[s.size()] = 0; }
    return p;
}

/* Assemble the request body: caller messages/tools + streaming + sampling. */
static std::string build_request(const char *messages_json, const char *tools_json,
                                 const SrvSampling *samp, int n_predict) {
    json req;
    req["messages"] = json::parse(messages_json);
    if (tools_json && *tools_json) {
        json t = json::parse(tools_json);
        if (!t.empty()) { req["tools"] = std::move(t); req["tool_choice"] = "auto"; }
    }
    req["stream"] = true;
    req["stream_options"] = json{{"include_usage", true}};
    if (n_predict > 0) req["max_tokens"] = n_predict;
    if (samp) {
        if (samp->temperature >= 0)   req["temperature"] = samp->temperature;
        if (samp->temperature == 0.0) req["top_k"] = 1;              /* greedy */
        else if (samp->top_k > 0)     req["top_k"] = samp->top_k;
        if (samp->repeat_penalty > 1.0) {
            req["repeat_penalty"] = samp->repeat_penalty;
            if (samp->repeat_last_n >= 0) req["repeat_last_n"] = samp->repeat_last_n;
        }
        if (samp->min_p >= 0)                      req["min_p"] = samp->min_p;
        if (samp->top_p >= 0 && samp->top_p < 1.0) req["top_p"] = samp->top_p;
        if (samp->seed >= 0)                       req["seed"] = samp->seed;
    }
    req["cache_prompt"] = true;
    return req.dump();
}

/* One streamed tool call, accumulated across deltas (arguments arrive in pieces). */
struct AccTool { std::string id, name, args; };

extern "C" SrvChatResult *srvchat_complete(
        int port, const char *messages_json, const char *tools_json,
        const SrvSampling *samp, int n_predict,
        void (*on_content)(const char *, void *),
        void (*on_reasoning)(const char *, void *),
        void *ud) {

    std::string body;
    try { body = build_request(messages_json, tools_json, samp, n_predict); }
    catch (...) { return nullptr; }   /* caller passed invalid messages/tools JSON */

    /* Write the request to a temp file (avoids quoting a big body on the argv). */
    char reqpath[] = "/tmp/basi_srvchat_XXXXXX";
    int rfd = mkstemp(reqpath);
    if (rfd < 0) return nullptr;
    { FILE *rf = fdopen(rfd, "w"); if (!rf) { close(rfd); unlink(reqpath); return nullptr; }
      fwrite(body.data(), 1, body.size(), rf); fclose(rf); }

    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "curl -N -s -X POST http://127.0.0.1:%d/v1/chat/completions "
        "-H 'Content-Type: application/json' --data-binary @%s", port, reqpath);

    FILE *p = popen(cmd, "r");
    if (!p) { unlink(reqpath); return nullptr; }

    std::string content, reasoning, finish;
    std::vector<AccTool> tools;
    int prompt_tokens = 0, completion_tokens = 0;
    double tps = 0, prompt_tps = 0;

    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, p)) != -1) {
        char *j = strstr(line, "data:");
        if (!j) continue;
        j += 5; while (*j == ' ') j++;
        if (strncmp(j, "[DONE]", 6) == 0) break;

        json d;
        try { d = json::parse(j); } catch (...) { continue; }

        if (d.contains("choices") && d["choices"].is_array() && !d["choices"].empty()) {
            const json &ch = d["choices"][0];
            if (ch.contains("delta") && ch["delta"].is_object()) {
                const json &delta = ch["delta"];
                if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                    std::string r = delta["reasoning_content"].get<std::string>();
                    if (!r.empty()) { reasoning += r; if (on_reasoning) on_reasoning(r.c_str(), ud); }
                }
                if (delta.contains("content") && delta["content"].is_string()) {
                    std::string c = delta["content"].get<std::string>();
                    if (!c.empty()) { content += c; if (on_content) on_content(c.c_str(), ud); }
                }
                if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
                    for (const json &tc : delta["tool_calls"]) {
                        size_t idx = tc.value("index", 0);
                        if (idx >= tools.size()) tools.resize(idx + 1);
                        AccTool &at = tools[idx];
                        if (tc.contains("id") && tc["id"].is_string()) at.id = tc["id"].get<std::string>();
                        if (tc.contains("function") && tc["function"].is_object()) {
                            const json &fn = tc["function"];
                            if (fn.contains("name") && fn["name"].is_string()) at.name = fn["name"].get<std::string>();
                            if (fn.contains("arguments") && fn["arguments"].is_string()) at.args += fn["arguments"].get<std::string>();
                        }
                    }
                }
            }
            if (ch.contains("finish_reason") && ch["finish_reason"].is_string())
                finish = ch["finish_reason"].get<std::string>();
        }
        if (d.contains("usage") && d["usage"].is_object()) {
            prompt_tokens     = d["usage"].value("prompt_tokens", 0);
            completion_tokens = d["usage"].value("completion_tokens", 0);
        }
        if (d.contains("timings") && d["timings"].is_object()) {
            tps        = d["timings"].value("predicted_per_second", 0.0);
            prompt_tps = d["timings"].value("prompt_per_second", 0.0);
        }
    }
    free(line);
    pclose(p);
    unlink(reqpath);

    SrvChatResult *res = (SrvChatResult *) calloc(1, sizeof *res);
    if (!res) return nullptr;
    res->content   = dup_cstr(content);
    res->reasoning = reasoning.empty() ? nullptr : dup_cstr(reasoning);
    res->finish_reason = finish.empty() ? nullptr : dup_cstr(finish);
    res->prompt_tokens = prompt_tokens;
    res->completion_tokens = completion_tokens;
    res->tps = tps;
    res->prompt_tps = prompt_tps;
    if (!tools.empty()) {
        res->tool_calls = (SrvToolCall *) calloc(tools.size(), sizeof(SrvToolCall));
        if (res->tool_calls) {
            for (size_t i = 0; i < tools.size(); i++) {
                res->tool_calls[i].id        = dup_cstr(tools[i].id);
                res->tool_calls[i].name      = dup_cstr(tools[i].name);
                res->tool_calls[i].arguments = dup_cstr(tools[i].args.empty() ? std::string("{}") : tools[i].args);
            }
            res->n_tool_calls = (int) tools.size();
        }
    }
    return res;
}

extern "C" void srvchat_free(SrvChatResult *r) {
    if (!r) return;
    free(r->content);
    free(r->reasoning);
    free(r->finish_reason);
    for (int i = 0; i < r->n_tool_calls; i++) {
        free(r->tool_calls[i].id);
        free(r->tool_calls[i].name);
        free(r->tool_calls[i].arguments);
    }
    free(r->tool_calls);
    free(r);
}

/* ── /embedding client ───────────────────────────────────────────────────── */

extern "C" int srvchat_embed(int port, const char *text, float *out, int max_dim) {
    if (!text || !out || max_dim <= 0) return -1;

    std::string body;
    try { nlohmann::json req; req["content"] = text; body = req.dump(); }
    catch (...) { return -1; }

    char reqpath[] = "/tmp/basi_srvemb_XXXXXX";
    int rfd = mkstemp(reqpath);
    if (rfd < 0) return -1;
    { FILE *rf = fdopen(rfd, "w"); if (!rf) { close(rfd); unlink(reqpath); return -1; }
      fwrite(body.data(), 1, body.size(), rf); fclose(rf); }

    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "curl -s -X POST http://127.0.0.1:%d/embedding "
        "-H 'Content-Type: application/json' --data-binary @%s", port, reqpath);

    FILE *p = popen(cmd, "r");
    if (!p) { unlink(reqpath); return -1; }
    std::string resp;
    { char buf[8192]; size_t n; while ((n = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, n); }
    pclose(p);
    unlink(reqpath);

    try {
        nlohmann::json d = nlohmann::json::parse(resp);
        // Response is [{"index":0,"embedding":[...] or [[...]]}] (native) or
        // {"embedding":[...]} — locate the float vector, flattening one nesting level.
        const nlohmann::json *emb = nullptr;
        if (d.is_array() && !d.empty() && d[0].is_object() && d[0].contains("embedding"))
            emb = &d[0]["embedding"];
        else if (d.is_object() && d.contains("embedding"))
            emb = &d["embedding"];
        if (!emb || !emb->is_array() || emb->empty()) return -1;
        const nlohmann::json *vec = emb;
        if ((*emb)[0].is_array()) vec = &(*emb)[0];   // "embedding":[[...]] → row 0
        if (!vec->is_array()) return -1;
        int n = (int) vec->size();
        if (n > max_dim) n = max_dim;
        for (int i = 0; i < n; i++) out[i] = (*vec)[i].get<float>();
        return n;
    } catch (...) { return -1; }
}

/* ── self-test ──────────────────────────────────────────────────────────── */

static void st_content(const char *c, void *) { fputs(c, stdout); fflush(stdout); }
static void st_reason(const char *c, void *)  { fprintf(stderr, "\033[90m%s\033[0m", c); fflush(stderr); }

extern "C" void srvchat_selftest(const char *model_path, int ngl, int ctx) {
    const char *server_bin = getenv("BASI_SERVER_BIN");
    if (!server_bin || !*server_bin)
        server_bin = "/home/alberto/llama.cpp/build_vulkan/bin/llama-server";
    const int port = 8181;

    char extra[160] = "--jinja --reasoning-format auto";
    const char *spec = getenv("BASI_SPEC");
    if (spec && *spec) {
        int nmax = 1; const char *e = getenv("BASI_SPEC_NMAX"); if (e && *e) nmax = atoi(e);
        char sp[80]; snprintf(sp, sizeof sp, " -fa on --spec-type %s --spec-draft-n-max %d", spec, nmax);
        strncat(extra, sp, sizeof extra - strlen(extra) - 1);
    }

    fprintf(stderr, "\n=== srvchat self-test (ngl=%d ctx=%d) ===\n[srvchat] spawning llama-server…\n", ngl, ctx);
    pid_t pid = srvgen_spawn(server_bin, model_path, ngl, ctx, extra, port, "/tmp/basi_srvgen.log", 300);
    if (pid < 0) { fprintf(stderr, "[srvchat] spawn/health FAILED (see /tmp/basi_srvgen.log)\n"); return; }
    fprintf(stderr, "[srvchat] ready. streaming a chat completion (bash tool available):\n---\n");

    const char *messages = "[{\"role\":\"user\",\"content\":\"Think briefly, then list the files in the current directory.\"}]";
    const char *tools =
        "[{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"description\":\"run a shell command\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}}}]";

    SrvSampling samp = { .temperature = 0.0, .repeat_penalty = 1.1, .repeat_last_n = 256,
                         .min_p = 0.05, .top_k = 0, .top_p = 1.0, .seed = -1 };
    SrvChatResult *r = srvchat_complete(port, messages, tools, &samp, 200, st_content, st_reason, nullptr);
    fprintf(stderr, "\n---\n");
    if (!r) { fprintf(stderr, "[srvchat] request FAILED\n"); srvgen_kill(pid); return; }

    fprintf(stderr, "[srvchat] finish=%s  prompt_tokens=%d  completion_tokens=%d  %.2f tok/s\n",
            r->finish_reason ? r->finish_reason : "(none)", r->prompt_tokens, r->completion_tokens, r->tps);
    fprintf(stderr, "[srvchat] content bytes=%zu  reasoning bytes=%zu  tool_calls=%d\n",
            r->content ? strlen(r->content) : 0, r->reasoning ? strlen(r->reasoning) : 0, r->n_tool_calls);
    for (int i = 0; i < r->n_tool_calls; i++)
        fprintf(stderr, "[srvchat]   tool_call[%d]: %s(%s)  id=%s\n",
                i, r->tool_calls[i].name, r->tool_calls[i].arguments, r->tool_calls[i].id);

    srvchat_free(r);
    srvgen_kill(pid);
    fprintf(stderr, "[srvchat] server killed. done.\n");
}
