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

/* Defined in model.c. When nonzero, a caller (study_ground) has scoped a no-think
   region: add enable_thinking=false to every request until it clears the flag. */
extern "C" int basi_srv_no_think;

static double srvchat_now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static char *dup_cstr(const std::string &s) {
    char *p = (char *) malloc(s.size() + 1);
    if (p) { memcpy(p, s.data(), s.size()); p[s.size()] = 0; }
    return p;
}

/* Assemble the request body: caller messages/tools + streaming + sampling.
 * n_choices>1 asks the server for N independent continuations of ONE prefill —
 * measured at 6720 prompt tokens: n=4 returns 4 answers in 11.5s with
 * cached_tokens=6716, vs 33s for 4 separate concurrent requests that each
 * re-prefill the shared prefix. Streaming preserves per-choice indices. */
static std::string build_request(const char *messages_json, const char *tools_json,
                                 const SrvSampling *samp, int n_predict, int n_choices) {
    json req;
    if (n_choices > 1) req["n"] = n_choices;
    req["messages"] = json::parse(messages_json);
    if (tools_json && *tools_json) {
        json t = json::parse(tools_json);
        if (!t.empty()) { req["tools"] = std::move(t); req["tool_choice"] = "auto"; }
    }
    req["stream"] = true;
    req["stream_options"] = json{{"include_usage", true}};
    /* BASI_NO_THINK=1: suppress the model's reasoning (<think>) blocks. On a
     * reasoning-tuned model (Qwen3.x) the per-turn <think> span dominates
     * generation time; disabling it via the chat template's enable_thinking=false
     * kwarg makes the agent loop several times faster at some accuracy cost. */
    const char *nt = getenv("BASI_NO_THINK");
    bool no_think = basi_srv_no_think || (nt && *nt && atoi(nt));
    if (no_think) {
        req["chat_template_kwargs"] = json{{"enable_thinking", false}};
        req["reasoning_budget"] = 0;   /* llama-server native no-think, if supported */
    }
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

/* One in-flight choice. With n=1 there is exactly one of these, which is why the
 * single-result entry point is a thin wrapper rather than a separate parser. */
struct AccChoice {
    std::string content, reasoning, finish;
    std::vector<AccTool> tools;
};

/* The response-level usage/timings block. One per response, shared by every
 * choice, so it is accumulated during the stream and handed to each finalize. */
struct RespStats {
    int    prompt_tokens = 0, completion_tokens = 0, prompt_n = 0;
    double tps = 0, prompt_tps = 0, prompt_ms = 0, predicted_ms = 0;
};

/* Turn an accumulated choice into the malloc'd C result the callers free with
 * srvchat_free(). Usage/timings are per-response, so they are passed in. */
static SrvChatResult *finalize_choice(const AccChoice &ac, const RespStats &st) {
    SrvChatResult *res = (SrvChatResult *) calloc(1, sizeof *res);
    if (!res) return nullptr;
    res->content       = dup_cstr(ac.content);
    res->reasoning     = ac.reasoning.empty() ? nullptr : dup_cstr(ac.reasoning);
    res->finish_reason = ac.finish.empty()    ? nullptr : dup_cstr(ac.finish);
    res->prompt_tokens     = st.prompt_tokens;
    res->completion_tokens = st.completion_tokens;
    res->tps               = st.tps;
    res->prompt_tps        = st.prompt_tps;
    res->prompt_n          = st.prompt_n;
    res->prompt_ms         = st.prompt_ms;
    res->predicted_ms      = st.predicted_ms;
    if (!ac.tools.empty()) {
        res->tool_calls = (SrvToolCall *) calloc(ac.tools.size(), sizeof(SrvToolCall));
        if (res->tool_calls) {
            for (size_t i = 0; i < ac.tools.size(); i++) {
                res->tool_calls[i].id        = dup_cstr(ac.tools[i].id);
                res->tool_calls[i].name      = dup_cstr(ac.tools[i].name);
                res->tool_calls[i].arguments = dup_cstr(ac.tools[i].args.empty()
                                                        ? std::string("{}") : ac.tools[i].args);
            }
            res->n_tool_calls = (int) ac.tools.size();
        }
    }
    return res;
}

extern "C" int srvchat_complete_n(
        int port, const char *messages_json, const char *tools_json,
        const SrvSampling *samp, int n_predict, int n_choices,
        void (*on_content)(int, const char *, void *),
        void (*on_reasoning)(int, const char *, void *),
        void *ud, SrvChatResult **out, int max_out) {

    if (!out || max_out <= 0) return -1;
    if (n_choices < 1) n_choices = 1;

    std::string body;
    try { body = build_request(messages_json, tools_json, samp, n_predict, n_choices); }
    catch (...) { return -1; }   /* caller passed invalid messages/tools JSON */

    /* Write the request to a temp file (avoids quoting a big body on the argv). */
    char reqpath[] = "/tmp/basi_srvchat_XXXXXX";
    int rfd = mkstemp(reqpath);
    if (rfd < 0) return -1;
    { FILE *rf = fdopen(rfd, "w"); if (!rf) { close(rfd); unlink(reqpath); return -1; }
      fwrite(body.data(), 1, body.size(), rf); fclose(rf); }

    char cmd[512];
    snprintf(cmd, sizeof cmd,
        "curl -N -s -X POST http://127.0.0.1:%d/v1/chat/completions "
        "-H 'Content-Type: application/json' --data-binary @%s", port, reqpath);

    FILE *p = popen(cmd, "r");
    if (!p) { unlink(reqpath); return -1; }

    /* Grown on demand: the server assigns each choice a stable `index` in every
     * chunk, so deltas route by that rather than by arrival order. */
    std::vector<AccChoice> choices(1);
    RespStats st;

    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, p)) != -1) {
        char *j = strstr(line, "data:");
        if (!j) continue;
        j += 5; while (*j == ' ') j++;
        if (strncmp(j, "[DONE]", 6) == 0) break;

        json d;
        try { d = json::parse(j); } catch (...) { continue; }

        if (d.contains("choices") && d["choices"].is_array()) {
            for (const json &ch : d["choices"]) {
                size_t ci = ch.value("index", 0);
                if (ci >= choices.size()) choices.resize(ci + 1);
                AccChoice &acc = choices[ci];

                if (ch.contains("delta") && ch["delta"].is_object()) {
                    const json &delta = ch["delta"];
                    if (delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                        std::string r = delta["reasoning_content"].get<std::string>();
                        if (!r.empty()) {
                            acc.reasoning += r;
                            if (on_reasoning) on_reasoning((int) ci, r.c_str(), ud);
                        }
                    }
                    if (delta.contains("content") && delta["content"].is_string()) {
                        std::string c = delta["content"].get<std::string>();
                        if (!c.empty()) {
                            acc.content += c;
                            if (on_content) on_content((int) ci, c.c_str(), ud);
                        }
                    }
                    if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
                        for (const json &tc : delta["tool_calls"]) {
                            size_t idx = tc.value("index", 0);
                            if (idx >= acc.tools.size()) acc.tools.resize(idx + 1);
                            AccTool &at = acc.tools[idx];
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
                    acc.finish = ch["finish_reason"].get<std::string>();
            }
        }
        if (d.contains("usage") && d["usage"].is_object()) {
            st.prompt_tokens     = d["usage"].value("prompt_tokens", 0);
            st.completion_tokens = d["usage"].value("completion_tokens", 0);
        }
        if (d.contains("timings") && d["timings"].is_object()) {
            const json &t = d["timings"];
            st.tps          = t.value("predicted_per_second", 0.0);
            st.prompt_tps   = t.value("prompt_per_second", 0.0);
            st.prompt_n     = t.value("prompt_n", 0);
            st.prompt_ms    = t.value("prompt_ms", 0.0);
            st.predicted_ms = t.value("predicted_ms", 0.0);
        }
    }
    free(line);
    pclose(p);
    unlink(reqpath);

    /* The server reports one usage block for the whole response, so per-choice
     * completion_tokens is not available; every choice carries the same figure.
     * Since best-of-N keeps exactly one choice, that is the count that actually
     * enters the conversation, which is what the ctx meter needs. */
    int filled = 0;
    for (size_t i = 0; i < choices.size() && filled < max_out; i++) {
        SrvChatResult *r = finalize_choice(choices[i], st);
        if (!r) break;
        out[filled++] = r;
    }
    return filled;
}

/* Single-choice entry point — unchanged signature, so the five existing call
 * sites (model.c:155 and the selftests) need no edit. */
struct OneShim { void (*c)(const char *, void *); void (*r)(const char *, void *); void *ud; };
static void shim_content(int, const char *chunk, void *ud) {
    OneShim *s = (OneShim *) ud; if (s->c) s->c(chunk, s->ud);
}
static void shim_reason(int, const char *chunk, void *ud) {
    OneShim *s = (OneShim *) ud; if (s->r) s->r(chunk, s->ud);
}

extern "C" SrvChatResult *srvchat_complete(
        int port, const char *messages_json, const char *tools_json,
        const SrvSampling *samp, int n_predict,
        void (*on_content)(const char *, void *),
        void (*on_reasoning)(const char *, void *),
        void *ud) {

    OneShim shim{on_content, on_reasoning, ud};
    SrvChatResult *one = nullptr;
    int got = srvchat_complete_n(port, messages_json, tools_json, samp, n_predict, 1,
                                 on_content   ? shim_content : nullptr,
                                 on_reasoning ? shim_reason  : nullptr,
                                 &shim, &one, 1);
    return got == 1 ? one : nullptr;
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

/* Locate the float vector inside one response object, flattening the one level
 * of nesting llama-server uses ("embedding":[[...]]). Shared by both embed paths. */
static const json *embed_vec_of(const json &obj) {
    if (!obj.is_object() || !obj.contains("embedding")) return nullptr;
    const json *emb = &obj["embedding"];
    if (!emb->is_array() || emb->empty()) return nullptr;
    if ((*emb)[0].is_array()) emb = &(*emb)[0];
    return emb->is_array() ? emb : nullptr;
}

extern "C" int srvchat_embed_batch(int port, const char **texts, int n,
                                   float *out, int max_dim) {
    if (!texts || !out || n <= 0 || max_dim <= 0) return -1;

    std::string body;
    try {
        json arr = json::array();
        for (int i = 0; i < n; i++) arr.push_back(texts[i] ? texts[i] : "");
        json req; req["content"] = std::move(arr);
        /* Tool output is arbitrary fetched bytes; never let one bad sequence
         * abort serialization (same reasoning as the chat path). */
        body = req.dump(-1, ' ', false, json::error_handler_t::replace);
    } catch (...) { return -1; }

    char reqpath[] = "/tmp/basi_srvembb_XXXXXX";
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
    { char buf[65536]; size_t k; while ((k = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, k); }
    pclose(p);
    unlink(reqpath);

    try {
        json d = json::parse(resp);
        if (!d.is_array() || d.empty()) return -1;      /* an error object lands here too */

        int dim = 0;
        int filled = 0;
        for (const json &obj : d) {
            const json *vec = embed_vec_of(obj);
            if (!vec) continue;
            /* Route by the server's index; arrival order is not guaranteed. */
            int idx = obj.value("index", -1);
            if (idx < 0 || idx >= n) continue;
            int cnt = (int) vec->size();
            if (cnt > max_dim) cnt = max_dim;
            if (dim == 0) dim = cnt;
            else if (cnt != dim) return -1;             /* ragged: refuse rather than corrupt */
            float *dst = out + (size_t) idx * (size_t) max_dim;
            for (int i = 0; i < cnt; i++) dst[i] = (*vec)[i].get<float>();
            filled++;
        }
        /* A partial batch would silently leave stale/zero vectors behind, which
         * corrupts retrieval quality invisibly — fail loudly instead. */
        if (filled != n || dim == 0) return -1;
        return dim;
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

    /* BASI_SRV_CHAT_N>1 exercises the best-of-N path: one prefill, N sampled
     * continuations, deltas routed by choice index. Uses temp 0.4 (the real agent
     * default) because temp 0 forces top_k=1 above and every choice would be
     * identical, which would make the divergence check meaningless. */
    int n_choices = 1;
    if (const char *e = getenv("BASI_SRV_CHAT_N"); e && *e && atoi(e) > 1) n_choices = atoi(e);
    if (n_choices > 1) {
        samp.temperature = 0.4;
        fprintf(stderr, "[srvchat] best-of-N mode: n=%d (one prefill, %d samples)\n",
                n_choices, n_choices);
        std::vector<SrvChatResult *> outs((size_t) n_choices, nullptr);
        int got = srvchat_complete_n(port, messages, tools, &samp, 200, n_choices,
                                     nullptr, nullptr, nullptr, outs.data(), n_choices);
        fprintf(stderr, "\n---\n[srvchat] choices returned: %d (asked %d)\n", got, n_choices);
        int distinct = 0;
        for (int i = 0; i < got; i++) {
            if (!outs[i]) { fprintf(stderr, "  [%d] NULL\n", i); continue; }
            const char *c = outs[i]->content ? outs[i]->content : "";
            bool dup = false;
            for (int k = 0; k < i; k++)
                if (outs[k] && outs[k]->content && strcmp(outs[k]->content, c) == 0) { dup = true; break; }
            if (!dup) distinct++;
            fprintf(stderr, "  [%d] %zu bytes  tool_calls=%d  finish=%s  %.60s\n",
                    i, strlen(c), outs[i]->n_tool_calls,
                    outs[i]->finish_reason ? outs[i]->finish_reason : "(none)", c);
        }
        fprintf(stderr, "[srvchat] distinct=%d/%d  prompt_tokens=%d (paid ONCE)\n",
                distinct, got, got > 0 && outs[0] ? outs[0]->prompt_tokens : 0);
        for (int i = 0; i < got; i++) srvchat_free(outs[i]);
        srvgen_kill(pid);
        fprintf(stderr, "[srvchat] server killed. done.\n");
        return;
    }

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
