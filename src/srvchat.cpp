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

#include "backend.h"   /* which llama-server binary to spawn */

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

/* ── Remote (hosted) endpoint ────────────────────────────────────────────────
 * See srvchat.h. Set once at startup by main.c; read-only afterwards. */
static std::string g_rem_url, g_rem_key, g_rem_model;
static bool        g_rem_on = false;

extern "C" int srvchat_set_remote(const char *base_url, const char *api_key, const char *model) {
    if (!base_url || !*base_url || !model || !*model) return -1;
    std::string u = base_url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    if (u.empty()) return -1;
    g_rem_url   = u;
    g_rem_key   = api_key ? api_key : "";
    g_rem_model = model;
    g_rem_on    = true;
    return 0;
}
extern "C" int   srvchat_remote_active(void)      { return g_rem_on ? 1 : 0; }
extern "C" const char *srvchat_remote_model(void) { return g_rem_on ? g_rem_model.c_str() : nullptr; }
extern "C" const char *srvchat_remote_base(void)  { return g_rem_on ? g_rem_url.c_str()   : nullptr; }

/* Single-quote for /bin/sh. The base URL comes from a config/env/CLI value, so it
 * reaches popen() as untrusted text: without this, one apostrophe in it would end
 * the quoted string and the rest would run as a command. */
static std::string shq(const std::string &s) {
    std::string q = "'";
    for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
    return q + "'";
}

/* Write the bearer token to a fresh 0600 file in curl's -K config format and
 * return its path (empty when there is no key, or on failure — the caller then
 * simply sends no Authorization header). The key never goes on a command line:
 * argv is world-readable through `ps`. Caller unlinks. */
static std::string write_auth_file(void) {
    if (g_rem_key.empty()) return "";
    char path[] = "/tmp/basi_srvauth_XXXXXX";
    int fd = mkstemp(path);                     /* mkstemp creates it 0600 */
    if (fd < 0) return "";
    FILE *f = fdopen(fd, "w");
    if (!f) { close(fd); unlink(path); return ""; }
    std::string k;                              /* -K syntax: quoted, backslash-escaped */
    for (char c : g_rem_key) { if (c == '"' || c == '\\') k += '\\'; k += c; }
    fprintf(f, "header = \"Authorization: Bearer %s\"\n", k.c_str());
    fclose(f);
    return path;
}

/* Ask the provider what context window the configured model has, over the
 * OpenAI-standard GET <base>/models. Returns the length, or 0 when the endpoint
 * is absent, the model is not listed, or no length is reported — the caller then
 * keeps its own default. Worth one short request at startup: the alternative is a
 * hardcoded number that is wrong for most models, and being wrong LOW silently
 * compacts conversations the provider would have accepted whole. */
/* One authenticated GET, body returned as a string (empty on failure). */
static std::string http_get(const std::string &url) {
    std::string auth = write_auth_file();
    std::string cmd = "curl -s --connect-timeout 5 --max-time 10 ";
    if (!auth.empty()) cmd += "-K " + shq(auth) + " ";
    cmd += shq(url);

    FILE *p = popen(cmd.c_str(), "r");
    if (!p) { if (!auth.empty()) unlink(auth.c_str()); return ""; }
    std::string resp;
    { char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, p)) > 0) resp.append(buf, n); }
    pclose(p);
    if (!auth.empty()) unlink(auth.c_str());
    return resp;
}

/* Pull a positive context length out of one model object, across the field names
 * different providers use for the same number. */
static int ctxlen_of(const json &m) {
    if (!m.is_object()) return 0;
    for (const char *k : { "context_length", "context_window",
                           "max_context_length", "contextLength" })
        if (m.contains(k) && m[k].is_number_integer()) {
            long v = m[k].get<long>();
            if (v > 0 && v < (1L << 31)) return (int) v;
        }
    return 0;
}

extern "C" int srvchat_remote_context_length(void) {
    if (!g_rem_on) return 0;

    /* (1) The OpenAI-standard listing. Works anywhere it is implemented. */
    try {
        json d = json::parse(http_get(g_rem_url + "/models"));
        if (d.contains("data") && d["data"].is_array())
            for (const json &m : d["data"])
                if (m.is_object() && m.value("id", std::string()) == g_rem_model) {
                    int v = ctxlen_of(m);
                    if (v > 0) return v;
                    break;
                }
    } catch (...) { /* absent or not JSON — fall through */ }

    /* (2) Fireworks' listing is INCOMPLETE: measured 2026-07-27 it returned five
     * models, omitting kimi-k3 — which serves requests fine and really has a 1M
     * window. Its control-plane API does know, so ask that rather than fall back
     * to a default that is wrong by 8x. The model id is already the full
     * "accounts/<acct>/models/<name>" path this endpoint expects. */
    if (g_rem_url.find("fireworks.ai") != std::string::npos) {
        size_t host_end = g_rem_url.find("/", g_rem_url.find("://") + 3);
        std::string host = (host_end == std::string::npos) ? g_rem_url
                                                           : g_rem_url.substr(0, host_end);
        try {
            json d = json::parse(http_get(host + "/v1/" + g_rem_model));
            int v = ctxlen_of(d);
            if (v > 0) return v;
        } catch (...) { /* fall through to the caller's default */ }
    }
    return 0;
}

/* Assemble the request body: caller messages/tools + streaming + sampling.
 * n_choices>1 asks the server for N independent continuations of ONE prefill —
 * measured at 6720 prompt tokens: n=4 returns 4 answers in 11.5s with
 * cached_tokens=6716, vs 33s for 4 separate concurrent requests that each
 * re-prefill the shared prefix. Streaming preserves per-choice indices. */
static std::string build_request(const char *messages_json, const char *tools_json,
                                 const SrvSampling *samp, int n_predict, int n_choices) {
    const bool remote = g_rem_on;
    json req;
    /* A hosted endpoint routes on `model`; llama-server serves the one it loaded
     * and has no use for the field. */
    if (remote) req["model"] = g_rem_model;
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
    if (no_think && !remote) {
        req["chat_template_kwargs"] = json{{"enable_thinking", false}};
        req["reasoning_budget"] = 0;   /* llama-server native no-think, if supported */
    } else if (no_think) {
        /* Hosted endpoints reject the llama-server extensions above — Fireworks
         * returns HTTP 400 for chat_template_kwargs — so BASI_NO_THINK used to be
         * silently ignored on the remote path. reasoning_effort is the
         * OpenAI-standard equivalent and is what a hosted reasoning model honours.
         * BASI_API_EXTRA_JSON is merged after this and can still override it. */
        req["reasoning_effort"] = "none";
    }
    if (n_predict > 0) req["max_tokens"] = n_predict;
    if (samp) {
        if (samp->temperature >= 0)   req["temperature"] = samp->temperature;
        if (samp->top_p >= 0 && samp->top_p < 1.0) req["top_p"] = samp->top_p;
        if (samp->seed >= 0)                       req["seed"] = samp->seed;
        /* The rest are llama-server sampler knobs, not OpenAI schema. top_k=1 as a
         * greedy stand-in is llama-specific too — a hosted provider gets
         * temperature=0 and decides for itself. Providers that DO accept these
         * (Fireworks takes top_k) can have them back via BASI_API_EXTRA_JSON. */
        if (!remote) {
            if (samp->temperature == 0.0) req["top_k"] = 1;          /* greedy */
            else if (samp->top_k > 0)     req["top_k"] = samp->top_k;
            if (samp->repeat_penalty > 1.0) {
                req["repeat_penalty"] = samp->repeat_penalty;
                if (samp->repeat_last_n >= 0) req["repeat_last_n"] = samp->repeat_last_n;
            }
            if (samp->min_p >= 0) req["min_p"] = samp->min_p;
        }
    }
    if (!remote) req["cache_prompt"] = true;   /* llama-server KV reuse; no remote analogue */

    /* Escape hatch for provider-specific fields BASI has no opinion on
     * (top_k, presence_penalty, reasoning_effort, provider routing…). Merged last
     * so it can also OVERRIDE anything set above. Invalid JSON is ignored with a
     * warning rather than silently dropping the whole turn. */
    if (remote) {
        const char *extra = getenv("BASI_API_EXTRA_JSON");
        if (extra && *extra) {
            try {
                json e = json::parse(extra);
                if (e.is_object()) for (auto &kv : e.items()) req[kv.key()] = kv.value();
                else fprintf(stderr, "\033[33m[api] BASI_API_EXTRA_JSON is not a JSON object — ignored\033[0m\n");
            } catch (...) {
                fprintf(stderr, "\033[33m[api] BASI_API_EXTRA_JSON is not valid JSON — ignored\033[0m\n");
            }
        }
    }
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
    int    cached_tokens = 0, reasoning_tokens = 0;
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
    res->cached_tokens     = st.cached_tokens;
    res->reasoning_tokens  = st.reasoning_tokens;
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

    /* Local: the llama-server we spawned. Remote: the configured hosted endpoint,
     * with the bearer token handed to curl through a 0600 config file rather than
     * on the argv — a command line is world-readable in `ps`. */
    std::string url, cmd, authpath;
    if (g_rem_on) {
        url = g_rem_url + "/chat/completions";
        authpath = write_auth_file();
        if (authpath.empty() && !g_rem_key.empty()) { unlink(reqpath); return -1; }
    } else {
        char b[96];
        snprintf(b, sizeof b, "http://127.0.0.1:%d/v1/chat/completions", port);
        url = b;
    }

    cmd = "curl -N -s -S ";
    /* A hosted endpoint can hang or stall in ways a loopback socket cannot; an
     * agent loop that blocks forever on one turn is worse than a failed turn. */
    if (g_rem_on) cmd += "--connect-timeout 20 --max-time 1800 ";
    if (!authpath.empty()) cmd += "-K " + shq(authpath) + " ";
    cmd += "-X POST " + shq(url) +
           " -H 'Content-Type: application/json' --data-binary @" + shq(reqpath);

    FILE *p = popen(cmd.c_str(), "r");
    if (!p) { unlink(reqpath); if (!authpath.empty()) unlink(authpath.c_str()); return -1; }

    /* Grown on demand: the server assigns each choice a stable `index` in every
     * chunk, so deltas route by that rather than by arrival order. */
    std::vector<AccChoice> choices(1);
    RespStats st;

    /* Anything that is not an SSE frame. On a rejected request the body is a plain
     * JSON error object, which the `data:` filter below would drop silently — the
     * turn would then come back blank with no reason given. Kept so the failure can
     * be reported verbatim. */
    std::string nonsse;
    bool saw_data = false;

    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, p)) != -1) {
        char *j = strstr(line, "data:");
        if (!j) {
            if (nonsse.size() < 4096) nonsse.append(line, (size_t) len);
            continue;
        }
        saw_data = true;
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
            const json &u = d["usage"];
            st.prompt_tokens     = u.value("prompt_tokens", 0);
            st.completion_tokens = u.value("completion_tokens", 0);
            /* OpenAI-schema breakdowns. Present on hosted providers (Fireworks sends
             * both), absent on llama-server — hence the guarded reads. */
            if (u.contains("prompt_tokens_details") && u["prompt_tokens_details"].is_object())
                st.cached_tokens = u["prompt_tokens_details"].value("cached_tokens", 0);
            if (u.contains("completion_tokens_details") && u["completion_tokens_details"].is_object())
                st.reasoning_tokens = u["completion_tokens_details"].value("reasoning_tokens", 0);
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
    if (!authpath.empty()) unlink(authpath.c_str());

    /* No SSE at all means the request never became a completion: an auth failure, an
     * unknown model id, a rate limit, a bad field. Say which, instead of handing the
     * caller an empty turn. Trimmed — provider error bodies can be long. */
    if (!saw_data && !nonsse.empty()) {
        std::string msg = nonsse;
        try {                                  /* pull out the human-readable part */
            json e = json::parse(nonsse);
            if (e.contains("error")) {
                const json &er = e["error"];
                if (er.is_string())            msg = er.get<std::string>();
                else if (er.is_object() && er.contains("message") && er["message"].is_string())
                    msg = er["message"].get<std::string>();
                else                           msg = er.dump();
            }
        } catch (...) { /* not JSON — show the raw body */ }
        while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
        if (msg.size() > 600) msg.resize(600);
        fprintf(stderr, "\033[1;31m[%s] request failed:\033[0m %s\n",
                g_rem_on ? "api" : "server", msg.c_str());
    }

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
    const char *server_bin = backend_active()->server_bin;   /* honors $BASI_SERVER_BIN */
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
