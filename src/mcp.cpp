/* mcp.cpp — Model Context Protocol client. See mcp.h for the design.
 *
 * Self-contained on purpose: nlohmann/json (vendored, header-only) plus POSIX
 * for the stdio transport and the `curl` BINARY for HTTP — the same choices the
 * rest of BASI already made, so MCP support adds no link-time dependency and no
 * new build step. */
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "nlohmann/json.hpp"
#include "mcp.h"
#include "tooldefs.h"     /* basi_tooldefs_set_extra */

using json = nlohmann::json;

/* The revision this client prefers. Everything else is negotiated: a modern
 * server that does not support it names what it does support, and a legacy
 * server never sees it at all. */
#define MCP_MODERN_VERSION  "2026-07-28"
/* Preferred legacy revision, offered in `initialize`. A legacy server answers
 * with the version IT chose, which is the one we then use. */
#define MCP_LEGACY_VERSION  "2025-11-25"
#define MCP_CLIENT_NAME     "basi-cli"
#define MCP_CLIENT_VERSION  "0.1"

namespace {

/* ── small helpers ──────────────────────────────────────────────────── */

/* Single-quote for /bin/sh. Every value that reaches a curl command line comes
 * from a user config file, so it is untrusted text: one apostrophe would
 * otherwise end the quoted string and run the rest as a command. */
std::string shq(const std::string &s) {
    std::string q = "'";
    for (char c : s) { if (c == '\'') q += "'\\''"; else q += c; }
    return q + "'";
}

long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    int n = atoi(v);
    return n > 0 ? n : dflt;
}

std::string b64(const std::string &in) {
    static const char *T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
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

/* Expand ${VAR} and ${VAR:-default} against the environment. An MCP config is
 * shared and often committed; the token belongs in the environment, and without
 * expansion the only way to pass one is to paste it into the file. */
std::string expand_env(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ) {
        if (s[i] == '$' && i + 1 < s.size() && s[i+1] == '{') {
            size_t end = s.find('}', i + 2);
            if (end == std::string::npos) { out += s[i++]; continue; }
            std::string body = s.substr(i + 2, end - i - 2);
            std::string name = body, dflt;
            size_t sep = body.find(":-");
            if (sep != std::string::npos) { name = body.substr(0, sep); dflt = body.substr(sep + 2); }
            const char *v = getenv(name.c_str());
            out += (v && *v) ? v : dflt;
            i = end + 1;
        } else {
            out += s[i++];
        }
    }
    return out;
}

/* Tool names reach a chat template and a server-derived grammar, so keep them to
 * the character set every model's format tolerates. MCP itself allows dots. */
std::string sanitize_name(const std::string &s) {
    std::string o;
    for (char c : s) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-';
        o += ok ? c : '_';
    }
    if (o.empty()) o = "x";
    return o;
}

std::string lower(std::string s) {
    for (char &c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}

/* ── data model ─────────────────────────────────────────────────────── */

struct HeaderParam {          /* an x-mcp-header mirror: where to read, what to send */
    std::vector<std::string> path;   /* chain of `properties` keys from the root */
    std::string header;              /* the {Name} in Mcp-Param-{Name} */
};

struct McpTool {
    std::string full;         /* mcp__<server>__<tool> — what the model calls */
    std::string remote;       /* the name on the wire */
    std::string server;
    std::string description;
    std::string schema;       /* inputSchema, verbatim JSON */
    int         readonly = -1;
    std::vector<HeaderParam> hdr_params;
};

enum class Transport { Stdio, Http };
enum class Era { Unknown, Modern, Legacy };

struct Server {
    std::string name;
    Transport   transport = Transport::Stdio;
    /* stdio */
    std::string command, cwd;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    pid_t pid = -1;
    int   in_fd = -1, out_fd = -1;
    std::string rdbuf;
    /* http */
    std::string url;
    std::map<std::string, std::string> headers;
    std::string session_id;          /* legacy HTTP only; modern has no sessions */
    /* negotiated */
    Era         era = Era::Unknown;
    std::string version;
    std::string server_name, server_version, instructions;
    json        capabilities = json::object();
    /* state */
    bool        enabled = true;
    bool        connected = false;
    std::string error;
    int         next_id = 1;
    int         timeout_ms = 0;      /* 0 = use the global default */
    int         n_tools = 0;
};

struct Registry {
    std::vector<Server>   servers;
    std::vector<McpTool>  tools;
    std::vector<BasiToolDef> defs;   /* points into `tools`; rebuilt whenever it changes */
    bool configured = false;
    bool disabled = false;          /* BASI_MCP=0 / --no-mcp: never even looked */
    bool sigpipe_ignored = false;
};

Registry g_reg;

int default_timeout_ms(void) { return env_int("BASI_MCP_TIMEOUT_MS", 60000); }
int probe_timeout_ms(void)   { return env_int("BASI_MCP_PROBE_MS", 5000); }

McpTool *find_tool(const char *name) {
    if (!name) return nullptr;
    for (auto &t : g_reg.tools) if (t.full == name) return &t;
    return nullptr;
}

Server *find_server(const std::string &name) {
    for (auto &s : g_reg.servers) if (s.name == name) return &s;
    return nullptr;
}

/* ── config ─────────────────────────────────────────────────────────── */

std::string read_file(const std::string &path, bool *found) {
    if (found) *found = false;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return "";
    if (found) *found = true;
    std::string out;
    char buf[8192]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}

std::string home_config_path(void) {
    const char *xdg = getenv("XDG_CONFIG_HOME");
    if (xdg && *xdg) return std::string(xdg) + "/basi-cli/mcp.json";
    const char *home = getenv("HOME");
    if (home && *home) return std::string(home) + "/.config/basi-cli/mcp.json";
    return "";
}

/* One config file → server entries, merged into `out` by name (a later file
 * overrides an earlier one, so a project can override the user's global entry).
 * Accepts both `mcpServers` (the de-facto convention) and `servers`. */
void load_config_file(const std::string &path, std::vector<Server> &out, std::string &warn) {
    bool found = false;
    std::string text = read_file(path, &found);
    if (!found) return;

    json j;
    try { j = json::parse(text, nullptr, true, /*ignore_comments=*/true); }
    catch (const std::exception &e) {
        warn += "  " + path + ": not valid JSON (" + e.what() + ")\n";
        return;
    }
    const json *m = nullptr;
    if (j.contains("mcpServers") && j["mcpServers"].is_object())   m = &j["mcpServers"];
    else if (j.contains("servers") && j["servers"].is_object())    m = &j["servers"];
    else if (j.is_object() && !j.contains("mcpServers"))           m = &j;   /* bare map */
    if (!m) return;

    for (auto it = m->begin(); it != m->end(); ++it) {
        const json &e = it.value();
        if (!e.is_object()) continue;

        Server s;
        s.name = sanitize_name(it.key());

        if (e.contains("disabled") && e["disabled"].is_boolean() && e["disabled"].get<bool>())
            s.enabled = false;
        if (e.contains("enabled") && e["enabled"].is_boolean() && !e["enabled"].get<bool>())
            s.enabled = false;
        if (e.contains("timeoutMs") && e["timeoutMs"].is_number_integer())
            s.timeout_ms = e["timeoutMs"].get<int>();

        std::string url;
        for (const char *k : {"url", "endpoint", "serverUrl"})
            if (e.contains(k) && e[k].is_string()) { url = expand_env(e[k].get<std::string>()); break; }

        if (!url.empty()) {
            s.transport = Transport::Http;
            s.url = url;
            if (e.contains("headers") && e["headers"].is_object())
                for (auto h = e["headers"].begin(); h != e["headers"].end(); ++h)
                    if (h.value().is_string())
                        s.headers[h.key()] = expand_env(h.value().get<std::string>());
        } else if (e.contains("command") && e["command"].is_string()) {
            s.transport = Transport::Stdio;
            s.command = expand_env(e["command"].get<std::string>());
            if (e.contains("args") && e["args"].is_array())
                for (const json &a : e["args"])
                    if (a.is_string()) s.args.push_back(expand_env(a.get<std::string>()));
            if (e.contains("env") && e["env"].is_object())
                for (auto v = e["env"].begin(); v != e["env"].end(); ++v)
                    if (v.value().is_string())
                        s.env[v.key()] = expand_env(v.value().get<std::string>());
            if (e.contains("cwd") && e["cwd"].is_string())
                s.cwd = expand_env(e["cwd"].get<std::string>());
        } else {
            warn += "  " + s.name + ": entry has neither `command` (stdio) nor `url` (http)\n";
            continue;
        }

        bool replaced = false;
        for (auto &o : out) if (o.name == s.name) { o = s; replaced = true; break; }
        if (!replaced) out.push_back(s);
    }
}

std::vector<Server> load_config(const char *extra, std::string &warn) {
    std::vector<Server> out;
    std::string hp = home_config_path();
    if (!hp.empty()) load_config_file(hp, out, warn);
    load_config_file(".basi/mcp.json", out, warn);
    load_config_file(".mcp.json", out, warn);      /* project convention */
    const char *envcfg = getenv("BASI_MCP_CONFIG");
    if (envcfg && *envcfg) load_config_file(envcfg, out, warn);
    if (extra && *extra)   load_config_file(extra, out, warn);
    return out;
}

/* ── stdio transport ────────────────────────────────────────────────── */

bool stdio_spawn(Server &s) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0) { s.error = "pipe failed"; return false; }
    if (pipe(out_pipe) < 0) { close(in_pipe[0]); close(in_pipe[1]); s.error = "pipe failed"; return false; }

    pid_t pid = fork();
    if (pid < 0) {
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);
        s.error = "fork failed";
        return false;
    }
    if (pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        /* The server MAY log anything to stderr and the client MUST NOT read it
         * as an error signal, so keep it off BASI's terminal. BASI_MCP_DEBUG
         * keeps it, because a server that dies at startup says why only here. */
        const char *dbg = getenv("BASI_MCP_DEBUG");
        if (!dbg) {
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        } else {
            char path[256];
            snprintf(path, sizeof path, "/tmp/basi-mcp-%s.log", s.name.c_str());
            int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, STDERR_FILENO); close(fd); }
        }
        close(in_pipe[0]); close(in_pipe[1]); close(out_pipe[0]); close(out_pipe[1]);

        for (const auto &kv : s.env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
        if (!s.cwd.empty() && chdir(s.cwd.c_str()) != 0) _exit(126);

        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(s.command.c_str()));
        for (auto &a : s.args) argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);
        execvp(s.command.c_str(), argv.data());
        _exit(127);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    s.pid = pid;
    s.in_fd = in_pipe[1];
    s.out_fd = out_pipe[0];
    s.rdbuf.clear();
    return true;
}

void stdio_kill(Server &s) {
    if (s.pid <= 0) return;
    /* The spec's shutdown order: close stdin first — servers SHOULD exit on EOF,
     * and it is the only portable graceful signal — then escalate. */
    if (s.in_fd >= 0) { close(s.in_fd); s.in_fd = -1; }
    for (int i = 0; i < 20; i++) {                 /* up to ~200ms for a clean exit */
        int st;
        pid_t r = waitpid(s.pid, &st, WNOHANG);
        if (r == s.pid) { s.pid = -1; break; }
        if (r < 0) { s.pid = -1; break; }
        struct timespec ts = {0, 10 * 1000 * 1000};
        nanosleep(&ts, nullptr);
    }
    if (s.pid > 0) {
        kill(s.pid, SIGTERM);
        struct timespec ts = {0, 50 * 1000 * 1000};
        nanosleep(&ts, nullptr);
        int st;
        if (waitpid(s.pid, &st, WNOHANG) != s.pid) {
            kill(s.pid, SIGKILL);
            waitpid(s.pid, &st, 0);
        }
        s.pid = -1;
    }
    if (s.out_fd >= 0) { close(s.out_fd); s.out_fd = -1; }
    s.connected = false;
}

bool stdio_write_line(Server &s, const std::string &msg) {
    if (s.in_fd < 0) return false;
    std::string line = msg;
    line += '\n';
    size_t off = 0;
    while (off < line.size()) {
        ssize_t w = write(s.in_fd, line.data() + off, line.size() - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;                 /* EPIPE: the child is gone */
        }
        off += (size_t)w;
    }
    return true;
}

/* One newline-delimited message, or "" on timeout/EOF. Messages MUST NOT contain
 * embedded newlines, so a line IS a message; a single read may still deliver
 * several of them, hence the buffer. */
std::string stdio_read_line(Server &s, long deadline) {
    for (;;) {
        size_t nl = s.rdbuf.find('\n');
        if (nl != std::string::npos) {
            std::string line = s.rdbuf.substr(0, nl);
            s.rdbuf.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            return line;
        }
        long left = deadline - now_ms();
        if (left <= 0) return "";
        if (s.out_fd < 0) return "";
        struct pollfd p = { s.out_fd, POLLIN, 0 };
        int rv = poll(&p, 1, (int)(left > 1000 ? 1000 : left));
        if (rv < 0) { if (errno == EINTR) continue; return ""; }
        if (rv == 0) continue;
        char buf[16384];
        ssize_t r = read(s.out_fd, buf, sizeof buf);
        if (r < 0) { if (errno == EINTR) continue; return ""; }
        if (r == 0) return "";                    /* server closed stdout */
        s.rdbuf.append(buf, (size_t)r);
        if (s.rdbuf.size() > 64u * 1024 * 1024) return "";   /* runaway output */
    }
}

/* ── HTTP transport ─────────────────────────────────────────────────── */

struct HttpReply {
    long status = 0;
    std::string body;
    std::map<std::string, std::string> headers;   /* keys lowercased */
    std::string transport_error;
};

/* A header value must be visible ASCII with no leading/trailing space and must
 * not itself look like the sentinel; anything else travels base64-wrapped. */
std::string encode_header_value(const std::string &v) {
    bool plain = !v.empty();
    for (unsigned char c : v) if (c < 0x21 || c > 0x7e) { plain = false; break; }
    if (plain && v.size() > 9 && v.compare(0, 9, "=?base64?") == 0 &&
        v.compare(v.size() - 2, 2, "?=") == 0) plain = false;
    if (plain) return v;
    return "=?base64?" + b64(v) + "?=";
}

HttpReply http_post(Server &s, const std::string &body,
                    const std::vector<std::pair<std::string, std::string>> &extra_headers,
                    int timeout_ms) {
    HttpReply rep;

    char bodypath[] = "/tmp/basi_mcp_req_XXXXXX";
    int bfd = mkstemp(bodypath);
    if (bfd < 0) { rep.transport_error = "cannot create a temp file for the request"; return rep; }
    { FILE *bf = fdopen(bfd, "w");
      if (!bf) { close(bfd); unlink(bodypath); rep.transport_error = "temp file"; return rep; }
      fwrite(body.data(), 1, body.size(), bf); fclose(bf); }

    char hdrpath[] = "/tmp/basi_mcp_hdr_XXXXXX";
    int hfd = mkstemp(hdrpath);
    if (hfd < 0) { unlink(bodypath); rep.transport_error = "temp file"; return rep; }
    close(hfd);
    char outpath[] = "/tmp/basi_mcp_out_XXXXXX";
    int ofd = mkstemp(outpath);
    if (ofd < 0) { unlink(bodypath); unlink(hdrpath); rep.transport_error = "temp file"; return rep; }
    close(ofd);

    int secs = (timeout_ms + 999) / 1000;
    if (secs < 2) secs = 2;
    char tbuf[64];
    snprintf(tbuf, sizeof tbuf, "--connect-timeout 10 --max-time %d ", secs);

    std::string cmd = "curl -sS -X POST ";
    cmd += tbuf;
    cmd += "-H 'Content-Type: application/json' ";
    /* The client MUST advertise both response shapes: a server picks per request
     * whether to answer with one JSON object or an SSE stream. */
    cmd += "-H 'Accept: application/json, text/event-stream' ";
    for (const auto &kv : s.headers)
        cmd += "-H " + shq(kv.first + ": " + kv.second) + " ";
    for (const auto &kv : extra_headers)
        cmd += "-H " + shq(kv.first + ": " + kv.second) + " ";
    cmd += "-D " + shq(hdrpath) + " -o " + shq(outpath) + " -w '%{http_code}' ";
    cmd += "--data-binary @" + shq(bodypath) + " " + shq(s.url);

    FILE *p = popen(cmd.c_str(), "r");
    if (!p) {
        unlink(bodypath); unlink(hdrpath); unlink(outpath);
        rep.transport_error = "cannot run curl";
        return rep;
    }
    std::string code;
    { char b[256]; size_t n; while ((n = fread(b, 1, sizeof b, p)) > 0) code.append(b, n); }
    int rc = pclose(p);

    rep.status = atol(code.c_str());
    bool found = false;
    rep.body = read_file(outpath, &found);
    std::string hdrs = read_file(hdrpath, &found);
    unlink(bodypath); unlink(hdrpath); unlink(outpath);

    if (rep.status == 0) {
        rep.transport_error = "no HTTP response from " + s.url +
                              " (curl exit " + std::to_string(WEXITSTATUS(rc)) + ")";
        return rep;
    }

    size_t pos = 0;
    while (pos < hdrs.size()) {
        size_t eol = hdrs.find('\n', pos);
        if (eol == std::string::npos) eol = hdrs.size();
        std::string line = hdrs.substr(pos, eol - pos);
        pos = eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = lower(line.substr(0, colon));
        std::string v = line.substr(colon + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        rep.headers[k] = v;      /* a redirect/continue re-sends headers; last wins */
    }
    return rep;
}

/* Pull the JSON-RPC message out of a reply body: either one JSON object, or an
 * SSE stream whose final `data:` frame carries the response. Notifications may
 * precede it, so take the frame that carries `result` or `error`. */
bool parse_http_message(const HttpReply &rep, json &msg, std::string &err) {
    std::string ctype;
    auto it = rep.headers.find("content-type");
    if (it != rep.headers.end()) ctype = lower(it->second);

    if (ctype.find("text/event-stream") != std::string::npos) {
        size_t pos = 0;
        bool got = false;
        while (pos < rep.body.size()) {
            size_t eol = rep.body.find('\n', pos);
            if (eol == std::string::npos) eol = rep.body.size();
            std::string line = rep.body.substr(pos, eol - pos);
            pos = eol + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == ':') continue;       /* keep-alive comment */
            if (line.compare(0, 5, "data:") != 0) continue;
            std::string d = line.substr(5);
            while (!d.empty() && d.front() == ' ') d.erase(0, 1);
            json f;
            try { f = json::parse(d); } catch (...) { continue; }
            if (f.contains("result") || f.contains("error")) { msg = f; got = true; }
        }
        if (!got) { err = "SSE stream carried no JSON-RPC response"; return false; }
        return true;
    }

    if (rep.body.empty()) { err = "empty response body (HTTP " + std::to_string(rep.status) + ")"; return false; }
    try { msg = json::parse(rep.body); }
    catch (...) {
        std::string snip = rep.body.substr(0, 200);
        err = "HTTP " + std::to_string(rep.status) + ", body is not JSON: " + snip;
        return false;
    }
    return true;
}

/* ── JSON-RPC ───────────────────────────────────────────────────────── */

struct RpcResult {
    bool ok = false;              /* a `result` came back */
    json result;
    bool is_error = false;        /* a JSON-RPC `error` came back */
    int  code = 0;
    std::string message;
    json data;
    std::string transport_error;  /* nothing came back at all */

    std::string describe() const {
        if (!transport_error.empty()) return transport_error;
        if (is_error) {
            std::string m = "server error " + std::to_string(code);
            if (!message.empty()) m += ": " + message;
            return m;
        }
        return "no result";
    }
};

/* Modern `_meta`: protocolVersion and clientCapabilities are REQUIRED on every
 * request (a request missing either is malformed and MUST be rejected -32602);
 * clientInfo is a SHOULD. BASI implements no client features — no sampling, no
 * roots, no elicitation — so the capability object is deliberately empty, and a
 * server that needs one gets a clean -32021 instead of a hang. */
json modern_meta(const Server &s) {
    json m;
    m["io.modelcontextprotocol/protocolVersion"] = s.version.empty() ? MCP_MODERN_VERSION : s.version;
    m["io.modelcontextprotocol/clientInfo"] = { {"name", MCP_CLIENT_NAME}, {"version", MCP_CLIENT_VERSION} };
    m["io.modelcontextprotocol/clientCapabilities"] = json::object();
    return m;
}

RpcResult rpc_call(Server &s, const std::string &method, json params, int timeout_ms,
                   const std::vector<std::pair<std::string, std::string>> &param_headers =
                       std::vector<std::pair<std::string, std::string>>()) {
    RpcResult out;
    if (timeout_ms <= 0) timeout_ms = s.timeout_ms > 0 ? s.timeout_ms : default_timeout_ms();

    int id = s.next_id++;
    if (!params.is_object()) params = json::object();
    if (s.era == Era::Modern) params["_meta"] = modern_meta(s);

    json req = { {"jsonrpc", "2.0"}, {"id", id}, {"method", method} };
    req["params"] = params;
    std::string body;
    try { body = req.dump(-1, ' ', false, json::error_handler_t::replace); }
    catch (...) { out.transport_error = "cannot serialize the request"; return out; }

    json msg;

    if (s.transport == Transport::Stdio) {
        if (!stdio_write_line(s, body)) {
            out.transport_error = "the server process is not accepting input (it exited?)";
            return out;
        }
        long deadline = now_ms() + timeout_ms;
        for (;;) {
            std::string line = stdio_read_line(s, deadline);
            if (line.empty()) {
                out.transport_error = "no response within " + std::to_string(timeout_ms) + "ms";
                return out;
            }
            json m;
            try { m = json::parse(line); } catch (...) { continue; }   /* not MCP; drop */
            /* A server MUST NOT write requests to stdout, so anything carrying a
             * `method` is a notification or a malformed request — never our answer.
             * Checking this BEFORE the id match matters: a server that echoes its
             * input (or a misconfigured `cat`) would otherwise hand back our own
             * request, whose id matches perfectly, and be read as a reply. */
            if (m.contains("method")) continue;
            if (!m.contains("id") || !m["id"].is_number_integer()) continue;
            if (m["id"].get<int>() != id) continue;                    /* someone else's */
            msg = m;
            break;
        }
    } else {
        std::vector<std::pair<std::string, std::string>> h;
        /* Mcp-Method / Mcp-Name / MCP-Protocol-Version are REQUIRED for compliance
         * from 2026-07-28, and a mismatch against the body is a -32020. The version
         * header has been required since 2025-06-18, so a legacy server gets it too
         * once a version is negotiated. */
        h.push_back({"Mcp-Method", method});
        if (!s.version.empty()) h.push_back({"MCP-Protocol-Version", s.version});
        else if (s.era == Era::Modern) h.push_back({"MCP-Protocol-Version", MCP_MODERN_VERSION});
        if (params.contains("name") && params["name"].is_string())
            h.push_back({"Mcp-Name", encode_header_value(params["name"].get<std::string>())});
        else if (params.contains("uri") && params["uri"].is_string())
            h.push_back({"Mcp-Name", encode_header_value(params["uri"].get<std::string>())});
        for (const auto &kv : param_headers) h.push_back(kv);
        if (!s.session_id.empty()) h.push_back({"Mcp-Session-Id", s.session_id});

        HttpReply rep = http_post(s, body, h, timeout_ms);
        if (!rep.transport_error.empty()) { out.transport_error = rep.transport_error; return out; }

        auto sid = rep.headers.find("mcp-session-id");
        if (sid != rep.headers.end() && !sid->second.empty()) s.session_id = sid->second;

        std::string perr;
        if (!parse_http_message(rep, msg, perr)) { out.transport_error = perr; return out; }
    }

    if (msg.contains("error") && msg["error"].is_object()) {
        out.is_error = true;
        const json &e = msg["error"];
        if (e.contains("code") && e["code"].is_number_integer()) out.code = e["code"].get<int>();
        if (e.contains("message") && e["message"].is_string()) out.message = e["message"].get<std::string>();
        if (e.contains("data")) out.data = e["data"];
        return out;
    }
    if (msg.contains("result")) { out.ok = true; out.result = msg["result"]; return out; }
    out.transport_error = "response carried neither result nor error";
    return out;
}

bool rpc_notify(Server &s, const std::string &method, const json &params) {
    json n = { {"jsonrpc", "2.0"}, {"method", method}, {"params", params} };
    std::string body;
    try { body = n.dump(); } catch (...) { return false; }
    if (s.transport == Transport::Stdio) return stdio_write_line(s, body);
    /* A legacy HTTP server binds the handshake to the session it just minted, so
     * `notifications/initialized` MUST carry the id it handed back — without it
     * the notification is orphaned and the server never leaves the uninitialized
     * state, which then fails every later request for no visible reason. */
    std::vector<std::pair<std::string, std::string>> h;
    h.push_back({"Mcp-Method", method});
    if (!s.version.empty()) h.push_back({"MCP-Protocol-Version", s.version});
    if (!s.session_id.empty()) h.push_back({"Mcp-Session-Id", s.session_id});
    HttpReply rep = http_post(s, body, h, 10000);
    return rep.transport_error.empty();
}

/* ── era detection + connect ────────────────────────────────────────── */

/* Choose a version we can speak from what the server offers. Prefer our own; a
 * server that lists only older modern revisions still gets a modern request
 * because the shape is identical — only the string differs. */
std::string pick_version(const json &list) {
    if (!list.is_array()) return "";
    for (const json &v : list)
        if (v.is_string() && v.get<std::string>() == MCP_MODERN_VERSION) return MCP_MODERN_VERSION;
    for (const json &v : list)
        if (v.is_string()) {
            std::string s = v.get<std::string>();
            if (s >= "2026-01-01") return s;   /* date-ordered: any modern revision */
        }
    return "";
}

/* The legacy handshake, for a server that predates per-request metadata. */
bool legacy_initialize(Server &s) {
    s.era = Era::Legacy;
    s.version = MCP_LEGACY_VERSION;
    json params = {
        {"protocolVersion", MCP_LEGACY_VERSION},
        {"capabilities", json::object()},
        {"clientInfo", { {"name", MCP_CLIENT_NAME}, {"version", MCP_CLIENT_VERSION} }},
    };
    s.version.clear();                       /* no version header until it answers */
    RpcResult r = rpc_call(s, "initialize", params, probe_timeout_ms() * 2);
    if (!r.ok && r.is_error) {
        /* Some servers reject a version they do not know rather than answering
         * with one they do. 2024-11-05 is the floor every implementation shipped. */
        params["protocolVersion"] = "2024-11-05";
        r = rpc_call(s, "initialize", params, probe_timeout_ms() * 2);
    }
    if (!r.ok) { s.error = "initialize failed: " + r.describe(); return false; }

    const json &res = r.result;
    if (res.contains("protocolVersion") && res["protocolVersion"].is_string())
        s.version = res["protocolVersion"].get<std::string>();
    else
        s.version = MCP_LEGACY_VERSION;
    if (res.contains("capabilities")) s.capabilities = res["capabilities"];
    if (res.contains("serverInfo") && res["serverInfo"].is_object()) {
        const json &si = res["serverInfo"];
        if (si.contains("name") && si["name"].is_string()) s.server_name = si["name"].get<std::string>();
        if (si.contains("version") && si["version"].is_string()) s.server_version = si["version"].get<std::string>();
    }
    if (res.contains("instructions") && res["instructions"].is_string())
        s.instructions = res["instructions"].get<std::string>();

    /* A legacy server may refuse everything until this lands. */
    rpc_notify(s, "notifications/initialized", json::object());
    return true;
}

/* Probe with server/discover and decide the era from what comes back. The
 * fallback MUST NOT be keyed to a single error code — legacy SDKs answer an
 * unknown pre-initialize method with -32601, -32602, or nothing at all. */
bool negotiate(Server &s) {
    s.era = Era::Modern;
    s.version = MCP_MODERN_VERSION;
    RpcResult r = rpc_call(s, "server/discover", json::object(), probe_timeout_ms());

    if (r.ok) {
        const json &res = r.result;
        std::string v = res.contains("supportedVersions") ? pick_version(res["supportedVersions"]) : "";
        if (!v.empty()) s.version = v;
        if (res.contains("capabilities")) s.capabilities = res["capabilities"];
        if (res.contains("instructions") && res["instructions"].is_string())
            s.instructions = res["instructions"].get<std::string>();
        if (res.contains("_meta") && res["_meta"].is_object()) {
            const json &m = res["_meta"];
            auto si = m.find("io.modelcontextprotocol/serverInfo");
            if (si != m.end() && si->is_object()) {
                if (si->contains("name") && (*si)["name"].is_string())
                    s.server_name = (*si)["name"].get<std::string>();
                if (si->contains("version") && (*si)["version"].is_string())
                    s.server_version = (*si)["version"].get<std::string>();
            }
        }
        return true;
    }

    if (r.is_error && r.code == -32022) {          /* UnsupportedProtocolVersion */
        std::string v;
        if (r.data.is_object() && r.data.contains("supported")) v = pick_version(r.data["supported"]);
        if (!v.empty()) { s.version = v; return true; }   /* still modern, other revision */
        s.error = "server supports no protocol version this client speaks";
        return false;
    }
    if (r.is_error && (r.code == -32020 || r.code == -32021)) {
        /* HeaderMismatch / MissingRequiredClientCapability: recognized modern
         * errors, so the server IS modern and the fault is ours to report. */
        s.error = "modern server rejected the probe: " + r.describe();
        return false;
    }

    return legacy_initialize(s);
}

/* ── tools ──────────────────────────────────────────────────────────── */

/* Walk an inputSchema for x-mcp-header annotations. Clients on Streamable HTTP
 * MUST mirror annotated values into Mcp-Param-{Name} headers and MUST reject a
 * tool whose annotation breaks the rules — reject means drop THAT tool, not the
 * whole list, so one malformed definition cannot cost the user every other tool.
 * Returns false (with `why`) when the tool must be dropped. */
bool collect_header_params(const json &schema, std::vector<std::string> &path,
                           std::vector<HeaderParam> &out, std::string &why) {
    if (!schema.is_object()) return true;

    auto props = schema.find("properties");
    if (props == schema.end() || !props->is_object()) return true;

    for (auto it = props->begin(); it != props->end(); ++it) {
        const json &p = it.value();
        if (!p.is_object()) continue;
        path.push_back(it.key());

        auto ann = p.find("x-mcp-header");
        if (ann != p.end()) {
            if (!ann->is_string()) { why = "x-mcp-header on '" + it.key() + "' is not a string"; return false; }
            std::string h = ann->get<std::string>();
            if (h.empty()) { why = "empty x-mcp-header on '" + it.key() + "'"; return false; }
            for (unsigned char c : h) {
                bool tchar = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                             (c >= '0' && c <= '9') || strchr("!#$%&'*+-.^_`|~", c) != nullptr;
                if (!tchar) { why = "x-mcp-header '" + h + "' is not a valid HTTP token"; return false; }
            }
            std::string ty = (p.contains("type") && p["type"].is_string()) ? p["type"].get<std::string>() : "";
            if (ty != "string" && ty != "integer" && ty != "boolean") {
                why = "x-mcp-header '" + h + "' is on a non-primitive (type '" + ty + "')";
                return false;
            }
            for (const auto &e : out)
                if (lower(e.header) == lower(h)) { why = "duplicate x-mcp-header '" + h + "'"; return false; }
            HeaderParam hp; hp.path = path; hp.header = h;
            out.push_back(hp);
        }

        /* Only a chain of `properties` keys is statically reachable; items/$ref/
         * oneOf/if are explicitly out, so we simply do not descend into them. */
        if (!collect_header_params(p, path, out, why)) return false;
        path.pop_back();
    }
    return true;
}

/* An annotation is only meaningful on HTTP; stdio clients MAY ignore it. */
bool tool_from_json(const Server &s, const json &t, McpTool &out, std::string &why) {
    if (!t.is_object() || !t.contains("name") || !t["name"].is_string()) {
        why = "tool has no name";
        return false;
    }
    out.remote = t["name"].get<std::string>();
    out.server = s.name;
    out.full   = "mcp__" + s.name + "__" + sanitize_name(out.remote);

    std::string desc;
    if (t.contains("description") && t["description"].is_string())
        desc = t["description"].get<std::string>();
    else if (t.contains("title") && t["title"].is_string())
        desc = t["title"].get<std::string>();
    if (desc.empty()) desc = out.remote;
    /* The model sees a flat list of tools from every server at once, so say which
     * server this one reaches — otherwise two similar tools are indistinguishable. */
    out.description = "[MCP: " + s.name + "] " + desc;

    json schema = json::object();
    if (t.contains("inputSchema") && t["inputSchema"].is_object()) schema = t["inputSchema"];
    else schema = json{{"type", "object"}};
    try { out.schema = schema.dump(); } catch (...) { out.schema = "{\"type\":\"object\"}"; }

    if (t.contains("annotations") && t["annotations"].is_object()) {
        const json &a = t["annotations"];
        if (a.contains("readOnlyHint") && a["readOnlyHint"].is_boolean())
            out.readonly = a["readOnlyHint"].get<bool>() ? 1 : 0;
    }

    if (s.transport == Transport::Http) {
        std::vector<std::string> path;
        if (!collect_header_params(schema, path, out.hdr_params, why)) return false;
    }
    return true;
}

/* tools/list, following `nextCursor` until the server stops paginating. */
bool list_tools(Server &s, std::vector<McpTool> &out, std::string &warn) {
    std::string cursor;
    int pages = 0;
    for (;;) {
        json params = json::object();
        if (!cursor.empty()) params["cursor"] = cursor;
        RpcResult r = rpc_call(s, "tools/list", params, 0);
        if (!r.ok) {
            /* A server with no tools capability is not broken — it may serve only
             * resources or prompts, which BASI does not consume yet. */
            if (r.is_error && r.code == -32601) { s.error = "exposes no tools"; return true; }
            s.error = "tools/list failed: " + r.describe();
            return false;
        }
        const json &res = r.result;
        if (res.contains("tools") && res["tools"].is_array()) {
            for (const json &t : res["tools"]) {
                McpTool mt;
                std::string why;
                if (!tool_from_json(s, t, mt, why)) {
                    warn += "  " + s.name + ": dropped a tool (" + why + ")\n";
                    continue;
                }
                bool dup = false;
                for (const auto &e : out) if (e.full == mt.full) { dup = true; break; }
                if (dup) { warn += "  " + s.name + ": duplicate tool name '" + mt.remote + "'\n"; continue; }
                out.push_back(mt);
            }
        }
        cursor.clear();
        if (res.contains("nextCursor") && res["nextCursor"].is_string())
            cursor = res["nextCursor"].get<std::string>();
        if (cursor.empty()) break;
        if (++pages > 50) { warn += "  " + s.name + ": stopped after 50 tool pages\n"; break; }
    }
    return true;
}

/* ── result → text ──────────────────────────────────────────────────── */

void append_content_block(std::string &out, const json &c) {
    std::string type = (c.contains("type") && c["type"].is_string()) ? c["type"].get<std::string>() : "";
    if (type == "text") {
        if (c.contains("text") && c["text"].is_string()) out += c["text"].get<std::string>();
        else out += c.dump();
    } else if (type == "image" || type == "audio") {
        std::string mime = (c.contains("mimeType") && c["mimeType"].is_string())
                           ? c["mimeType"].get<std::string>() : type;
        size_t n = (c.contains("data") && c["data"].is_string()) ? c["data"].get<std::string>().size() : 0;
        /* BASI's chat path is text-only, so say what arrived rather than dumping
         * a megabyte of base64 into the context window. */
        out += "[" + type + " content: " + mime + ", " + std::to_string(n) + " base64 chars, not shown]";
    } else if (type == "resource_link") {
        out += "[resource_link] ";
        if (c.contains("uri") && c["uri"].is_string()) out += c["uri"].get<std::string>();
        if (c.contains("description") && c["description"].is_string())
            out += " — " + c["description"].get<std::string>();
    } else if (type == "resource") {
        const json &r = c.contains("resource") ? c["resource"] : json::object();
        if (r.contains("text") && r["text"].is_string()) {
            if (r.contains("uri") && r["uri"].is_string())
                out += "[" + r["uri"].get<std::string>() + "]\n";
            out += r["text"].get<std::string>();
        } else {
            out += "[embedded resource: " +
                   (r.contains("uri") && r["uri"].is_string() ? r["uri"].get<std::string>() : "?") +
                   ", binary, not shown]";
        }
    } else {
        out += c.dump();
    }
}

std::string result_to_text(const json &res) {
    std::string out;
    bool have_text = false;
    if (res.contains("content") && res["content"].is_array()) {
        for (const json &c : res["content"]) {
            if (!out.empty()) out += "\n";
            append_content_block(out, c);
            have_text = true;
        }
    }
    /* structuredContent is the authoritative form; servers SHOULD also mirror it
     * into a text block, and when one did we would otherwise print it twice. */
    if (!have_text && res.contains("structuredContent")) {
        try { out = res["structuredContent"].dump(2); } catch (...) {}
    }
    if (out.empty()) out = "(the tool returned no content)";
    return out;
}

/* ── registry plumbing ──────────────────────────────────────────────── */

/* Rebuild the BasiToolDef view over `tools` and hand it to the shared registry.
 * Pointers are into the McpTool strings, so this must run after `tools` is final
 * — a later push_back would reallocate the vector out from under them. */
void publish_tools(void) {
    g_reg.defs.clear();
    g_reg.defs.reserve(g_reg.tools.size());
    for (const auto &t : g_reg.tools) {
        BasiToolDef d;
        d.name        = t.full.c_str();
        d.description = t.description.c_str();
        d.parameters  = t.schema.c_str();
        g_reg.defs.push_back(d);
    }
    basi_tooldefs_set_extra(g_reg.defs.empty() ? nullptr : g_reg.defs.data(),
                            (int)g_reg.defs.size());
}

/* Did the child die, and if so why? A wrong `command` is the most common thing
 * to get wrong in an MCP config, and "no response within 5000ms" does not tell
 * anyone that the binary simply is not on PATH — the exit status does. */
bool stdio_died(Server &s, std::string &why) {
    if (s.pid <= 0) return false;
    int st;
    if (waitpid(s.pid, &st, WNOHANG) != s.pid) return false;
    s.pid = -1;
    int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    if (code == 127)
        why = "'" + s.command + "' not found — is it on your PATH?";
    else if (code == 126)
        why = "'" + s.command + "' could not be executed" +
              (s.cwd.empty() ? "" : " (cwd '" + s.cwd + "')");
    else if (WIFSIGNALED(st))
        why = "the server process was killed by signal " + std::to_string(WTERMSIG(st));
    else
        why = "the server process exited immediately (status " + std::to_string(code) + ")";
    if (!getenv("BASI_MCP_DEBUG"))
        why += "; set BASI_MCP_DEBUG=1 to keep its stderr";
    return true;
}

bool connect_server(Server &s, std::vector<McpTool> &tools, std::string &warn) {
    s.connected = false;
    s.error.clear();
    s.era = Era::Unknown;
    s.version.clear();
    s.session_id.clear();
    s.n_tools = 0;

    if (s.transport == Transport::Stdio && !stdio_spawn(s)) return false;

    if (!negotiate(s)) {
        /* A dead process explains itself better than the handshake timeout it
         * caused, so let that reason win. */
        std::string why;
        if (s.transport == Transport::Stdio && stdio_died(s, why)) s.error = why;
        if (s.transport == Transport::Stdio) stdio_kill(s);
        return false;
    }

    size_t before = tools.size();
    if (!list_tools(s, tools, warn)) {
        if (s.transport == Transport::Stdio) stdio_kill(s);
        return false;
    }
    s.n_tools = (int)(tools.size() - before);
    s.connected = true;
    return true;
}

const char *era_name(Era e) {
    return e == Era::Modern ? "modern" : e == Era::Legacy ? "legacy" : "?";
}

}  // namespace

/* ── public API ─────────────────────────────────────────────────────── */

extern "C" const BasiToolDef *mcp_init(const char *extra_config, int verbose, int *n) {
    if (n) *n = 0;
    const char *off = getenv("BASI_MCP");
    if (off && strcmp(off, "0") == 0) { g_reg.disabled = true; return nullptr; }
    g_reg.disabled = false;

    std::string warn;
    g_reg.servers = load_config(extra_config, warn);
    g_reg.configured = !g_reg.servers.empty();
    if (!g_reg.configured) {
        if (!warn.empty() && verbose) fprintf(stderr, "\033[33m[MCP config]\033[0m\n%s", warn.c_str());
        return nullptr;
    }

    /* Writing to a server that died between the check and the write raises
     * SIGPIPE, whose default action would kill BASI outright. Ignore it once so
     * the write returns EPIPE and the failure is reported as a tool error. */
    if (!g_reg.sigpipe_ignored) { signal(SIGPIPE, SIG_IGN); g_reg.sigpipe_ignored = true; }

    g_reg.tools.clear();
    for (auto &s : g_reg.servers) {
        if (!s.enabled) {
            if (verbose) printf("\033[90m[MCP] %s: disabled\033[0m\n", s.name.c_str());
            continue;
        }
        long t0 = now_ms();
        bool ok = connect_server(s, g_reg.tools, warn);
        long ms = now_ms() - t0;
        if (verbose) {
            if (ok)
                printf("\033[90m[MCP] %s: %d tool%s (%s %s, %ldms)\033[0m\n",
                       s.name.c_str(), s.n_tools, s.n_tools == 1 ? "" : "s",
                       era_name(s.era), s.version.c_str(), ms);
            else
                printf("\033[33m[MCP] %s: not available — %s\033[0m\n",
                       s.name.c_str(), s.error.c_str());
            fflush(stdout);
        }
    }
    if (verbose && !warn.empty()) fprintf(stderr, "\033[33m[MCP warnings]\033[0m\n%s", warn.c_str());

    publish_tools();
    if (n) *n = (int)g_reg.defs.size();
    return g_reg.defs.empty() ? nullptr : g_reg.defs.data();
}

extern "C" int mcp_configured(void) { return g_reg.configured ? 1 : 0; }

extern "C" int mcp_is_tool(const char *name) { return find_tool(name) != nullptr; }

extern "C" int mcp_tool_readonly(const char *name) {
    McpTool *t = find_tool(name);
    return t ? t->readonly : -1;
}

extern "C" const char *mcp_tool_server(const char *name) {
    McpTool *t = find_tool(name);
    return t ? t->server.c_str() : nullptr;
}

extern "C" char *mcp_call_tool(const char *name, const char *arguments_json) {
    McpTool *t = find_tool(name);
    if (!t) return strdup("Error: unknown MCP tool.");
    Server *s = find_server(t->server);
    if (!s) return strdup("Error: the MCP server for this tool is gone.");
    if (!s->connected) {
        std::string m = "Error: MCP server '" + s->name + "' is not connected";
        if (!s->error.empty()) m += " (" + s->error + ")";
        m += ". Do not retry this tool.";
        return strdup(m.c_str());
    }

    json args = json::object();
    if (arguments_json && *arguments_json) {
        try { args = json::parse(arguments_json); } catch (...) { args = json::object(); }
        if (!args.is_object()) args = json::object();
    }

    json params = { {"name", t->remote}, {"arguments", args} };

    /* Mirror x-mcp-header parameters into headers, reading each annotated value at
     * its exact property path; a value that is absent simply omits its header. */
    std::vector<std::pair<std::string, std::string>> hdrs;
    if (s->transport == Transport::Http) {
        for (const auto &hp : t->hdr_params) {
            const json *cur = &args;
            bool found = true;
            for (const auto &k : hp.path) {
                if (!cur->is_object() || !cur->contains(k)) { found = false; break; }
                cur = &(*cur)[k];
            }
            if (!found || cur->is_null()) continue;
            std::string v;
            if (cur->is_string())            v = cur->get<std::string>();
            else if (cur->is_boolean())      v = cur->get<bool>() ? "true" : "false";
            else if (cur->is_number_integer()) v = std::to_string(cur->get<long long>());
            else continue;
            hdrs.push_back({"Mcp-Param-" + hp.header, encode_header_value(v)});
        }
    }

    RpcResult r = rpc_call(*s, "tools/call", params, 0, hdrs);

    if (!r.ok) {
        /* A protocol error is still information the model can act on — an unknown
         * tool or a schema violation is fixable by calling differently. */
        std::string m = "Error from MCP server '" + s->name + "': " + r.describe();
        if (!r.transport_error.empty() && s->transport == Transport::Stdio && s->pid > 0) {
            int st;
            if (waitpid(s->pid, &st, WNOHANG) == s->pid) {
                s->pid = -1; s->connected = false;
                s->error = "process exited";
                m += " (the server process exited; run /mcp reconnect " + s->name + ")";
            }
        }
        return strdup(m.c_str());
    }

    const json &res = r.result;

    /* MRTR: the server wants elicitation/sampling/roots before it can finish. BASI
     * advertises none of those capabilities, so say so plainly instead of looping. */
    if (res.contains("resultType") && res["resultType"].is_string() &&
        res["resultType"].get<std::string>() == "input_required") {
        std::string m = "Error: this tool needs interactive input (elicitation or sampling) "
                        "that BASI does not provide. Try a different tool or supply the "
                        "missing values as arguments.";
        return strdup(m.c_str());
    }

    std::string text = result_to_text(res);
    bool is_err = res.contains("isError") && res["isError"].is_boolean() && res["isError"].get<bool>();
    if (is_err) text = "Tool error: " + text;   /* actionable: the model can self-correct */
    return strdup(text.c_str());
}

extern "C" char *mcp_status_report(int detail, const char *server_filter) {
    std::string out;
    if (g_reg.disabled) return strdup("MCP is disabled for this run (--no-mcp / BASI_MCP=0).\n");
    if (!g_reg.configured) {
        out = "No MCP servers configured.\n\n"
              "Declare them in ~/.config/basi-cli/mcp.json (or ./.basi/mcp.json for this\n"
              "project only):\n\n"
              "  {\n"
              "    \"mcpServers\": {\n"
              "      \"filesystem\": {\n"
              "        \"command\": \"npx\",\n"
              "        \"args\": [\"-y\", \"@modelcontextprotocol/server-filesystem\", \".\"]\n"
              "      },\n"
              "      \"example-http\": {\n"
              "        \"url\": \"https://example.com/mcp\",\n"
              "        \"headers\": { \"Authorization\": \"Bearer ${EXAMPLE_TOKEN}\" }\n"
              "      }\n"
              "    }\n"
              "  }\n";
        return strdup(out.c_str());
    }

    for (const auto &s : g_reg.servers) {
        if (server_filter && *server_filter && s.name != server_filter) continue;
        out += s.name;
        if (!s.enabled)            out += "  [disabled]";
        else if (s.connected)      out += "  [connected]";
        else                       out += "  [FAILED]";
        out += "\n";
        out += "  transport: ";
        out += (s.transport == Transport::Stdio ? "stdio  " + s.command : "http   " + s.url);
        out += "\n";
        if (s.connected) {
            out += "  protocol:  " + std::string(era_name(s.era)) + " " + s.version + "\n";
            if (!s.server_name.empty())
                out += "  server:    " + s.server_name +
                       (s.server_version.empty() ? "" : " " + s.server_version) + "\n";
            out += "  tools:     " + std::to_string(s.n_tools) + "\n";
            if (!s.instructions.empty()) {
                std::string i = s.instructions;
                if (i.size() > 200) i = i.substr(0, 200) + "...";
                out += "  notes:     " + i + "\n";
            }
        } else if (!s.error.empty()) {
            out += "  error:     " + s.error + "\n";
        }
        if (detail) {
            for (const auto &t : g_reg.tools) {
                if (t.server != s.name) continue;
                std::string d = t.description;
                size_t br = d.find("] ");
                if (br != std::string::npos) d = d.substr(br + 2);
                if (d.size() > 90) d = d.substr(0, 90) + "...";
                out += "    " + t.full + "\n        " + d + "\n";
            }
        }
        out += "\n";
    }
    if (out.empty()) out = "No MCP server matches that name.\n";
    return strdup(out.c_str());
}

extern "C" char *mcp_reconnect(const char *server_filter) {
    if (!g_reg.configured) return strdup("No MCP servers configured.\n");

    std::string report, warn;
    /* Rebuild the whole tool list: a reconnected server's tools must land in the
     * same flat vector, and the surviving servers' entries move with it. */
    std::vector<McpTool> keep;
    for (auto &s : g_reg.servers) {
        bool target = !server_filter || !*server_filter || s.name == server_filter;
        if (!target) {
            for (const auto &t : g_reg.tools) if (t.server == s.name) keep.push_back(t);
            continue;
        }
        if (s.transport == Transport::Stdio) stdio_kill(s);
        s.connected = false;
        if (!s.enabled) { report += s.name + ": disabled\n"; continue; }
        bool ok = connect_server(s, keep, warn);
        report += s.name + ": " + (ok ? std::to_string(s.n_tools) + " tools ("
                                        + era_name(s.era) + " " + s.version + ")"
                                      : "FAILED — " + s.error) + "\n";
    }
    g_reg.tools = keep;
    publish_tools();
    /* basi_tool_defs() now reports the new set, but what the MODEL is shown is
     * whatever was last handed to basi_set_tools — and only the caller knows
     * whether this session is scoped by --tools. Re-advertising is therefore the
     * caller's job; doing it here would silently widen a hard-scoped phase. */

    if (!warn.empty()) report += warn;
    return strdup(report.c_str());
}

extern "C" void mcp_shutdown(void) {
    for (auto &s : g_reg.servers)
        if (s.transport == Transport::Stdio) stdio_kill(s);
    g_reg.tools.clear();
    g_reg.defs.clear();
    g_reg.servers.clear();
    g_reg.configured = false;
    basi_tooldefs_set_extra(nullptr, 0);
}
