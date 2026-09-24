/* `basi-cli sleep` — turn this project's past chats into reviewed lessons.
 *
 * Measured before it was built (eval/ab, 2026-09-24): on the /version task a one-
 * line habit note took deepseek-v4p1-flash from 7/12 to 12/12, and a pasted old
 * chat cut rounds 38 → 17 on the runs that succeeded, because its tool results
 * were a map of where things live in the code. Pasting whole chats costs ~24 KB
 * per call and carries stale tool names; this keeps the two useful kinds of
 * content — HABITS and a CODE MAP — as a few lines each.
 *
 * Deterministic first, the model only where it is irreducible:
 *   1. episodes   — split each session into (request, work, your reaction) and keep
 *                   the ones with signal: hit the round cap, tool errors, repeats,
 *                   a correction or a pasted review in your next message, or real
 *                   work (≥5 tool calls). Telemetry adds capped/failed turns.
 *   2. extract    — the model proposes lessons from each kept episode, as JSON.
 *   3. verify     — code-map lessons must name files and symbols that exist TODAY
 *                   (old chats describe code that moved or was deleted); habit
 *                   lessons must quote evidence that is really in the chat; near-
 *                   duplicates of existing, rejected or earlier proposals are dropped.
 *   4. review     — nothing reaches the prompt until you accept it. Accepted lessons
 *                   go to .basi/lessons.md, which BASI loads beside ./BASI.md.
 *
 * State lives in .basi/dream/: seen.json (sessions already processed, by size),
 * pending.jsonl (proposals awaiting review), rejected.jsonl. */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nlohmann/json.hpp"
#include "dream.h"
#include "srvchat.h"
#include "tooldefs.h"

extern "C" {
char *session_dir_path(void);
int   mkdir_p(const char *path);
}

using json = nlohmann::json;
using std::string;
using std::vector;

namespace {

const char *DREAM_DIR    = ".basi/dream";
const char *LESSONS_PATH = ".basi/lessons.md";

/* ── small helpers ──────────────────────────────────────────────────────── */

string read_file(const string &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

bool file_exists(const string &p) { struct stat st; return stat(p.c_str(), &st) == 0; }

long file_size(const string &p) { struct stat st; return stat(p.c_str(), &st) == 0 ? (long)st.st_size : -1; }

/* Session logs can hold invalid UTF-8 (a tool result cut mid-codepoint), and the
 * JSON parser rejects it. Replace bad bytes rather than lose the whole line. */
string sanitize_utf8(const string &s) {
    string out; out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        bool ok = len > 0 && i + len <= n;
        for (int k = 1; ok && k < len; k++) ok = ((unsigned char)s[i + k] >> 6) == 2;
        if (ok) { out.append(s, i, len); i += len; }
        else    { out += '?'; i++; }
    }
    return out;
}

string clip(const string &s, size_t n) {
    if (s.size() <= n) return s;
    size_t h = n * 2 / 3, t = n - h;
    return s.substr(0, h) + "\n[...]\n" + s.substr(s.size() - t);
}

string lower_ws(const string &s) {   /* lowercase, whitespace collapsed */
    string o; bool sp = false;
    for (unsigned char c : s) {
        if (std::isspace(c)) { sp = true; continue; }
        if (sp && !o.empty()) o += ' ';
        sp = false; o += (char)std::tolower(c);
    }
    return o;
}

bool starts_with(const string &s, const string &p) { return s.rfind(p, 0) == 0; }

std::set<string> words(const string &s) {
    std::set<string> w; string cur;
    for (unsigned char c : s + " ") {
        if (std::isalnum(c) || c == '_') cur += (char)std::tolower(c);
        else { if (cur.size() > 2) w.insert(cur); cur.clear(); }
    }
    return w;
}

/* Shared content words over the SHORTER lesson's words. Plain Jaccard failed on
 * real proposals: six rewordings of "slash commands are a strcmp chain in
 * src/main.c" scored 0.18-0.38, inside the range of distinct lessons. With
 * stopwords removed, duplicates scored >= 0.39 and distinct pairs <= 0.29 on 19
 * hand-labelled pairs (2026-09-24) — a small sample, so the cut is 0.35. */
const std::set<string> STOPWORDS = {
    "the","and","for","with","are","from","into","not","use","than","then","when",
    "before","after","same","each","this","that","its","don","instead","rather",
    "you","your","any","all","only","will","can","should","must","has","have"};

double overlap(const std::set<string> &a0, const std::set<string> &b0) {
    std::set<string> a, b;
    for (auto &x : a0) if (!STOPWORDS.count(x)) a.insert(x);
    for (auto &x : b0) if (!STOPWORDS.count(x)) b.insert(x);
    if (a.empty() || b.empty()) return 0;
    size_t in = 0;
    for (auto &x : a) in += b.count(x);
    return (double)in / (double)std::min(a.size(), b.size());
}

/* ── sessions → episodes ────────────────────────────────────────────────── */

struct Rec { string role, content; };

/* Every tool name the sessions ever called. A habit that names one of these
 * which BASI no longer has ("terminate every apply_patch hunk with @@ END") is
 * advice for a tool that is gone. */
std::set<string> g_hist_tools;

vector<Rec> load_session(const string &path) {
    vector<Rec> out;
    std::ifstream f(path, std::ios::binary);
    string line;
    while (std::getline(f, line)) {
        try {
            json j = json::parse(sanitize_utf8(line));
            out.push_back({j.value("role", ""), j.value("content", "")});
        } catch (...) {}
    }
    return out;
}

/* Messages BASI itself injects as "user" — nudges, not the person. */
bool is_nudge(const string &c) {
    return starts_with(c, "You have used your entire tool budget") ||
           starts_with(c, "Your last message looked like a tool call") ||
           starts_with(c, "That image could not be sent") ||
           starts_with(c, "<tool_result>");          /* pre-June log format */
}

string strip_date(const string &c) {
    static const std::regex re(R"(^\[Today's date: [^\]]*\]\s*)");
    return std::regex_replace(c, re, "");
}

struct Episode {
    string session;
    int    index = 0;
    string request, reaction, text;   /* text = condensed episode for the model */
    int    tool_calls = 0, tool_errors = 0, repeats = 0;
    bool   capped = false, correction = false, review = false, check_failed = false;
    string signals() const {
        string s;
        auto add = [&](const string &x) { if (!s.empty()) s += ", "; s += x; };
        if (check_failed) add("automatic check failed");
        if (capped)      add("hit the round cap");
        if (tool_errors) add(std::to_string(tool_errors) + " tool error(s)");
        if (repeats)     add(std::to_string(repeats) + " repeated call(s)");
        if (correction)  add("user corrected it");
        if (review)      add("user pasted a review");
        if (s.empty())   add(std::to_string(tool_calls) + " tool calls of work");
        return s;
    }
};

bool looks_like_correction(const string &reaction) {
    static const std::regex re(
        R"((^|\W)(no|not true|wrong|incorrect|doesn'?t work|didn'?t|still|isn'?t|broken|nothing is|why (did|does|is)|you (never|forgot|missed)|that'?s not|stop)(\W|$))",
        std::regex::icase);
    return std::regex_search(reaction.substr(0, 600), re);
}

vector<Episode> episodes_of(const string &path, const string &name) {
    vector<Rec> recs = load_session(path);
    vector<Episode> eps;
    vector<size_t> starts;
    for (size_t i = 0; i < recs.size(); i++)
        if (recs[i].role == "user" && !is_nudge(recs[i].content)) starts.push_back(i);

    for (size_t k = 0; k < starts.size(); k++) {
        size_t b = starts[k], e = k + 1 < starts.size() ? starts[k + 1] : recs.size();
        Episode ep;
        ep.session = name;
        ep.index = (int)k;
        ep.request = strip_date(recs[b].content);
        if (k + 1 < starts.size()) ep.reaction = strip_date(recs[e].content);

        string work;
        for (size_t i = b + 1; i < e; i++) {
            const Rec &r = recs[i];
            if (r.role == "tool_call") {
                ep.tool_calls++;
                string nm = r.content, args;
                try { json j = json::parse(r.content); nm = j.value("name", "?"); g_hist_tools.insert(nm);
                      args = j.contains("arguments") ? j["arguments"].dump() : ""; } catch (...) {}
                work += "CALL " + nm + " " + clip(args, 200) + "\n";
            } else if (r.role == "tool_result" || (r.role == "user" && is_nudge(r.content))) {
                string c = r.content;
                try { json j = json::parse(c); c = j.value("content", c); } catch (...) {}
                if (c.find("TOOL ERROR") != string::npos) ep.tool_errors++;
                if (c.find("[REPEAT:") != string::npos)   ep.repeats++;
                if (starts_with(r.content, "You have used your entire tool budget")) ep.capped = true;
                work += "RESULT " + clip(c, 300) + "\n";
            } else if (r.role == "assistant") {
                /* pre-June logs carried the call inside the text: <tool>name ...</tool> */
                for (size_t p = 0; (p = r.content.find("<tool>", p)) != string::npos; p += 6) {
                    ep.tool_calls++;
                    size_t q = p + 6;
                    while (q < r.content.size() && (std::isalnum((unsigned char)r.content[q]) || r.content[q] == '_')) q++;
                    if (q > p + 6) g_hist_tools.insert(r.content.substr(p + 6, q - p - 6));
                }
                work += "ASSISTANT " + clip(r.content, i + 1 == e ? 1500 : 400) + "\n";
            }
        }
        /* an automatic check result is not the user: its command text ("--no-mcp")
           would otherwise read as a correction */
        bool automatic = starts_with(ep.reaction, "[AUTOMATIC CHECK");
        ep.correction = !ep.reaction.empty() && !automatic && looks_like_correction(ep.reaction);
        ep.check_failed = starts_with(ep.reaction, "[AUTOMATIC CHECK FAILED]");
        ep.review     = ep.reaction.size() > 800 && !starts_with(ep.reaction, "[AUTOMATIC CHECK");
        ep.text = "REQUEST:\n" + clip(ep.request, 1500) + "\n\nWHAT BASI DID:\n" + clip(work, 9000) +
                  (ep.reaction.empty() ? "" : "\n\nTHE USER'S NEXT MESSAGE:\n" + clip(ep.reaction, 2500));
        eps.push_back(ep);
    }
    return eps;
}

bool has_signal(const Episode &e, bool session_flagged) {
    if (e.check_failed) return true;       /* even with no tool calls: talked, did nothing, failed */
    if (e.tool_calls == 0) return false;               /* chit-chat: nothing learned */
    return e.capped || e.tool_errors || e.repeats || e.correction || e.review ||
           e.tool_calls >= 5 || session_flagged;
}

/* Sessions whose turns the telemetry saw go wrong (capped, repeat-stopped, tool
 * failures). Telemetry cannot say WHICH turn, so it flags the whole session. */
std::map<string, string> telemetry_flags() {
    std::map<string, string> flags;
    const char *xdg = getenv("XDG_DATA_HOME"), *home = getenv("HOME");
    string p = (xdg && *xdg) ? string(xdg) + "/basi-cli/telemetry.jsonl"
                             : string(home ? home : ".") + "/.local/share/basi-cli/telemetry.jsonl";
    std::ifstream f(p);
    string line;
    while (std::getline(f, line)) {
        try {
            json r = json::parse(sanitize_utf8(line));
            string s = r.value("session", "");
            if (s.empty()) continue;
            string o = r.value("outcome", "");
            int fails = 0;
            if (r.contains("tools"))
                for (auto &t : r["tools"]) fails += t.value("fails", 0);
            if (o != "answered" || fails)
                flags[s] += (flags[s].empty() ? "" : ", ") + o + (fails ? " +" + std::to_string(fails) + " tool fail" : "");
        } catch (...) {}
    }
    return flags;
}

/* ── the model ──────────────────────────────────────────────────────────── */

const char *EXTRACT_PROMPT =
    "You are reviewing ONE episode from a coding agent's past session in this project: "
    "the user's request, what the agent did (tool calls and results, abbreviated), and "
    "the user's next message (their reaction). The reaction may instead be the result "
    "of an AUTOMATIC CHECK run after the agent finished: if it failed, work out from "
    "what the agent did why the task was not done, and what habit would have avoided it.\n\n"
    "Extract lessons that would make the agent do BETTER next time. Two kinds only:\n"
    "- \"habit\": how to work. General, imperative, reusable on OTHER tasks. Only from "
    "something that went wrong or that the user had to correct: a false claim, a missed "
    "check, stopping early, a wasted loop. Not a summary of the task.\n"
    "- \"map\": where something lives in THIS codebase, useful for future tasks — "
    "file path plus function/symbol names (e.g. \"slash commands are dispatched in "
    "handle_slash_command in src/main.c; the /help text is in the same function\"). "
    "No line numbers: they go stale.\n\n"
    "Rules: do not invent. Every habit needs an \"evidence\" field quoting a short "
    "phrase (10-120 chars) copied EXACTLY from the episode. Every map lesson must list "
    "\"paths\" (repo-relative files) and \"symbols\" (identifiers as written in code). "
    "Prefer zero lessons to weak ones; most episodes deserve 0-2.\n\n"
    "Answer with ONLY a JSON array, no prose:\n"
    "[{\"kind\":\"habit\",\"lesson\":\"...\",\"evidence\":\"...\"},"
    "{\"kind\":\"map\",\"lesson\":\"...\",\"paths\":[\"src/x.c\"],\"symbols\":[\"fn\"]}]";

json ask_model(int port, const Episode &e, const string &flag, string &err) {
    json msgs = json::array();
    msgs.push_back({{"role", "system"}, {"content", EXTRACT_PROMPT}});
    string body = "Session " + e.session + ", episode " + std::to_string(e.index) +
                  " (signal: " + e.signals() + (flag.empty() ? "" : "; telemetry: " + flag) + ")\n\n" + e.text;
    msgs.push_back({{"role", "user"}, {"content", body}});
    string mj = msgs.dump(-1, ' ', false, json::error_handler_t::replace);
    SrvChatResult *r = srvchat_complete(port, mj.c_str(), nullptr, nullptr, 3000,
                                        nullptr, nullptr, nullptr);
    if (!r) { err = "no response from the model"; return json::array(); }
    string c = r->content ? r->content : "";
    srvchat_free(r);
    size_t a = c.find('['), b = c.rfind(']');
    if (a == string::npos || b == string::npos || b < a) { err = "no JSON array in the answer"; return json::array(); }
    try { return json::parse(c.substr(a, b - a + 1)); }
    catch (...) { err = "unparseable JSON"; return json::array(); }
}

/* ── verification (deterministic) ───────────────────────────────────────── */

string repo_relative(string p, const string &cwd) {
    if (starts_with(p, "./")) p = p.substr(2);
    if (!p.empty() && p[0] == '/') {
        if (starts_with(p, cwd + "/")) p = p.substr(cwd.size() + 1);
        else return "";                        /* outside this project */
    }
    return p;
}

bool symbol_in(const string &sym, const vector<string> &paths) {
    for (auto &p : paths) if (read_file(p).find(sym) != string::npos) return true;
    return false;
}

std::set<string> current_tools() {
    std::set<string> cur;
    int n = 0;
    const BasiToolDef *td = basi_tool_defs(&n);
    for (int i = 0; i < n; i++) if (td[i].name) cur.insert(td[i].name);
    return cur;
}

/* Name of a tool the lesson mentions that BASI no longer has, or "". */
string gone_tool(const string &text) {
    static const std::set<string> cur = current_tools();
    for (auto &t : g_hist_tools) {
        if (t.size() < 4 || cur.count(t) || starts_with(t, "mcp__")) continue;
        std::regex re("\\b" + t + "\\b");
        if (std::regex_search(text, re)) return t;
    }
    return "";
}

/* Session filenames are YYYYMMDD-HHMMSS.jsonl. */
time_t session_time(const string &src) {
    struct tm tm = {};
    if (!strptime(src.c_str(), "%Y%m%d-%H%M%S", &tm)) return 0;
    tm.tm_isdst = -1;
    return mktime(&tm);
}

/* Files a map lesson names that were modified after the chat it came from: the
 * lesson may describe code that has since changed, even if every symbol exists. */
json changed_since(const json &L, time_t when) {
    json out = json::array();
    if (!when || !L.contains("paths")) return out;
    for (auto &p : L["paths"]) {
        struct stat st;
        if (p.is_string() && stat(p.get<string>().c_str(), &st) == 0 && st.st_mtime > when)
            out.push_back(p);
    }
    return out;
}

/* Returns "" when the lesson holds, else why it was dropped. */
string verify(json &L, const Episode *e, const string &cwd) {
    string kind = L.value("kind", ""), text = L.value("lesson", "");
    if (text.size() < 15) return "empty lesson";
    static const std::regex lineno(R"((\bline[s]? \d+(\s*[-–]\s*\d+)?|:\d{2,}\b))", std::regex::icase);
    L["lesson"] = std::regex_replace(text, lineno, "");
    string gone = gone_tool(L.value("lesson", ""));
    if (!gone.empty()) return "stale: names the tool '" + gone + "', which BASI no longer has";
    if (kind == "habit") {
        string ev = L.value("evidence", "");
        if (ev.size() < 10) return "habit without evidence";
        if (e && lower_ws(e->text + "\n" + e->reaction).find(lower_ws(ev)) == string::npos)
            return "evidence not found in the chat";
        return "";
    }
    if (kind == "map") {
        vector<string> paths;
        if (L.contains("paths") && L["paths"].is_array())
            for (auto &p : L["paths"]) if (p.is_string()) {
                string r = repo_relative(p.get<string>(), cwd);
                if (r.empty() || !file_exists(r)) return "stale: " + p.get<string>() + " does not exist now";
                paths.push_back(r);
            }
        if (paths.empty()) return "map lesson without a file";
        L["paths"] = paths;
        if (L.contains("symbols") && L["symbols"].is_array())
            for (auto &s : L["symbols"]) if (s.is_string() && !symbol_in(s.get<string>(), paths))
                return "stale: " + s.get<string>() + " not found in " + paths[0];
        return "";
    }
    return "unknown kind '" + kind + "'";
}

/* ── state ──────────────────────────────────────────────────────────────── */

vector<json> read_jsonl(const string &p) {
    vector<json> v;
    std::ifstream f(p);
    string line;
    while (std::getline(f, line)) try { v.push_back(json::parse(line)); } catch (...) {}
    return v;
}

void write_jsonl(const string &p, const vector<json> &v) {
    std::ofstream f(p, std::ios::trunc);
    for (auto &j : v) f << j.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
}

/* The accepted lessons, one per bullet, for de-duplication. */
vector<string> lesson_lines() {
    vector<string> out;
    std::istringstream in(read_file(LESSONS_PATH));
    string line;
    while (std::getline(in, line)) if (starts_with(line, "- ")) out.push_back(line.substr(2));
    return out;
}

bool is_duplicate(const string &text, const vector<string> &against) {
    auto w = words(text);
    for (auto &x : against) if (overlap(w, words(x)) >= 0.35) return true;
    return false;
}

void append_lesson(const json &L) {
    string body = read_file(LESSONS_PATH);
    string kind = L.value("kind", "habit");
    const char *head = kind == "map" ? "## Code map" : "## Habits";
    if (body.empty())
        body = "# Lessons\n\nWritten by `basi-cli sleep` from this project's past sessions, and\n"
               "kept only after you accepted them. BASI reads this file at startup;\n"
               "edit or delete lines freely.\n\n## Code map\n\n## Habits\n";
    string line = "- " + L.value("lesson", "") + "  <!-- " + L.value("source", "") + " -->";
    size_t h = body.find(head);
    if (h == string::npos) { body += string("\n") + head + "\n"; h = body.find(head); }
    size_t next = body.find("\n## ", h + 1);
    /* insert after the section's last non-blank line, keeping the blank line
       that separates it from the next heading */
    size_t at = next == string::npos ? body.size() : next + 1;
    while (at > h && body[at - 1] == '\n') at--;
    body.insert(at, "\n" + line);
    if (next == string::npos) body += "\n";
    mkdir_p(".basi");
    std::ofstream(LESSONS_PATH, std::ios::trunc) << body;
}

/* Proposal state (and anything quoted from chats) never belongs in git, in any
 * project — so the folder ignores itself. */
void ensure_dream_dir() {
    mkdir_p(DREAM_DIR);
    string gi = string(DREAM_DIR) + "/.gitignore";
    if (!file_exists(gi)) std::ofstream(gi) << "# basi-cli sleep state — local only\n*\n";
}

string now_stamp() {
    char b[32]; time_t t = time(nullptr);
    strftime(b, sizeof b, "%Y-%m-%d", localtime(&t));
    return b;
}

/* ── commands ───────────────────────────────────────────────────────────── */

int cmd_run(int argc, char **argv) {
    bool all = false, dry = false;
    vector<string> from_dirs;
    int port = 8181, limit = 0;
    if (const char *pe = getenv("BASI_SERVER_PORT")) if (*pe) port = atoi(pe);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) all = true;
        else if (!strcmp(argv[i], "--dry-run")) dry = true;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) from_dirs.push_back(argv[++i]);
    }
    char *sd = session_dir_path();
    if (!sd) { fprintf(stderr, "No session directory for this project.\n"); return 1; }
    string sdir = sd; free(sd);
    char cwdb[4096]; string cwd = getcwd(cwdb, sizeof cwdb) ? cwdb : ".";

    ensure_dream_dir();
    string seen_p = string(DREAM_DIR) + "/seen.json";
    json seen = json::object();
    try { string s = read_file(seen_p); if (!s.empty()) seen = json::parse(s); } catch (...) {}

    /* files are keyed by their path relative to sdir, or by full path for
       --from sources (logs from `basi -p` runs with BASI_SESSION_LOG set) */
    vector<string> files;
    if (from_dirs.empty()) {
        if (DIR *d = opendir(sdir.c_str())) {
            while (dirent *de = readdir(d)) {
                string n = de->d_name;
                if (n.size() > 6 && n.substr(n.size() - 6) == ".jsonl") files.push_back(n);
            }
            closedir(d);
        }
    } else {
        sdir = "";
        for (auto &fd : from_dirs) {
            string cmd = "find '" + fd + "' -name 'session-*.jsonl' -type f 2>/dev/null";
            if (FILE *pp = popen(cmd.c_str(), "r")) {
                char buf[4096];
                while (fgets(buf, sizeof buf, pp)) {
                    string f = buf;
                    while (!f.empty() && (f.back() == '\n' || f.back() == '\r')) f.pop_back();
                    if (!f.empty()) files.push_back(f);
                }
                pclose(pp);
            }
        }
    }
    std::sort(files.begin(), files.end());
    auto full = [&](const string &n) { return sdir.empty() ? n : sdir + "/" + n; };
    auto flags = telemetry_flags();

    vector<Episode> todo;
    int n_sessions = 0, n_new = 0;
    for (auto &n : files) {
        long sz = file_size(full(n));
        n_sessions++;
        if (!all && seen.contains(n) && seen[n].get<long>() == sz) continue;
        n_new++;
        bool flagged = flags.count(n) > 0;
        for (auto &e : episodes_of(full(n), n))
            if (has_signal(e, flagged)) todo.push_back(e);
    }
    if (limit > 0 && (int)todo.size() > limit) todo.resize(limit);

    printf("sleep: %d session(s) in %s, %d new or changed, %zu episode(s) with signal\n",
           n_sessions, sdir.empty() ? "the --from dirs" : sdir.c_str(), n_new, todo.size());
    if (dry) {
        for (auto &e : todo)
            printf("  %s #%d  [%s]  %s\n", e.session.c_str(), e.index, e.signals().c_str(),
                   lower_ws(e.request).substr(0, 90).c_str());
        return 0;
    }
    if (todo.empty()) { printf("Nothing to learn from yet.\n"); return 0; }

    string pend_p = string(DREAM_DIR) + "/pending.jsonl";
    vector<json> pending = read_jsonl(pend_p);
    vector<json> rejected = read_jsonl(string(DREAM_DIR) + "/rejected.jsonl");
    vector<string> known = lesson_lines();
    for (auto &j : pending)  known.push_back(j.value("lesson", ""));
    for (auto &j : rejected) known.push_back(j.value("lesson", ""));
    int next_id = 1;
    for (auto &j : pending) next_id = std::max(next_id, j.value("id", 0) + 1);

    std::map<string, int> dropped;
    int kept = 0;
    for (size_t i = 0; i < todo.size(); i++) {
        Episode &e = todo[i];
        printf("  [%zu/%zu] %s #%d (%s) … ", i + 1, todo.size(), e.session.c_str(), e.index,
               e.signals().c_str());
        fflush(stdout);
        string err;
        json ls = ask_model(port, e, flags.count(e.session) ? flags[e.session] : "", err);
        if (!err.empty()) { printf("%s\n", err.c_str()); dropped[err]++; continue; }
        int k = 0;
        for (auto &L : ls) {
            if (!L.is_object()) continue;
            string why = verify(L, &e, cwd);
            if (why.empty() && is_duplicate(L.value("lesson", ""), known)) why = "duplicate";
            if (!why.empty()) {
                dropped[why.substr(0, why.find(':'))]++;
                continue;
            }
            L["id"] = next_id++;
            L["source"] = e.session + "#" + std::to_string(e.index);
            L["signal"] = e.signals();
            L["proposed"] = now_stamp();
            if (L.value("kind", "") == "map") {
                json ch = changed_since(L, session_time(e.session));
                if (!ch.empty()) L["changed_since"] = ch;
            }
            known.push_back(L.value("lesson", ""));
            pending.push_back(L);
            k++;
        }
        kept += k;
        printf("%d lesson(s)\n", k);
        write_jsonl(pend_p, pending);          /* save as we go: a crash keeps the work */
    }
    for (auto &n : files) {
        long sz = file_size(full(n));
        if (sz >= 0) seen[n] = sz;
    }
    std::ofstream(seen_p, std::ios::trunc) << seen.dump(1);

    printf("\n%d new lesson(s) proposed; %zu waiting for review.\n", kept, pending.size());
    for (auto &d : dropped) printf("  dropped %d: %s\n", d.second, d.first.c_str());
    printf("Review them with: basi-cli sleep review\n");
    return 0;
}

/* Re-apply the deterministic checks to pending proposals — the code keeps moving
 * after a proposal is made. No model call. Dropped ones go to dropped.jsonl. */
int cmd_recheck() {
    char *sd = session_dir_path();
    if (sd) {                                   /* rebuild the historical tool set */
        if (DIR *d = opendir(sd)) {
            while (dirent *de = readdir(d)) {
                string n = de->d_name;
                if (n.size() > 6 && n.substr(n.size() - 6) == ".jsonl") episodes_of(string(sd) + "/" + n, n);
            }
            closedir(d);
        }
        free(sd);
    }
    char cwdb[4096]; string cwd = getcwd(cwdb, sizeof cwdb) ? cwdb : ".";
    string pend_p = string(DREAM_DIR) + "/pending.jsonl", drop_p = string(DREAM_DIR) + "/dropped.jsonl";
    vector<json> pending = read_jsonl(pend_p), dropped = read_jsonl(drop_p), keep;
    vector<string> known = lesson_lines();
    for (auto &j : read_jsonl(string(DREAM_DIR) + "/rejected.jsonl")) known.push_back(j.value("lesson", ""));
    std::map<string, int> why_n;
    for (auto &L : pending) {
        string why = verify(L, nullptr, cwd);
        if (why.empty() && is_duplicate(L.value("lesson", ""), known)) why = "duplicate";
        if (!why.empty()) {
            L["dropped"] = why;
            dropped.push_back(L);
            why_n[why.substr(0, why.find(':'))]++;
            continue;
        }
        if (L.value("kind", "") == "map") {
            json ch = changed_since(L, session_time(L.value("source", "")));
            if (!ch.empty()) L["changed_since"] = ch; else L.erase("changed_since");
        }
        known.push_back(L.value("lesson", ""));
        keep.push_back(L);
    }
    write_jsonl(pend_p, keep);
    write_jsonl(drop_p, dropped);
    printf("recheck: %zu kept, %zu dropped\n", keep.size(), pending.size() - keep.size());
    for (auto &d : why_n) printf("  %d: %s\n", d.second, d.first.c_str());
    return 0;
}

void print_proposal(const json &L) {
    printf("\n#%d  %s  (from %s — %s)\n  %s\n", L.value("id", 0), L.value("kind", "").c_str(),
           L.value("source", "").c_str(), L.value("signal", "").c_str(), L.value("lesson", "").c_str());
    if (L.contains("evidence")) printf("  evidence: \"%s\"\n", L.value("evidence", "").c_str());
    if (L.contains("paths"))    printf("  paths: %s\n", L["paths"].dump().c_str());
    if (L.contains("changed_since"))
        printf("  \033[33m⚠ changed since that chat: %s — check it still holds\033[0m\n",
               L["changed_since"].dump().c_str());
}

int cmd_decide(const vector<json> &pending_in, const std::set<int> &accept, const std::set<int> &reject) {
    vector<json> pending, rej = read_jsonl(string(DREAM_DIR) + "/rejected.jsonl");
    int na = 0, nr = 0;
    for (auto &L : pending_in) {
        int id = L.value("id", 0);
        if (accept.count(id))      { append_lesson(L); na++; }
        else if (reject.count(id)) { rej.push_back(L); nr++; }
        else pending.push_back(L);
    }
    write_jsonl(string(DREAM_DIR) + "/pending.jsonl", pending);
    write_jsonl(string(DREAM_DIR) + "/rejected.jsonl", rej);
    printf("%d accepted into %s, %d rejected, %zu still pending.\n", na, LESSONS_PATH, nr, pending.size());
    return 0;
}

int cmd_review(int argc, char **argv) {
    vector<json> pending = read_jsonl(string(DREAM_DIR) + "/pending.jsonl");
    if (pending.empty()) { printf("Nothing waiting for review.\n"); return 0; }
    string sub = argc >= 3 ? argv[2] : "review";
    std::set<int> acc, rej;
    if (sub == "list") { for (auto &L : pending) print_proposal(L); return 0; }
    if (sub == "accept" || sub == "reject") {
        for (int i = 3; i < argc; i++) (sub == "accept" ? acc : rej).insert(atoi(argv[i]));
        return cmd_decide(pending, acc, rej);
    }
    if (!isatty(0)) {
        fprintf(stderr, "Not a terminal. Use: basi-cli sleep list | accept <id…> | reject <id…>\n");
        return 2;
    }
    for (auto &L : pending) {
        print_proposal(L);
        printf("  [a]ccept  [r]eject  [s]kip  [q]uit > ");
        fflush(stdout);
        char buf[16];
        if (!fgets(buf, sizeof buf, stdin) || buf[0] == 'q') break;
        if (buf[0] == 'a') acc.insert(L.value("id", 0));
        else if (buf[0] == 'r') rej.insert(L.value("id", 0));
    }
    return cmd_decide(pending, acc, rej);
}

}  // namespace


/* ── `basi-cli import claude` ───────────────────────────────────────────────
 * Claude Code keeps per-project memory notes in ~/.claude/projects/<path>/memory:
 * short markdown files with frontmatter (name, description, type). They hold
 * what was learned working on THIS project — measured numbers, dead ends,
 * how the user wants to work. This copies the ones the user picks into BASI's
 * knowledge base (the pinned shelf docs_search already reads), and proposes the
 * "feedback" ones — rules about how to work — as lessons for `sleep review`.
 * Transcripts are not imported: hundreds of MB, mostly tool output. */

struct ClaudeNote {
    string file, id, name, description, type, body;   /* id = file stem: stable, filename-safe */
};

string unquote(string v) {
    while (!v.empty() && (v.back() == ' ' || v.back() == '\r')) v.pop_back();
    size_t a = v.find_first_not_of(' ');
    v = a == string::npos ? "" : v.substr(a);
    if (v.size() >= 2 && (v[0] == '"' || v[0] == '\'') && v.back() == v[0]) {
        bool dq = v[0] == '"';
        v = v.substr(1, v.size() - 2);
        if (dq)                                   /* YAML double-quoted: \" and \\ are escapes */
            for (size_t i = 0; i + 1 < v.size(); i++)
                if (v[i] == '\\' && (v[i + 1] == '"' || v[i + 1] == '\\')) v.erase(i, 1);
    }
    return v;
}

bool parse_claude_note(const string &path, ClaudeNote &n) {
    string s = read_file(path);
    if (!starts_with(s, "---")) return false;
    size_t end = s.find("\n---", 3);
    if (end == string::npos) return false;
    std::istringstream fm(s.substr(3, end - 3));
    string line;
    while (std::getline(fm, line)) {
        size_t c = line.find(':');
        if (c == string::npos) continue;
        string key = unquote(line.substr(0, c)), val = unquote(line.substr(c + 1));
        if (key == "name" && n.name.empty()) n.name = val;
        else if (key == "description") n.description = val;
        else if (key == "type") n.type = val;            /* top-level or under metadata: */
    }
    size_t b = s.find('\n', end + 4);
    n.body = b == string::npos ? "" : s.substr(b + 1);
    n.file = path;
    string base = path.substr(path.rfind('/') + 1);
    n.id = base.substr(0, base.size() - 3);
    if (n.name.empty()) n.name = n.id;
    return true;
}

/* Claude Code names a project's folder after its path with every character that
 * is not a letter, digit or '-' turned into '-'. */
string claude_project_dir(const string &cwd) {
    string enc;
    for (unsigned char c : cwd) enc += (std::isalnum(c) || c == '-') ? (char)c : '-';
    const char *home = getenv("HOME");
    return string(home ? home : ".") + "/.claude/projects/" + enc;
}

std::set<int> parse_selection(const string &in, int n) {
    std::set<int> out;
    string tok;
    std::istringstream ss(in);
    while (std::getline(ss, tok, ',')) {
        tok = unquote(tok);
        if (tok.empty()) continue;
        size_t d = tok.find('-');
        int a = atoi(tok.c_str()), b = d == string::npos ? a : atoi(tok.c_str() + d + 1);
        for (int i = std::max(1, a); i <= std::min(n, b); i++) out.insert(i);
    }
    return out;
}

extern "C" int dream_import_claude(int argc, char **argv) {
    char cwdb[4096]; string cwd = getcwd(cwdb, sizeof cwdb) ? cwdb : ".";
    string dir = claude_project_dir(cwd) + "/memory";
    bool list = false, all = false;
    vector<string> names;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) list = true;
        else if (!strcmp(argv[i], "--all")) all = true;
        else if (!strcmp(argv[i], "--from") && i + 1 < argc) dir = argv[++i];
        else if (argv[i][0] != '-') names.push_back(argv[i]);
    }

    vector<ClaudeNote> notes;
    if (DIR *d = opendir(dir.c_str())) {
        vector<string> files;
        while (dirent *de = readdir(d)) {
            string f = de->d_name;
            if (f.size() > 3 && f.substr(f.size() - 3) == ".md" && f != "MEMORY.md") files.push_back(f);
        }
        closedir(d);
        std::sort(files.begin(), files.end());
        for (auto &f : files) {
            ClaudeNote n;
            if (parse_claude_note(dir + "/" + f, n)) notes.push_back(n);
        }
    }
    if (notes.empty()) {
        fprintf(stderr, "No Claude Code memory for this project at %s\n"
                        "  (use --from <dir> to point at another memory folder)\n", dir.c_str());
        return 1;
    }

    const string dest_dir = ".basi/knowledge/pinned";
    auto dest_of = [&](const ClaudeNote &n) { return dest_dir + "/claude-" + n.id + ".md"; };
    auto state_of = [&](const ClaudeNote &n) -> string {
        string cur = read_file(dest_of(n));
        if (cur.empty()) return "new";
        return cur.find(n.body) != string::npos ? "imported" : "changed";
    };

    printf("Claude Code memory: %s (%zu notes)\n", dir.c_str(), notes.size());
    for (size_t i = 0; i < notes.size(); i++)
        printf("  %2zu. [%-9s] %-8s %s — %s\n", i + 1, notes[i].type.c_str(), state_of(notes[i]).c_str(),
               notes[i].id.c_str(), notes[i].description.substr(0, 90).c_str());
    if (list) return 0;

    std::set<int> pick;
    if (all) for (size_t i = 1; i <= notes.size(); i++) pick.insert((int)i);
    for (auto &nm : names)
        for (size_t i = 0; i < notes.size(); i++)
            if (notes[i].id == nm || notes[i].name == nm) pick.insert((int)i + 1);
    if (pick.empty()) {
        if (!isatty(0)) {
            fprintf(stderr, "Not a terminal: pass --all or note names to import.\n");
            return 2;
        }
        printf("\nImport which? numbers/ranges (e.g. 1,4,7-9), 'a' = all, Enter = cancel: ");
        fflush(stdout);
        char buf[1024];
        if (!fgets(buf, sizeof buf, stdin)) return 0;
        string in = unquote(buf);
        if (!in.empty() && in.back() == '\n') in.pop_back();
        if (in == "a") for (size_t i = 1; i <= notes.size(); i++) pick.insert((int)i);
        else pick = parse_selection(in, (int)notes.size());
        if (pick.empty()) { printf("Nothing imported.\n"); return 0; }
    }

    mkdir_p(dest_dir.c_str());
    /* Claude's notes can hold hosts, paths and private numbers, and .basi/ is
       often committed. Keep the imported copies out of git in ANY project. */
    string gi = dest_dir + "/.gitignore";
    if (read_file(gi).find("claude-*.md") == string::npos)
        std::ofstream(gi, std::ios::app) << "# imported Claude Code memory — private, never commit\nclaude-*.md\n";

    string pend_p = string(DREAM_DIR) + "/pending.jsonl";
    ensure_dream_dir();
    vector<json> pending = read_jsonl(pend_p);
    vector<string> known = lesson_lines();
    for (auto &j : pending) known.push_back(j.value("lesson", ""));
    for (auto &j : read_jsonl(string(DREAM_DIR) + "/rejected.jsonl")) known.push_back(j.value("lesson", ""));
    int next_id = 1;
    for (auto &j : pending) next_id = std::max(next_id, j.value("id", 0) + 1);

    int added = 0, updated = 0, same = 0, proposed = 0;
    for (int k : pick) {
        const ClaudeNote &n = notes[k - 1];
        string st = state_of(n);
        if (st == "imported") { same++; continue; }
        std::ofstream(dest_of(n), std::ios::trunc)
            << "---\nsource: " << n.file << "\nshelf: pinned\ntitle: " << n.name
            << "\ntype: " << n.type << "\nimported: " << now_stamp()
            << "\n---\n\n# " << n.name << "\n\n> " << n.description << "\n\n" << n.body;
        (st == "new" ? added : updated)++;
        if (n.type == "feedback" && !n.description.empty() && !is_duplicate(n.description, known)) {
            json L = {{"kind", "habit"}, {"lesson", n.description}, {"id", next_id++},
                      {"source", "claude:" + n.id}, {"signal", "Claude Code feedback memory"},
                      {"proposed", now_stamp()}};
            pending.push_back(L);
            known.push_back(n.description);
            proposed++;
        }
    }
    write_jsonl(pend_p, pending);
    printf("\n%d added, %d updated, %d unchanged → %s/claude-*.md (git-ignored)\n",
           added, updated, same, dest_dir.c_str());
    if (proposed)
        printf("%d feedback note(s) proposed as lessons — accept them with: basi-cli sleep review\n", proposed);
    printf("BASI finds the notes through docs_search; nothing is added to every prompt.\n");
    return 0;
}

extern "C" int dream_cmd(int argc, char **argv) {
    string sub = argc >= 3 ? argv[2] : "";
    if (sub == "review" || sub == "list" || sub == "accept" || sub == "reject")
        return cmd_review(argc, argv);
    if (sub == "recheck") return cmd_recheck();
    if (sub == "show") {
        string s = read_file(LESSONS_PATH);
        printf("%s", s.empty() ? "No lessons yet. Run: basi-cli sleep\n" : s.c_str());
        return 0;
    }
    if (sub == "-h" || sub == "--help" || sub == "help") {
        printf("Usage:\n"
               "  basi-cli sleep [--dry-run] [--all] [--limit N] [--from DIR]\n"
               "                  learn from this project's past sessions (new/changed ones\n"
               "                  only unless --all); --dry-run lists the episodes, no model;\n"
               "                  --from DIR reads session-*.jsonl logs of `basi -p` runs\n"
               "                  (BASI_SESSION_LOG) instead, e.g. an eval/ab run\n"
               "  basi-cli sleep review            accept/reject proposals one by one\n"
               "  basi-cli sleep list              print the proposals waiting for review\n"
               "  basi-cli sleep accept|reject <id…>\n"
               "  basi-cli sleep recheck           re-apply the checks to pending proposals (no model)\n"
               "  basi-cli sleep show              print .basi/lessons.md\n"
               "  BASI_LESSONS=0 basi ...          run without loading the lessons\n");
        return 0;
    }
    return cmd_run(argc, argv);
}
