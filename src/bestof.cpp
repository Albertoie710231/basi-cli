/* Deterministic best-of-N selection. See bestof.h for why consensus beats a
 * judge here. Pure C++ + nlohmann; no libllama/ggml symbols. */
#include "bestof.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

extern "C" {
#include "reuse.h"   /* reuse_similarity — the deterministic text scorer */
}

using json = nlohmann::json;

static char *bo_dup(const std::string &s) {
    char *p = (char *) malloc(s.size() + 1);
    if (p) { memcpy(p, s.data(), s.size()); p[s.size()] = 0; }
    return p;
}

/* Trim and collapse internal whitespace runs — so two answers differing only in
 * line wrapping are not treated as different candidates. */
static std::string normalize_text(const char *s) {
    std::string out;
    if (!s) return out;
    bool in_ws = true;                       /* leading whitespace is skipped */
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char) *p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!in_ws) { out += ' '; in_ws = true; }
        } else { out += (char) c; in_ws = false; }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

/* Canonical signature for a tool-call turn: name(args) per call, joined.
 * nlohmann stores objects in a sorted map, so dump() already canonicalizes key
 * order — {"b":1,"a":2} and {"a":2,"b":1} produce the same string. Returns
 * false if any call is structurally unusable (no name, or args that aren't a
 * JSON object), which marks the whole candidate invalid. */
static bool tool_signature(const SrvChatResult *r, std::string &sig) {
    sig.clear();
    for (int i = 0; i < r->n_tool_calls; i++) {
        const SrvToolCall &tc = r->tool_calls[i];
        if (!tc.name || !tc.name[0]) return false;
        json args;
        try { args = json::parse(tc.arguments ? tc.arguments : "{}"); }
        catch (...) { return false; }        /* malformed args — a degenerate draw */
        if (!args.is_object()) return false;
        sig += tc.name;
        sig += '(';
        sig += args.dump();
        sig += ");";
    }
    return true;
}

extern "C" BestOfPick bestof_select(SrvChatResult *const *cands, int n) {
    BestOfPick pick = { -1, 0, 0, nullptr };
    if (!cands || n <= 0) return pick;

    std::vector<int>         valid;      /* indices surviving the filter */
    std::vector<std::string> sig(n);     /* tool signature, when in tool mode */
    std::vector<std::string> text(n);    /* normalized content, text mode */
    int n_tool = 0, n_text = 0;

    for (int i = 0; i < n; i++) {
        const SrvChatResult *r = cands[i];
        if (!r) continue;
        if (r->n_tool_calls > 0) {
            if (!tool_signature(r, sig[i])) continue;   /* drop: malformed call */
            valid.push_back(i); n_tool++;
        } else {
            text[i] = normalize_text(r->content);
            if (text[i].empty()) continue;              /* drop: empty turn */
            valid.push_back(i); n_text++;
        }
    }
    pick.n_valid = (int) valid.size();
    if (valid.empty()) return pick;                     /* caller falls back */

    /* Mixed turns mean the samples disagree on whether the task is DONE. Go with
     * the larger group; on a tie prefer acting over answering, since a premature
     * final answer ends the turn while an extra tool call is recoverable. */
    const bool tool_mode = (n_tool >= n_text) && n_tool > 0;

    if (tool_mode) {
        /* Two levels, because argument shape differs by tool. Exact matching works
         * for path-like args (read/edit agree byte-for-byte across samples) but
         * FAILS for free-text args — four samples all wanting web_search with
         * slightly different query wording is unanimous agreement that exact
         * matching scores as 1/4. So: vote on the tool NAME (always discrete),
         * then settle the arguments within the winning group. */
        std::vector<int> tools_v;
        for (int i : valid) if (cands[i]->n_tool_calls > 0) tools_v.push_back(i);

        std::vector<std::string> name_sig(n);
        for (int i : tools_v) {
            for (int k = 0; k < cands[i]->n_tool_calls; k++) {
                name_sig[i] += cands[i]->tool_calls[k].name ? cands[i]->tool_calls[k].name : "?";
                name_sig[i] += ';';
            }
        }

        int lead = -1, lead_votes = 0;
        for (int i : tools_v) {
            int votes = 0;
            for (int j : tools_v) if (name_sig[j] == name_sig[i]) votes++;
            if (votes > lead_votes) { lead_votes = votes; lead = i; }   /* > keeps lowest index */
        }
        if (lead < 0) return pick;

        /* The group that agreed on WHICH tool to call. */
        std::vector<int> group;
        for (int i : tools_v) if (name_sig[i] == name_sig[lead]) group.push_back(i);

        int exact = 0;
        for (int i : group) if (sig[i] == sig[lead]) exact++;

        int best = lead;
        bool by_medoid = false;
        if (exact * 2 > (int) group.size()) {
            best = lead;                     /* a real majority on the exact args */
        } else if (group.size() > 1) {
            /* Same tool, differing arguments — take the most central argument set
             * rather than declaring no consensus and falling back to sample 0. */
            double best_score = -1.0;
            for (int i : group) {
                double score = 0.0;
                for (int j : group)
                    if (i != j) score += reuse_similarity(sig[i].c_str(), sig[j].c_str());
                if (score > best_score) { best_score = score; best = i; }
            }
            by_medoid = true;
        }

        pick.winner = best;
        pick.votes  = lead_votes;
        {
            char buf[192];
            const char *nm = cands[best]->tool_calls[0].name ? cands[best]->tool_calls[0].name : "?";
            if (by_medoid)
                snprintf(buf, sizeof buf, "%d/%d chose %s, args varied (medoid)",
                         lead_votes, pick.n_valid, nm);
            else
                snprintf(buf, sizeof buf, "%d/%d agreed on %s", lead_votes, pick.n_valid, nm);
            pick.reason = bo_dup(buf);
        }
        return pick;
    }

    /* Text mode: medoid. Sum similarity to every other text candidate and take
     * the most central one — the answer the samples collectively agree on. */
    std::vector<int> texts;
    for (int i : valid) if (cands[i]->n_tool_calls == 0) texts.push_back(i);
    if (texts.size() == 1) {
        pick.winner = texts[0];
        pick.votes  = 1;
        pick.reason = bo_dup("single valid answer");
        return pick;
    }

    int    best = texts[0];
    double best_score = -1.0;
    for (int i : texts) {
        double score = 0.0;
        for (int j : texts)
            if (i != j) score += reuse_similarity(text[i].c_str(), text[j].c_str());
        if (score > best_score) { best_score = score; best = i; }   /* > keeps lowest index */
    }
    pick.winner = best;
    pick.votes  = 1;                       /* medoid is a ranking, not a vote count */
    {
        char buf[160];
        double avg = texts.size() > 1 ? best_score / (double) (texts.size() - 1) : 0.0;
        snprintf(buf, sizeof buf, "medoid of %d answers (avg sim %.2f)", (int) texts.size(), avg);
        pick.reason = bo_dup(buf);
    }
    return pick;
}

extern "C" void bestof_free(BestOfPick *p) {
    if (!p) return;
    free(p->reason);
    p->reason = nullptr;
}
