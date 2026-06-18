#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <ctype.h>
#include <stdint.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "util.h"
#include "globals.h"
#include "web.h"

/* ── Local document extraction (shared by readfile) ────────────────── */

#define EXTRACT_MIN_LINE_LEN 30  /* skip short lines (nav, buttons, junk) */
#define READFILE_MAX_CHARS 5000  /* cap on extracted document text per call */

/* Awk paragraph extractor. Placeholders consumed (in order):
     %s  regex pattern
     %d  maxchars
     %d  minlen
     %d  dbg flag (0/1)
   Literal %d inside the script is written as %%d so snprintf leaves it alone. */
#define AWK_EXTRACT_PARAGRAPHS \
    "awk -v pat='%s' -v maxchars=%d -v minlen=%d -v dbg=%d '" \
    "{" \
    "  gsub(/^[[:space:]]+|[[:space:]]+$/, \"\");" \
    "  if (length($0) < minlen) {" \
    "    if (cur_len > 0) { paragraphs[++pn] = cur; cur = \"\"; cur_len = 0 }" \
    "    next" \
    "  }" \
    "  if (cur_len == 0) cur = $0; else cur = cur \"\\n\" $0;" \
    "  cur_len++;" \
    "  sz += length($0) + 1" \
    "}" \
    "END {" \
    "  if (cur_len > 0) paragraphs[++pn] = cur;" \
    "  IGNORECASE=1;" \
    "  if (sz <= 5000) {" \
    "    for (i=1; i<=pn; i++) print paragraphs[i];" \
    "    if (dbg) printf \"[DEBUG] %%d bytes, %%d paragraphs (small page, full output)\\n\", sz, pn > \"/dev/stderr\";" \
    "  } else {" \
    "    total=0; matches=0;" \
    "    for (i=1; i<=pn; i++) {" \
    "      if (total >= maxchars) break;" \
    "      if (paragraphs[i] ~ pat) {" \
    "        matches++;" \
    "        if (matches > 1) { print \"---\"; total+=4 }" \
    "        if (total >= maxchars) break;" \
    "        print paragraphs[i]; total+=length(paragraphs[i])+1;" \
    "      }" \
    "    }" \
    "    if (dbg) printf \"[DEBUG] %%d bytes page, %%d paragraphs, %%d matched, %%d bytes output\\n\", sz, pn, matches, total > \"/dev/stderr\";" \
    "  }" \
    "}'"

static bool url_has_pdf_suffix(const char *url) {
    size_t len = strlen(url);
    size_t end = len;
    for (size_t i = 0; i < len; i++) {
        if (url[i] == '?' || url[i] == '#') { end = i; break; }
    }
    if (end < 4) return false;
    return strncasecmp(url + end - 4, ".pdf", 4) == 0;
}

/* ════════════════════════════════════════════════════════════════════
   Phase 1: web_search + web_fetch — Claude/Codex-style web tooling.

   Two clean primitives the agent loop drives to "navigate and investigate":
     web_search(query)  -> ranked {title,url,snippet} list (no content fetch)
     web_fetch(url)     -> clean extracted text of ONE page (curl-first)

   Both fetch through an SSRF guard ported from Odysseus' _public_http_url:
   reject private/loopback/link-local/reserved/multicast targets and
   re-validate every redirect hop. curl is the fetch engine; JS rendering is a
   planned, gated tier-3 — no browser is launched on the common path.
   ════════════════════════════════════════════════════════════════════ */

#define WEB_UA "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " \
               "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
#define WEB_MAX_REDIRECTS       5
#define WEB_FETCH_MAX_CHARS  6000
#define WEB_SEARCH_MAX_RESULTS  8
#define WEB_SEARCH_FETCH_COUNT  3     /* top-N results auto-fetched in parallel */
#define WEB_SEARCH_FETCH_CHARS  3000  /* per-page char cap for the bundled fetch */

/* ── SSRF guard ─────────────────────────────────────────────────────── */

/* `a` is in host byte order. Covers RFC1918 + loopback + link-local + CGNAT +
   multicast/reserved + this-network, matching Odysseus' _is_private_address. */
static bool ipv4_is_private(uint32_t a) {
    uint8_t b0 = (a >> 24) & 0xff, b1 = (a >> 16) & 0xff;
    if (b0 == 0)   return true;                       /* 0.0.0.0/8      */
    if (b0 == 10)  return true;                       /* 10.0.0.0/8     */
    if (b0 == 127) return true;                       /* loopback       */
    if (b0 == 169 && b1 == 254) return true;          /* link-local     */
    if (b0 == 172 && (b1 & 0xf0) == 16) return true;  /* 172.16/12      */
    if (b0 == 192 && b1 == 168) return true;          /* 192.168/16     */
    if (b0 == 100 && (b1 & 0xc0) == 64) return true;  /* 100.64/10 CGNAT*/
    if (b0 >= 224) return true;                       /* multicast+rsvd */
    return false;
}

static bool ipv6_is_private(const struct in6_addr *a6) {
    const uint8_t *b = a6->s6_addr;
    bool zero = true;
    for (int i = 0; i < 16; i++) if (b[i]) { zero = false; break; }
    if (zero) return true;                            /* ::  unspecified*/
    bool loop = b[15] == 1;
    for (int i = 0; i < 15; i++) if (b[i]) { loop = false; break; }
    if (loop) return true;                            /* ::1 loopback   */
    static const uint8_t v4map[12] = {0,0,0,0,0,0,0,0,0,0,0xff,0xff};
    if (memcmp(b, v4map, 12) == 0) {                  /* ::ffff:a.b.c.d */
        uint32_t a = ((uint32_t)b[12] << 24) | ((uint32_t)b[13] << 16) |
                     ((uint32_t)b[14] << 8) | b[15];
        return ipv4_is_private(a);
    }
    if ((b[0] & 0xfe) == 0xfc) return true;           /* fc00::/7 ULA   */
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) return true; /* fe80::/10 */
    if (b[0] == 0xff) return true;                    /* ff00::/8 mcast */
    return false;
}

static char *str_tolower_dup(const char *s) {
    char *o = strdup(s);
    if (o) for (char *p = o; *p; p++) *p = (char)tolower((unsigned char)*p);
    return o;
}

static bool host_ends_with(const char *host, const char *suffix) {
    size_t hl = strlen(host), sl = strlen(suffix);
    return hl >= sl && strcmp(host + hl - sl, suffix) == 0;
}

/* True iff `host_raw` is safe to fetch: not an internal name and (after DNS
   resolution) maps only to public IPs. Mirrors Odysseus' _public_http_url. */
static bool host_is_public(const char *host_raw) {
    if (!host_raw || !host_raw[0]) return false;
    char *host = str_tolower_dup(host_raw);
    if (!host) return false;

    bool ok = true;
    if (strcmp(host, "localhost") == 0 || strcmp(host, "metadata") == 0 ||
        strcmp(host, "metadata.google.internal") == 0) {
        ok = false;
    } else if (host_ends_with(host, ".local") || host_ends_with(host, ".localhost") ||
               host_ends_with(host, ".internal") || host_ends_with(host, ".lan") ||
               host_ends_with(host, ".intranet")) {
        ok = false;
    }

    if (ok) {
        struct in_addr v4;
        struct in6_addr v6;
        if (inet_pton(AF_INET, host, &v4) == 1) {
            ok = !ipv4_is_private(ntohl(v4.s_addr));
        } else if (inet_pton(AF_INET6, host, &v6) == 1) {
            ok = !ipv6_is_private(&v6);
        } else {
            struct addrinfo hints;
            memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            struct addrinfo *res = NULL;
            if (getaddrinfo(host, NULL, &hints, &res) != 0 || !res) {
                ok = false;
            } else {
                bool any = false;
                for (struct addrinfo *p = res; p; p = p->ai_next) {
                    if (p->ai_family == AF_INET) {
                        any = true;
                        struct sockaddr_in *s = (struct sockaddr_in *)p->ai_addr;
                        if (ipv4_is_private(ntohl(s->sin_addr.s_addr))) { ok = false; break; }
                    } else if (p->ai_family == AF_INET6) {
                        any = true;
                        struct sockaddr_in6 *s = (struct sockaddr_in6 *)p->ai_addr;
                        if (ipv6_is_private(&s->sin6_addr)) { ok = false; break; }
                    }
                }
                if (!any) ok = false;
                freeaddrinfo(res);
            }
        }
    }
    free(host);
    return ok;
}

/* Validate scheme is http(s), extract the host, and reject characters that are
   unsafe inside a single-quoted shell argument (only ' and control chars can
   break out of single quotes; everything else is literal). */
static bool url_is_fetchable(const char *url, char *host_out, size_t hostsz) {
    if (!url) return false;
    for (const char *c = url; *c; c++) {
        if (*c == '\'' || (unsigned char)*c < 0x20 || (unsigned char)*c == 0x7f)
            return false;
    }
    const char *p;
    if (strncasecmp(url, "http://", 7) == 0)       p = url + 7;
    else if (strncasecmp(url, "https://", 8) == 0) p = url + 8;
    else return false;

    const char *end = p;
    while (*end && *end != '/' && *end != '?' && *end != '#') end++;
    const char *at = NULL;
    for (const char *q = p; q < end; q++) if (*q == '@') at = q;  /* strip userinfo */
    const char *hstart = at ? at + 1 : p;
    const char *hend;
    if (*hstart == '[') {                          /* [ipv6] literal */
        const char *close = hstart + 1;
        while (close < end && *close != ']') close++;
        hstart++;
        hend = close;
    } else {
        hend = hstart;
        while (hend < end && *hend != ':') hend++;  /* drop :port */
    }
    size_t hlen = (size_t)(hend - hstart);
    if (hlen == 0 || hlen >= hostsz) return false;
    memcpy(host_out, hstart, hlen);
    host_out[hlen] = '\0';
    return true;
}

/* Fetch `url` into a fresh temp file, following redirects MANUALLY so every hop
   is re-validated against the SSRF guard (curl's own -L would skip the check).
   On success returns the temp-file path (caller unlinks + frees) and fills
   final_url / content_type (caller frees). Returns NULL on any failure. */
static char *web_curl_to_tmp(const char *url, char **final_url, char **content_type) {
    char *current = strdup(url);
    if (!current) return NULL;

    char tmpl[] = "/tmp/basi-web.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) { free(current); return NULL; }
    close(fd);

    char *ctype = NULL;
    bool ok = false;
    char host[256];

    for (int hop = 0; hop <= WEB_MAX_REDIRECTS; hop++) {
        if (!url_is_fetchable(current, host, sizeof(host)) || !host_is_public(host))
            break;

        char cmd[8192];
        snprintf(cmd, sizeof(cmd),
            "curl -sS --compressed --max-redirs 0 -m 20 --max-filesize 20000000 "
            "-A '" WEB_UA "' -o '%s' "
            "-w '%%{http_code}|%%{content_type}|%%{redirect_url}' '%s' 2>/dev/null",
            tmpl, current);

        char *meta = run_command(cmd, 4096);
        if (!meta) break;

        char *bar1 = strchr(meta, '|');
        char *bar2 = bar1 ? strchr(bar1 + 1, '|') : NULL;
        if (!bar1 || !bar2) { free(meta); break; }
        *bar1 = '\0';
        *bar2 = '\0';
        int code = atoi(meta);
        const char *ct = bar1 + 1;
        char *redir = bar2 + 1;
        char *nl = strchr(redir, '\n');
        if (nl) *nl = '\0';

        if (code >= 300 && code < 400 && redir[0]) {
            char *next = strdup(redir);          /* curl resolved it to absolute */
            free(meta);
            free(current);
            current = next;
            if (!current) break;
            continue;
        }

        free(ctype);
        ctype = strdup(ct);
        ok = (code >= 200 && code < 300);
        free(meta);
        break;
    }

    if (!ok) {
        unlink(tmpl);
        free(current);
        free(ctype);
        return NULL;
    }
    if (final_url) *final_url = current; else free(current);
    if (content_type) *content_type = ctype; else free(ctype);
    return strdup(tmpl);
}

/* ── tier-0 host rewrites ───────────────────────────────────────────── */

/* reddit.com / www.reddit.com → old.reddit.com (server-rendered HTML curl can
   read). Discourse/.json rewrites are deferred to the metasearch phase. */
static char *web_rewrite_host(const char *url) {
    const char *scheme, *rest;
    if (strncasecmp(url, "https://", 8) == 0)      { scheme = "https://"; rest = url + 8; }
    else if (strncasecmp(url, "http://", 7) == 0)  { scheme = "http://";  rest = url + 7; }
    else return strdup(url);

    const char *slash = rest;
    while (*slash && *slash != '/' && *slash != '?' && *slash != '#') slash++;
    size_t hl = (size_t)(slash - rest);
    if ((hl == 10 && strncasecmp(rest, "reddit.com", 10) == 0) ||
        (hl == 14 && strncasecmp(rest, "www.reddit.com", 14) == 0)) {
        size_t need = strlen(scheme) + 14 + strlen(slash) + 1;
        char *o = malloc(need);
        if (o) snprintf(o, need, "%sold.reddit.com%s", scheme, slash);
        return o;
    }
    return strdup(url);
}

/* ── search results ─────────────────────────────────────────────────── */

typedef struct {
    char  *url;
    char  *title;
    char  *snippet;
    char  *age;
    double score;
} SearchResult;

static void search_result_free(SearchResult *r) {
    free(r->url);     r->url = NULL;
    free(r->title);   r->title = NULL;
    free(r->snippet); r->snippet = NULL;
    free(r->age);     r->age = NULL;
}

/* ── ranking (port of Odysseus rank_search_results) ─────────────────── */

static bool has_word(const char *hay_lc, const char *term_lc) {
    size_t tl = strlen(term_lc);
    if (tl == 0) return false;
    for (const char *p = hay_lc; (p = strstr(p, term_lc)) != NULL; p++) {
        bool lb = (p == hay_lc) || !isalnum((unsigned char)p[-1]);
        bool rb = !isalnum((unsigned char)p[tl]);
        if (lb && rb) return true;
    }
    return false;
}

static int tokenize_query(const char *q, char tokens[][64], int max) {
    int n = 0;
    const char *p = q;
    while (*p && n < max) {
        while (*p && !isalnum((unsigned char)*p)) p++;
        const char *s = p;
        while (*p && isalnum((unsigned char)*p)) p++;
        if (p > s) {
            size_t l = (size_t)(p - s);
            if (l > 63) l = 63;
            for (size_t i = 0; i < l; i++) tokens[n][i] = (char)tolower((unsigned char)s[i]);
            tokens[n][l] = '\0';
            n++;
        }
    }
    return n;
}

static double domain_score_of(const char *url) {
    const char *h = strstr(url, "://");
    if (!h) return 0.4;
    h += 3;
    const char *e = h;
    while (*e && *e != '/' && *e != ':') e++;
    char dom[256];
    size_t l = (size_t)(e - h);
    if (l >= sizeof(dom)) l = sizeof(dom) - 1;
    for (size_t i = 0; i < l; i++) dom[i] = (char)tolower((unsigned char)h[i]);
    dom[l] = '\0';
    if (host_ends_with(dom, ".edu") || host_ends_with(dom, ".gov")) return 1.0;
    if (host_ends_with(dom, ".org")) return 0.7;
    return 0.4;
}

/* Developer/preview hosts (dev., tip., beta., nightly, canary, …) document the
   NEXT, unreleased version and poison "latest stable" answers — e.g.
   dev.golang.org/release is full of the in-development Go 1.27 while the stable
   is 1.26.4. Detect them so the ranker sinks them: they stay in the results list
   but drop out of the auto-fetched top set. ("dev." won't match "developer." —
   the trailing dot is required.) */
static bool is_preview_host(const char *url) {
    const char *h = strstr(url, "://");
    if (!h) return false;
    h += 3;
    char host[256];
    size_t i = 0;
    for (; h[i] && h[i] != '/' && h[i] != ':' && i < sizeof(host) - 1; i++)
        host[i] = (char)tolower((unsigned char)h[i]);
    host[i] = '\0';
    if (strncmp(host, "dev.", 4) == 0)     return true;
    if (strncmp(host, "tip.", 4) == 0)     return true;
    if (strncmp(host, "beta.", 5) == 0)    return true;
    if (strncmp(host, "next.", 5) == 0)    return true;
    if (strncmp(host, "staging.", 8) == 0) return true;
    if (strstr(host, ".beta.")) return true;
    if (strstr(host, "nightly")) return true;
    if (strstr(host, "canary"))  return true;
    return false;
}

/* Score = 2·title + 1·snippet + 1.5·domain, minus a big penalty for
   developer/preview hosts. Recency/news adjustments omitted (no ages yet). */
static void rank_results(const char *query, SearchResult *r, int n) {
    char toks[16][64];
    int nt = tokenize_query(query, toks, 16);
    for (int i = 0; i < n; i++) {
        char *tl = r[i].title   ? str_tolower_dup(r[i].title)   : strdup("");
        char *sl = r[i].snippet ? str_tolower_dup(r[i].snippet) : strdup("");
        double tscore = 0.0, sscore = 0.0;
        if (nt > 0 && tl && sl) {
            int th = 0, sh = 0;
            for (int k = 0; k < nt; k++) {
                if (has_word(tl, toks[k])) th++;
                if (has_word(sl, toks[k])) sh++;
            }
            tscore = (double)th / nt;
            double lenf = r[i].snippet ? (double)strlen(r[i].snippet) : 0.0;
            if (lenf > 200.0) lenf = 200.0;
            lenf /= 200.0;
            sscore = (lenf + (double)sh / nt) / 2.0;
        }
        r[i].score = 2.0 * tscore + 1.0 * sscore + 1.5 * domain_score_of(r[i].url);
        if (is_preview_host(r[i].url)) r[i].score -= 100.0;  /* sink dev/preview hosts */
        free(tl);
        free(sl);
    }
    for (int i = 1; i < n; i++) {          /* insertion sort, descending (n ≤ 8) */
        SearchResult key = r[i];
        int j = i - 1;
        while (j >= 0 && r[j].score < key.score) { r[j + 1] = r[j]; j--; }
        r[j + 1] = key;
    }
}

/* ── web_search (SearXNG JSON, port of Odysseus searxng_search_api) ──── */

/* The SearXNG instance to query. BASI is a thin client; the user runs SearXNG
   (Docker/native/remote) with the JSON format enabled. Default mirrors
   Odysseus' SEARXNG_INSTANCE. */
static const char *searxng_instance(void) {
    const char *env = getenv("SEARXNG_INSTANCE");
    return (env && env[0]) ? env : "http://localhost:8888";
}

/* Best-effort: make sure a SearXNG instance is reachable so web_search works
   out of the box. If the configured instance (SEARXNG_INSTANCE, default
   localhost:8888) isn't up and a local install is present (SEARXNG_HOME,
   default ~/Documentos/searxng), launch it in the background. Non-blocking:
   SearXNG boots while the model loads. Set BASI_NO_SEARXNG=1 to disable. */
void web_ensure_searxng(void) {
    if (getenv("BASI_NO_SEARXNG")) return;

    const char *inst = searxng_instance();
    if (strchr(inst, '\'')) return;   /* keep it shell-safe below */

    /* Quick reachability probe — connection-refused returns instantly. */
    char check[512];
    snprintf(check, sizeof(check),
        "curl -sS -m 2 -o /dev/null -w '%%{http_code}' '%s/' 2>/dev/null", inst);
    char *code = run_command(check, 32);
    bool up = code && atoi(code) > 0;
    free(code);
    if (up) return;

    /* Only ever auto-launch a local instance — never a remote URL the user set. */
    if (!strstr(inst, "localhost") && !strstr(inst, "127.0.0.1")) return;

    const char *home_env = getenv("SEARXNG_HOME");
    char home[512];
    snprintf(home, sizeof(home), "%s",
             (home_env && home_env[0]) ? home_env : "/home/alberto/Documentos/searxng");
    if (strchr(home, '\'')) return;

    char py[600];
    snprintf(py, sizeof(py), "%s/venv/bin/python", home);
    if (access(py, X_OK) != 0) return;   /* no local install — nothing to start */

    char launch[4096];
    snprintf(launch, sizeof(launch),
        "cd '%s' && SEARXNG_SETTINGS_PATH='%s/basi-settings.yml' PYTHONPATH='%s' "
        "'%s/venv/bin/python' -m searx.webapp >/tmp/searxng.log 2>&1 &",
        home, home, home, home);
    printf("\033[90m[web] starting local SearXNG at %s …\033[0m\n", inst);
    fflush(stdout);
    int rc = system(launch);
    (void)rc;
}

/* Fetch + extract ONE url into cleaned text capped at max_chars. Returns
   malloc'd content (NULL if nothing could be fetched/extracted). If
   final_url_out != NULL it receives the post-redirect URL (malloc'd; caller
   frees). Shared by execute_web_fetch and the web_search auto-fetch. */
static char *web_fetch_extract(const char *url, int max_chars, char **final_url_out) {
    if (final_url_out) *final_url_out = NULL;
    if (!url || !url[0]) return NULL;

    char *norm;
    if (strncasecmp(url, "http://", 7) == 0 || strncasecmp(url, "https://", 8) == 0) {
        norm = strdup(url);
    } else {
        size_t n = strlen(url) + 9;
        norm = malloc(n);
        if (norm) snprintf(norm, n, "https://%s", url);
    }
    if (!norm) return NULL;

    char *target = web_rewrite_host(norm);   /* tier 0 */
    free(norm);
    if (!target) return NULL;

    char host[256];
    if (!url_is_fetchable(target, host, sizeof(host))) { free(target); return NULL; }

    char *final_url = NULL, *ctype = NULL;
    char *body = web_curl_to_tmp(target, &final_url, &ctype);
    free(target);
    if (!body) { free(final_url); free(ctype); return NULL; }

    bool is_pdf = (ctype && strcasestr(ctype, "pdf")) || url_has_pdf_suffix(final_url);

    char cmd[2048];
    if (is_pdf) {
        snprintf(cmd, sizeof(cmd),
            "pdftotext -enc UTF-8 '%s' - 2>/dev/null | sed '/^[[:space:]]*$/d' | head -c %d",
            body, max_chars);
    } else {
        snprintf(cmd, sizeof(cmd),
            "w3m -dump -T text/html -cols 120 '%s' 2>/dev/null | sed '/^[[:space:]]*$/d' | head -c %d",
            body, max_chars);
    }
    char *content = run_command(cmd, (size_t)max_chars + 256);

    /* tier-2 thin fallback for HTML: <title> + og/meta description. */
    char *fallback = NULL;
    if (!is_pdf && (!content || strlen(content) < 200)) {
        char fcmd[2048];
        snprintf(fcmd, sizeof(fcmd),
            "{ grep -oiE '<title>[^<]*</title>' '%s' 2>/dev/null | sed -E 's/<[^>]+>//g' | head -1; "
            "  grep -oiE '<meta[^>]+(name|property)=\"(og:description|description)\"[^>]*>' '%s' 2>/dev/null "
            "    | grep -oiE 'content=\"[^\"]*\"' | sed -E 's/content=\"//I; s/\"$//' | head -1; }",
            body, body);
        fallback = run_command(fcmd, 4096);
    }

    unlink(body);
    free(body);
    free(ctype);

    char *result = NULL;
    if (content && content[0]) {
        if (fallback && fallback[0]) {
            StringBuf sb;
            sb_init(&sb);
            sb_append_str(&sb, content);
            sb_append_char(&sb, '\n');
            sb_append_str(&sb, fallback);
            result = sb_to_str(&sb);
        } else {
            result = strdup(content);
        }
    } else if (fallback && fallback[0]) {
        result = strdup(fallback);
    }
    free(content);
    free(fallback);

    if (final_url_out) *final_url_out = final_url; else free(final_url);
    return result;
}

/* Fetch up to WEB_SEARCH_FETCH_COUNT urls concurrently (fork + pipe), each
   capped at max_chars. out_content[i] gets malloc'd text or NULL. Wall-clock is
   ~one fetch, not the sum. Per-page cap keeps each well under the pipe buffer,
   so the parent can drain pipes sequentially without deadlock. */
static void web_fetch_parallel(char *const *urls, int n, int max_chars, char **out_content) {
    if (n > WEB_SEARCH_FETCH_COUNT) n = WEB_SEARCH_FETCH_COUNT;
    int pipes[WEB_SEARCH_FETCH_COUNT][2];
    pid_t pids[WEB_SEARCH_FETCH_COUNT];

    for (int i = 0; i < n; i++) {
        out_content[i] = NULL;
        if (pipe(pipes[i]) != 0) { pids[i] = -1; pipes[i][0] = pipes[i][1] = -1; continue; }
        pids[i] = fork();
        if (pids[i] == 0) {
            close(pipes[i][0]);
            char *c = web_fetch_extract(urls[i], max_chars, NULL);
            if (c && c[0]) { ssize_t w = write(pipes[i][1], c, strlen(c)); (void)w; }
            close(pipes[i][1]);
            _exit(0);
        } else if (pids[i] < 0) {
            close(pipes[i][0]);
            close(pipes[i][1]);
            pipes[i][0] = pipes[i][1] = -1;
        } else {
            close(pipes[i][1]);
        }
    }

    for (int i = 0; i < n; i++) {
        if (pids[i] <= 0 || pipes[i][0] < 0) continue;
        StringBuf sb;
        sb_init(&sb);
        char buf[4096];
        ssize_t r;
        while ((r = read(pipes[i][0], buf, sizeof(buf))) > 0) sb_append(&sb, buf, (size_t)r);
        close(pipes[i][0]);
        waitpid(pids[i], NULL, 0);
        if (sb.len) out_content[i] = sb_to_str(&sb);
        else sb_free(&sb);
    }
}

char *execute_web_search(const char *query, const char *time_filter) {
    if (!query || !query[0]) return strdup("Error: web_search requires a query.");

    const char *instance = searxng_instance();
    if (strchr(instance, '\'')) return strdup("Error: invalid SEARXNG_INSTANCE (contains a quote).");

    char *encoded = url_encode(query);
    if (!encoded) return strdup("Error: out of memory.");

    /* Map day/week/month/year (or d/w/m/y) → SearXNG time_range. */
    char tr[32] = "";
    if (time_filter && time_filter[0]) {
        const char *t = time_filter, *val = NULL;
        if      (!strcasecmp(t, "day")   || !strcasecmp(t, "d")) val = "day";
        else if (!strcasecmp(t, "week")  || !strcasecmp(t, "w")) val = "week";
        else if (!strcasecmp(t, "month") || !strcasecmp(t, "m")) val = "month";
        else if (!strcasecmp(t, "year")  || !strcasecmp(t, "y")) val = "year";
        if (val) snprintf(tr, sizeof(tr), "&time_range=%s", val);
    }

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "curl -sS --compressed -m 15 -A '" WEB_UA "' -H 'Accept: application/json' "
        "'%s/search?q=%s&format=json&language=en&safesearch=1%s' 2>/dev/null",
        instance, encoded, tr);
    free(encoded);

    char *json = run_command(cmd, 1024 * 1024);
    if (!json || json[0] != '{') {       /* unreachable, or HTML/error, not JSON */
        free(json);
        char msg[600];
        snprintf(msg, sizeof(msg),
            "No search results: could not get JSON from a SearXNG instance at %s.\n"
            "Run SearXNG with the JSON format enabled, or set SEARXNG_INSTANCE to one that is.",
            instance);
        return strdup(msg);
    }

    SearchResult results[WEB_SEARCH_MAX_RESULTS];
    memset(results, 0, sizeof(results));
    int n = 0;
    for (int i = 0; i < WEB_SEARCH_MAX_RESULTS; i++) {
        char path[40];
        snprintf(path, sizeof(path), "results.%d.url", i);
        char *url = jx_get_string(json, path);
        if (!url) break;                 /* past the end of the results array */
        if (!url[0]) { free(url); break; }

        snprintf(path, sizeof(path), "results.%d.title", i);
        char *title = jx_get_string(json, path);
        snprintf(path, sizeof(path), "results.%d.content", i);
        char *snippet = jx_get_string(json, path);

        results[n].url     = url;
        results[n].title   = title ? title : strdup("");
        results[n].snippet = snippet;    /* may be NULL */
        results[n].age     = NULL;
        results[n].score   = 0.0;
        n++;
    }
    free(json);
    if (n == 0) return strdup("No search results found.");

    rank_results(query, results, n);

    /* Auto-fetch the top results in parallel so the model gets real page text,
       not just snippets — snippets routinely omit or mis-state specific facts
       (e.g. a "latest version" number lives on the page, not in the blurb). */
    int nfetch = n < WEB_SEARCH_FETCH_COUNT ? n : WEB_SEARCH_FETCH_COUNT;
    char *fetched[WEB_SEARCH_FETCH_COUNT] = {0};
    char *fetch_urls[WEB_SEARCH_FETCH_COUNT];
    for (int i = 0; i < nfetch; i++) fetch_urls[i] = results[i].url;
    web_fetch_parallel(fetch_urls, nfetch, WEB_SEARCH_FETCH_CHARS, fetched);

    StringBuf out;
    sb_init(&out);
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
        "WEB SEARCH: %s\n%d results; the top %d pages are fetched below — prefer that "
        "page content over the snippets, and over your own prior knowledge.\n\n",
        query, n, nfetch);
    sb_append_str(&out, hdr);

    /* Ranked sources list (all results, snippet preview). */
    for (int i = 0; i < n; i++) {
        char head[160];
        snprintf(head, sizeof(head), "[%d] %s\n    %s\n", i + 1,
                 results[i].title && results[i].title[0] ? results[i].title : "(no title)",
                 results[i].url);
        sb_append_str(&out, head);
        if (results[i].snippet && results[i].snippet[0]) {
            sb_append_str(&out, "    ");
            const char *s = results[i].snippet;
            size_t slen = strlen(s);
            if (slen > 200) slen = 200;
            sb_append(&out, s, slen);
            if (strlen(s) > 200) sb_append_str(&out, "…");
            sb_append_char(&out, '\n');
        }
        sb_append_char(&out, '\n');
    }

    /* Fetched page content for the top results. */
    sb_append_str(&out, "==== FETCHED PAGE CONTENT (top results) ====\n");
    bool any = false;
    for (int i = 0; i < nfetch; i++) {
        if (!fetched[i] || !fetched[i][0]) continue;
        any = true;
        char lbl[96];
        snprintf(lbl, sizeof(lbl), "\n[CONTENT %d] %s\n", i + 1, results[i].url);
        sb_append_str(&out, lbl);
        sb_append_str(&out, fetched[i]);
        sb_append_char(&out, '\n');
    }
    if (!any)
        sb_append_str(&out, "(no page content could be extracted; rely on the snippets above)\n");

    for (int i = 0; i < n; i++) search_result_free(&results[i]);
    for (int i = 0; i < nfetch; i++) free(fetched[i]);
    return sb_to_str(&out);
}

/* ── web_fetch ──────────────────────────────────────────────────────── */

char *execute_web_fetch(const char *url) {
    if (!url || !url[0]) return strdup("Error: web_fetch requires a URL.");

    char *final_url = NULL;
    char *content = web_fetch_extract(url, WEB_FETCH_MAX_CHARS, &final_url);

    StringBuf out;
    sb_init(&out);
    sb_append_str(&out, "URL: ");
    sb_append_str(&out, final_url ? final_url : url);
    sb_append_str(&out, "\n--------\n");
    if (content && content[0]) {
        sb_append_str(&out, content);
    } else {
        sb_append_str(&out,
            "(could not fetch this URL, or no extractable text — it may be unreachable, "
            "blocked, a private/internal address, or require JavaScript.)");
    }

    free(content);
    free(final_url);
    return sb_to_str(&out);
}

/* readfile <path> [regex]
   Reads a local document and returns its plain text. Dispatches by extension:
     .pdf        → pdftotext
     .docx       → unzip + XML strip (Word OOXML)
     .odt        → unzip + XML strip (OpenDocument)
     .epub       → unzip + XML strip across all xhtml/html files
     everything  → cat (treat as text: source code, markdown, txt, json, ...)
   Without regex, emits the first READFILE_MAX_CHARS bytes of extracted text.
   With regex, paragraph-extracts matching paragraphs. */
char *execute_readfile(const char *path, const char *pattern) {
    if (strncmp(path, "http://", 7) == 0 ||
        strncmp(path, "https://", 8) == 0) {
        return strdup("Error: readfile is for local paths. URLs are fetched by web_fetch.");
    }
    if (strchr(path, '\'')) {
        return strdup("Error: readfile path must not contain single quotes.");
    }
    if (pattern && strchr(pattern, '\'')) {
        return strdup("Error: readfile regex must not contain single quotes.");
    }
    if (access(path, R_OK) != 0) {
        char *msg = malloc(512);
        snprintf(msg, 512, "Error: Cannot read file '%s'", path);
        return msg;
    }

    const char *ext = strrchr(path, '.');
    char extractor[2048];
    if (ext && strcasecmp(ext, ".pdf") == 0) {
        snprintf(extractor, sizeof(extractor),
            "pdftotext -enc UTF-8 '%s' - 2>/dev/null", path);
    } else if (ext && strcasecmp(ext, ".docx") == 0) {
        /* Turn paragraph closers into blank lines so the paragraph-extract
           awk still sees paragraph boundaries, then strip all remaining XML
           tags and a few common entities. */
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' word/document.xml 2>/dev/null "
            "| sed -E 's|</w:p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else if (ext && strcasecmp(ext, ".odt") == 0) {
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' content.xml 2>/dev/null "
            "| sed -E 's|</text:p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else if (ext && strcasecmp(ext, ".epub") == 0) {
        snprintf(extractor, sizeof(extractor),
            "unzip -p '%s' '*.xhtml' '*.html' 2>/dev/null "
            "| sed -E 's|</p>|\\n\\n|g; s|<[^>]+>||g; "
            "s/&amp;/\\&/g; s/&lt;/</g; s/&gt;/>/g; s/&quot;/\"/g; s/&apos;/'\\''/g'",
            path);
    } else {
        snprintf(extractor, sizeof(extractor), "cat '%s' 2>/dev/null", path);
    }

    char cmd[16384];
    if (pattern && pattern[0]) {
        snprintf(cmd, sizeof(cmd),
            "%s | " AWK_EXTRACT_PARAGRAPHS " | head -c %d",
            extractor, pattern, READFILE_MAX_CHARS, EXTRACT_MIN_LINE_LEN,
            debug_mode ? 1 : 0, READFILE_MAX_CHARS);
    } else {
        snprintf(cmd, sizeof(cmd), "%s | head -c %d", extractor, READFILE_MAX_CHARS);
    }

    char *result = run_command(cmd, READFILE_MAX_CHARS + 256);
    if (result && strlen(result) > (size_t)READFILE_MAX_CHARS) {
        result[READFILE_MAX_CHARS] = '\0';
    }
    if (!result || !result[0]) {
        free(result);
        return strdup("No text extracted from file (unsupported format, scanned, or empty).");
    }
    return result;
}
