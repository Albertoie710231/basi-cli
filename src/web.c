#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <strings.h>

#include "util.h"
#include "globals.h"
#include "web.h"

/* ── Webfetch: search + parallel fetch + grep ──────────────────────── */

#define WEBFETCH_MAX_RESULTS   5
#define WEBFETCH_MAX_CHARS  2500  /* max chars of extracted content per page */
#define WEBFETCH_MIN_LINE_LEN 30  /* skip short lines (nav, buttons, junk) */
#define READFILE_MAX_CHARS   5000  /* cap on extracted document text per call */
#define WEBFETCH_PDF_CHARS   2000  /* budget for an inlined PDF attached to an HTML result */

#define MINI_BROWSER "/home/alberto/Documentos/MINI_BROWSER/build/mini_browser"
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

/* Split a URL into its scheme+host (e.g. "https://arxiv.org") and its base
   directory (URL truncated to the last '/', e.g. "https://arxiv.org/abs/").
   Relative hrefs on that page get resolved against these two prefixes. */
static void url_split_base(const char *url,
                           char *scheme_host, size_t shsz,
                           char *base_dir, size_t bdsz) {
    scheme_host[0] = '\0';
    base_dir[0]    = '\0';

    const char *sep = strstr(url, "://");
    if (!sep) return;
    const char *host_start = sep + 3;
    const char *path_start = strchr(host_start, '/');

    size_t shlen = path_start ? (size_t)(path_start - url) : strlen(url);
    if (shlen >= shsz) shlen = shsz - 1;
    memcpy(scheme_host, url, shlen);
    scheme_host[shlen] = '\0';

    if (path_start) {
        const char *last_slash = strrchr(path_start, '/');
        if (last_slash && last_slash >= path_start) {
            size_t bdlen = (size_t)(last_slash - url) + 1;
            if (bdlen >= bdsz) bdlen = bdsz - 1;
            memcpy(base_dir, url, bdlen);
            base_dir[bdlen] = '\0';
            return;
        }
    }
    snprintf(base_dir, bdsz, "%s/", scheme_host);
}

/*
 * Fetch a URL, strip HTML, then use awk to extract whole paragraphs around matches.
 *
 * A paragraph is a run of consecutive lines each of length >= minlen; any
 * shorter/blank line ends the current paragraph. The awk script:
 * - Builds the paragraph array from the stream
 * - For small pages (<=5000 usable chars), prints every paragraph
 * - For large pages, prints every paragraph whose text matches the pattern
 * - Each paragraph is emitted at most once — multiple hits inside the same
 *   paragraph do not duplicate it
 * - Separates distinct paragraphs with "---"
 * - Stops after MAX_CHARS total output (hard cap via head -c)
 */
/* fetch_and_extract: if `claim_dir` is non-NULL and non-empty, parallel
   workers coordinate PDF-fetch dedupe by atomically mkdir'ing a subdirectory
   named after the md5 of the resolved PDF URL. The first worker wins; losers
   skip the fetch. Pass NULL/empty to disable dedupe. */
static char *fetch_and_extract(const char *url, const char *pattern,
                                const char *claim_dir) {
    char cmd[16384];

    /* Debug trace lines emitted to stderr (popen only captures stdout, so
       stderr reaches the user's terminal alongside the [Fetching...] banner).
       Single quotes in URLs are already rejected at webfetch entry, so
       inlining `url` between single quotes here is safe. */
    char dbg_start[1024]  = "";
    char dbg_nopdf[256]   = "";
    if (debug_mode) {
        snprintf(dbg_start, sizeof(dbg_start),
            "echo '[wf/%s] %s' >&2; ",
            url_has_pdf_suffix(url) ? "pdf " : "html", url);
        snprintf(dbg_nopdf, sizeof(dbg_nopdf),
            "echo '[wf/html]   (no pdf link on page)' >&2; ");
    }

    /* Claim guard: either a real `if mkdir` race-gate, or a no-op `if true`.
       Hash is computed from a normalized URL so that equivalent links
       (trailing '.pdf', '/', or '?query') collapse to the same key —
       arxiv.org/pdf/ID and arxiv.org/pdf/ID.pdf resolve to the same
       document and shouldn't both be fetched. */
    char claim_begin[1024];
    char claim_end[1024];
    if (claim_dir && claim_dir[0]) {
        snprintf(claim_begin, sizeof(claim_begin),
            "pdfhash=$(printf '%%s' \"$pdfurl\" "
            " | sed -E 's|[?#].*||; s|\\.pdf$||i; s|/$||' "
            " | md5sum | cut -c1-32); "
            "if mkdir \"%s/$pdfhash\" 2>/dev/null; then %s",
            claim_dir,
            debug_mode
              ? "echo \"[wf/html]   -> pdf-follow: $pdfurl\" >&2; "
              : "");
        snprintf(claim_end, sizeof(claim_end),
            "else %s:; fi",
            debug_mode
              ? "echo \"[wf/html]   (pdf already claimed by another worker: $pdfurl)\" >&2; "
              : "");
    } else {
        snprintf(claim_begin, sizeof(claim_begin),
            "if true; then %s",
            debug_mode
              ? "echo \"[wf/html]   -> pdf-follow: $pdfurl\" >&2; "
              : "");
        snprintf(claim_end, sizeof(claim_end), "fi");
    }

    if (url_has_pdf_suffix(url)) {
        /* PDF branch: raw bytes via curl → pdftotext → paragraph extract. */
        snprintf(cmd, sizeof(cmd),
            "%s"
            "curl -sL -m 20 --max-filesize 20000000 "
            "-A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36' "
            "'%s' 2>/dev/null "
            "| pdftotext -enc UTF-8 - - 2>/dev/null "
            "| " AWK_EXTRACT_PARAGRAPHS
            " | head -c %d",
            dbg_start,
            url, pattern, WEBFETCH_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
            debug_mode ? 1 : 0, WEBFETCH_MAX_CHARS);

        char *result = run_command(cmd, WEBFETCH_MAX_CHARS + 256);
        if (result && strlen(result) > (size_t)WEBFETCH_MAX_CHARS) {
            result[WEBFETCH_MAX_CHARS] = '\0';
        }
        return result;
    }

    /* HTML branch: mini_browser (bypasses CAPTCHAs that hit plain curl) →
       save raw HTML to a tempfile so we can both flatten it with w3m AND
       scan the hrefs for a PDF link. If one is found we follow it (one PDF
       max per landing page) and append its extracted text to the result.

       PDF-picking priority:
         1. link host matches the landing page's host (e.g. arxiv→arxiv)
         2. anchor text/href contains 'download', 'full', or 'paper'
         3. first PDF link on the page
       Relative hrefs are resolved against the landing page's scheme+host and
       base directory. */
    char scheme_host[512];
    char base_dir[1024];
    url_split_base(url, scheme_host, sizeof(scheme_host),
                       base_dir,    sizeof(base_dir));

    snprintf(cmd, sizeof(cmd),
        "%s"  /* dbg_start */
        "tmp=$(mktemp 2>/dev/null) && "
        MINI_BROWSER " --headless --timeout 15000 '%s' 2>/dev/null > \"$tmp\"; "

        /* 1) Paragraph-extract the landing page's visible text. */
        "w3m -dump -T text/html -cols 120 < \"$tmp\" 2>/dev/null "
        "| " AWK_EXTRACT_PARAGRAPHS
        " | head -c %d; "

        /* 2) Pick the best PDF link from the raw HTML. The grep catches both
              '.pdf' endings and '/pdf/' path segments (arxiv-style links
              like arxiv.org/pdf/2301.00001 have no extension). Awk then ranks
              by host match, then by 'download|full|paper' href hint, then
              first-seen. */
        "pdf=$(grep -oiE 'href=\"[^\"]*(\\.pdf|/pdf/)[^\"]*\"' \"$tmp\" 2>/dev/null "
        " | sed -E 's/^href=\"//; s/\"$//' "
        " | awk -v host='%s' '"
        "     BEGIN { IGNORECASE=1 }"
        "     {"
        "       h = $0;"
        "       if (same==\"\" && index(h, host)==1) same=h;"
        "       else if (prio==\"\" && h ~ /download|full|paper/) prio=h;"
        "       else if (first==\"\") first=h;"
        "     }"
        "     END { print (same != \"\" ? same : (prio != \"\" ? prio : first)) }'"
        "); "

        /* 3) Resolve relative URLs against scheme+host / base_dir, then fetch
              the PDF, extract, and append. Skip on empty. */
        "if [ -n \"$pdf\" ]; then "
        "  case \"$pdf\" in "
        "    http://*|https://*) pdfurl=\"$pdf\" ;; "
        "    //*)                pdfurl=\"https:$pdf\" ;; "
        "    /*)                 pdfurl='%s'\"$pdf\" ;; "
        "    *)                  pdfurl='%s'\"$pdf\" ;; "
        "  esac; "
        "  %s"  /* claim_begin — opens dedupe race-gate (or `if true`);
                   debug line for "pdf-follow" lives inside the winner side */
        "    printf '\\n--- PDF: %%s ---\\n' \"$pdfurl\"; "
        "    curl -sL -m 20 --max-filesize 20000000 "
        "      -A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36' "
        "      \"$pdfurl\" 2>/dev/null "
        "    | pdftotext -enc UTF-8 - - 2>/dev/null "
        "    | " AWK_EXTRACT_PARAGRAPHS
        "    | head -c %d; "
        "  %s; "  /* claim_end — closes race-gate */
        "else "
        "  %s"  /* dbg_nopdf */
        "  :; "
        "fi; "
        "rm -f \"$tmp\"",
        dbg_start,
        url,
        /* 1st awk */ pattern, WEBFETCH_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
                     debug_mode ? 1 : 0, WEBFETCH_MAX_CHARS,
        /* pdf picker */ scheme_host,
        /* case resolution */ scheme_host, base_dir,
        claim_begin,
        /* 2nd awk */ pattern, WEBFETCH_PDF_CHARS, WEBFETCH_MIN_LINE_LEN,
                     debug_mode ? 1 : 0, WEBFETCH_PDF_CHARS,
        claim_end,
        dbg_nopdf);

    char *result = run_command(cmd, WEBFETCH_MAX_CHARS + WEBFETCH_PDF_CHARS + 512);
    /* Hard truncate if still too long */
    size_t hard_cap = (size_t)(WEBFETCH_MAX_CHARS + WEBFETCH_PDF_CHARS);
    if (result && strlen(result) > hard_cap) {
        result[hard_cap] = '\0';
    }
    return result;
}

/* webfetch <search_query> <grep_query>
   1. Search DDG for search_query → 5 URLs
   2. Fetch all 5 in parallel with mini_browser
   3. Grep each for grep_query
   4. Return structured results */
char *execute_webfetch(const char *search_query, const char *grep_query) {
    char *encoded = url_encode(search_query);
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        MINI_BROWSER " --headless --timeout 30000 "
        "'https://lite.duckduckgo.com/lite/?q=%s' 2>/dev/null "
        "| grep -oP '(?<=class=\"result-link\" href=\")[^\"]+' "
        "| grep -v 'ad_domain' | grep -v 'ad_provider' "
        "| grep -v 'duckduckgo.com/duckduckgo-help-pages' | head -15",
        encoded);
    free(encoded);

    char *raw = run_command(cmd, 256 * 1024);
    if (!raw || !raw[0]) {
        free(raw);
        return strdup("No search results found.");
    }

    /* Parse DDG redirect URLs → collect up to 5 */
    char *urls[WEBFETCH_MAX_RESULTS];
    int url_count = 0;

    char *raw_copy = strdup(raw);
    char *line = strtok(raw_copy, "\n");
    while (line && url_count < WEBFETCH_MAX_RESULTS) {
        char *uddg = strstr(line, "uddg=");
        if (uddg) {
            uddg += 5;
            char *end = strchr(uddg, '&');
            if (!end) end = uddg + strlen(uddg);
            char saved = *end;
            *end = '\0';
            urls[url_count] = url_decode(uddg);
            *end = saved;
            url_count++;
        }
        line = strtok(NULL, "\n");
    }
    free(raw_copy);
    free(raw);

    if (url_count == 0)
        return strdup("No search results found.");

    printf("\033[90m[Fetching %d results in parallel, grepping for: %s]\033[0m\n",
           url_count, grep_query);
    fflush(stdout);

    /* Shared claim directory so parallel workers can dedupe PDF fetches.
       If mkdtemp fails we fall back to the no-dedupe path (empty string). */
    char claim_dir[] = "/tmp/basi-wf.XXXXXX";
    if (mkdtemp(claim_dir) == NULL) {
        claim_dir[0] = '\0';
    }

    /* Fork parallel fetchers */
    int pipes[WEBFETCH_MAX_RESULTS][2];
    pid_t pids[WEBFETCH_MAX_RESULTS];

    for (int i = 0; i < url_count; i++) {
        pipe(pipes[i]);
        pids[i] = fork();
        if (pids[i] == 0) {
            close(pipes[i][0]);
            char *content = fetch_and_extract(urls[i], grep_query, claim_dir);
            if (content && content[0]) {
                write(pipes[i][1], content, strlen(content));
            }
            close(pipes[i][1]);
            free(content);
            _exit(0);
        } else {
            close(pipes[i][1]);
        }
    }

    /* Parent: collect results from all children */
    StringBuf results;
    sb_init(&results);

    for (int i = 0; i < url_count; i++) {
        StringBuf child_out;
        sb_init(&child_out);
        char buf[4096];
        ssize_t n;
        while ((n = read(pipes[i][0], buf, sizeof(buf))) > 0) {
            sb_append(&child_out, buf, n);
        }
        close(pipes[i][0]);
        waitpid(pids[i], NULL, 0);

        char header[1024];
        snprintf(header, sizeof(header), "\n--- Result %d: %s ---\n", i + 1, urls[i]);
        sb_append_str(&results, header);

        if (child_out.len > 0) {
            size_t cap = child_out.len < (size_t)WEBFETCH_MAX_CHARS
                       ? child_out.len : (size_t)WEBFETCH_MAX_CHARS;
            sb_append(&results, child_out.data, cap);
        } else {
            sb_append_str(&results, "(no relevant content found)\n");
        }
        sb_free(&child_out);
    }

    /* Cleanup */
    for (int i = 0; i < url_count; i++)
        free(urls[i]);
    if (claim_dir[0]) {
        char rm_cmd[300];
        snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf '%s'", claim_dir);
        int rc = system(rm_cmd);
        (void)rc;
    }

    if (results.len == 0) {
        sb_free(&results);
        return strdup("No search results found.");
    }
    return sb_to_str(&results);
}

/* readfile <path> [regex]
   Reads a local document and returns its plain text. Dispatches by extension:
     .pdf        → pdftotext
     .docx       → unzip + XML strip (Word OOXML)
     .odt        → unzip + XML strip (OpenDocument)
     .epub       → unzip + XML strip across all xhtml/html files
     everything  → cat (treat as text: source code, markdown, txt, json, ...)
   Without regex, emits the first READFILE_MAX_CHARS bytes of extracted text.
   With regex, paragraph-extracts matching paragraphs (same logic as webfetch). */
char *execute_readfile(const char *path, const char *pattern) {
    if (strncmp(path, "http://", 7) == 0 ||
        strncmp(path, "https://", 8) == 0) {
        return strdup("Error: readfile is for local paths. URLs are fetched by webfetch.");
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
            extractor, pattern, READFILE_MAX_CHARS, WEBFETCH_MIN_LINE_LEN,
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
