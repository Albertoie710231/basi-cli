#ifndef BASI_WEB_H
#define BASI_WEB_H

/* web_search: ranked {title,url,snippet} list for a query. time_filter is
   NULL or one of "day"/"week"/"month"/"year". Returns malloc'd output. */
char *execute_web_search(const char *query, const char *time_filter);

/* web_fetch: fetch + extract the readable text of ONE url. SSRF-guarded,
   curl-first, handles PDFs. Returns malloc'd output. */
char *execute_web_fetch(const char *url);

/* Best-effort auto-start of the local SearXNG instance web_search uses.
   Call once at startup; non-blocking. */
/* What web_search's backend looks like this run. UNAVAILABLE is the one the
   caller must act on: advertising a search tool that cannot work invites the
   model to spend turns discovering that, and to read the failures as "the web
   has nothing on this". */
typedef enum {
    WEB_SEARCH_UP = 0,       /* an instance answered the probe */
    WEB_SEARCH_STARTING,     /* none was up; a local install was launched */
    WEB_SEARCH_UNAVAILABLE   /* nothing reachable, and nothing to start */
} WebSearchStatus;

WebSearchStatus web_ensure_searxng(void);

/* readfile: read a local document (md/pdf/docx/odt/epub/text). Optional regex. */
char *execute_readfile(const char *path, const char *pattern);

#endif /* BASI_WEB_H */
