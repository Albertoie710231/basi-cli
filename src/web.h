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
void web_ensure_searxng(void);

/* readfile: read a local document (md/pdf/docx/odt/epub/text). Optional regex. */
char *execute_readfile(const char *path, const char *pattern);

#endif /* BASI_WEB_H */
