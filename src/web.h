#ifndef BASI_WEB_H
#define BASI_WEB_H

/* webfetch: 2 quoted args (search query, grep regex). Returns malloc'd output. */
char *execute_webfetch(const char *search_query, const char *grep_query);

/* readfile: read a local document (md/pdf/docx/odt/epub/text). Optional regex. */
char *execute_readfile(const char *path, const char *pattern);

#endif /* BASI_WEB_H */
