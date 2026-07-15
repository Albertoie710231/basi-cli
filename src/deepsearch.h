#ifndef BASI_DEEPSEARCH_H
#define BASI_DEEPSEARCH_H

#include "basi_types.h"

/* deep_search: an IterResearch deep-research loop (ported from Alibaba
   DeepResearch / WebResearcher). Runs N rounds server-backed (generate_chat)
   built from `model` (so it never touches the main conversation's KV cache):
   each round the model emits Think/Report/Action; only the distilled Report
   carries forward, raw observations are dropped, keeping the context bounded.
   Web (web_search/web_fetch) and KB (docs_search/docs_get) tools back it; each
   visited source is distilled to goal-relevant evidence before it enters the
   report. Prints one-line round progress; the model's internal generations are
   silent (server-backed). Returns the malloc'd final cited answer (caller frees). */
char *execute_deep_search(const char *question);

#endif /* BASI_DEEPSEARCH_H */
