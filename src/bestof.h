#ifndef BASI_BESTOF_H
#define BASI_BESTOF_H
/* Best-of-N selection: given N sampled assistant turns from one prefill
 * (srvchat_complete_n), pick the winner DETERMINISTICALLY — no LLM judge, no
 * extra generation, no compile/test oracle.
 *
 * Why consensus rather than a judge: BASI's motto is that anything implementable
 * deterministically must be. An LLM judge would add a whole extra generation per
 * turn and, on a 9B, rank about as reliably as a coin. Consensus over independent
 * samples costs nothing and directly attacks the failure mode best-of-N exists to
 * fix — sampling noise, where one draw emits a malformed or off-target tool call
 * that three other draws got right.
 *
 * Two modes, chosen by what the candidates actually are:
 *
 *   TOOL turns  — exact-match majority vote on a canonical signature
 *                 (name + JSON args with sorted keys). Tool calls are a small
 *                 discrete space, so identical intent really does produce
 *                 identical signatures and voting is exact.
 *   TEXT turns  — medoid by reuse_similarity(): the candidate most similar to
 *                 all the others. Free-form prose almost never matches exactly,
 *                 so exact voting would degenerate to "pick the first"; the
 *                 medoid is the consensus answer with outliers discarded.
 *
 * Candidates that are structurally unusable (tool args that aren't valid JSON,
 * or an empty turn with no tool call) are dropped before voting — that filter
 * alone removes the degenerate draws that best-of-N is meant to survive. */
#include "srvchat.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int   winner;   /* index into cands of the chosen turn; -1 if none usable */
    int   votes;    /* candidates agreeing with the winner (1 = no consensus) */
    int   n_valid;  /* candidates that survived the validity filter */
    char *reason;   /* short human-readable explanation (malloc'd, may be NULL) */
} BestOfPick;

/* Select among cands[0..n). Never returns a winner that failed validity unless
 * every candidate failed, in which case winner is -1 and the caller should fall
 * back to its normal single-sample path. Deterministic: equal scores always
 * resolve to the lowest index, so the same N candidates always pick the same
 * winner. */
BestOfPick bestof_select(SrvChatResult *const *cands, int n);

void bestof_free(BestOfPick *p);

#ifdef __cplusplus
}
#endif
#endif /* BASI_BESTOF_H */
