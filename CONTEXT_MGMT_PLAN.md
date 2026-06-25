# Context Management (Reclamation) — design

**Goal.** Keep an interactive BASI session alive when the KV cache fills, instead
of hitting the `[Context limit reached]` wall (`model.c:184`) with no recovery.
Built on the occupancy meter (commit `0cf26bc`): now that we can *see* the fill
exactly, we reclaim space before we run out.

**Not malloc-OOM.** The KV cache and compute buffers are allocated at load and do
not grow mid-session; what grows is the KV *fill level*. So "reclamation" = freeing
KV occupancy, not freeing host/VRAM memory.

---

## Strategy (decided after investigating opencode / Claude Code / Codex)

All three agentic harnesses **summarize**, not evict — pure eviction silently drops
the goal/decisions/file-paths and the agent forgets its task mid-loop. We adopt the
same shape, adapted to our local KV cache:

- **Verbatim recent window** kept as-is (most recent ~`KEEP` tokens).
- **Anchored rolling summary** of everything older (one LLM pass, updated in place).
- **Exact, free trigger**: we own the tokenizer + KV, so the meter is ground truth
  — no `chars/4` estimate, no fat safety buffer needed (opencode needs both because
  the provider only reports usage after the fact).

We build it in phases so the **deterministic, reliable** mechanism lands and is
testable before the LLM-assisted tier (deterministic-first motto: the trigger,
window selection, and KV resync are all deterministic; only the summary text is the
irreducible LLM step).

---

## The core invariant (this is where the risk is)

Today (`main.c`): `messages[]` grows; each turn renders the *whole* history into
`formatted_buf`, then feeds only the suffix `formatted_buf + prev_len` to
`generate()`, trusting the KV cache to already hold the `prev_len` prefix. KV grows
monotonically.

**Any reclamation rewrites `messages[]`, which changes the rendered prefix — so the
KV cache (holding the OLD prefix) is now invalid.** After every reclamation we MUST
resync: `llama_memory_clear(...)`, `prev_len = 0`, re-render. The next `generate()`
then decodes the whole (small, compacted) prompt once. This is exactly what `/clear`
already does (`main.c:1207`), except we keep the recent tail + a summary instead of
dropping everything. The one-time re-decode is bounded by the compacted size (~KEEP
+ summary), which is the point.

**Latent bug to fix first (Phase 0).** `prev_len` is advanced unconditionally after
`generate()` (`main.c:1781`) to the full render length — even when `generate()`
bailed at the context guard *without decoding*. Then next turn `prompt =
formatted_buf + prev_len` skips never-decoded tokens → silent context corruption.
Reclamation makes the guard unreachable in normal use (we reclaim first), but the
resync logic must be correct regardless.

---

## Phase 0 — correctness floor (small, independent)
- Make `prev_len` always reflect what is *actually* in the KV cache. Concretely:
  centralize "resync after history mutation" into one helper
  `kv_resync_full(ctx, &prev_len)` = `llama_memory_clear` + `prev_len = 0`, and call
  it from `/clear` (replacing the inline code) and from reclamation.
- Guarantee `generate()` is never entered without room: check the meter before the
  turn; if a single delta cannot fit even after reclamation, truncate it (tool
  results already cap at `MAX_TOOL_RESULT_SZ`; extend the same guard to oversized
  user pastes) rather than half-decoding.

## Phase 1 — deterministic recent-window reclamation (no LLM yet)
Trigger (checked before feeding a turn, and after each tool iteration):
```
used = context_used_tokens(ctx)          // exact, from the meter
if (used > n_ctx - RESERVE) reclaim();    // RESERVE scaled to local ctx
```
- `RESERVE` ≈ answer budget + margin. For a 32k local ctx, ~4k (not opencode's 20k,
  which assumes 128k+). Tunable; surfaced in a `[Compacting context...]` notice.
- `reclaim()` (deterministic):
  1. Pin `messages[0]` (system prompt) — never evict.
  2. Walk newest→oldest summing token estimates of message contents until `KEEP`
     (~half of `n_ctx - RESERVE`); everything newer = keep verbatim.
  3. Drop the older messages from `messages[]` **on whole-turn boundaries** — never
     split a `tool_call`/`tool_result` pair (they must stay adjacent or both go).
  4. `kv_resync_full()`, re-render, continue. The dropped turns are gone (Phase 1
     forgets them; Phase 2 will summarize them instead).
- This alone keeps the session alive forever. Loses old detail verbatim — acceptable
  as the reliable floor; Phase 2 fixes the forgetting.

## Phase 2 — anchored rolling summary (the LLM tier)
- Before dropping the old head in `reclaim()`, run ONE quiet `generate()`
  (`generate_quiet=1`, capped ~1k tokens) over the head using opencode's template
  (Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context /
  Relevant Files).
- Store it as a single reserved-role `summary` message right after the system prompt
  (rendered as a `<conversation-checkpoint>` block, like opencode). Keep the recent
  window verbatim after it.
- **Anchored**: subsequent reclaims *update* the existing summary (preserve
  still-true, drop stale, merge new) rather than re-summarizing from scratch.
- Reuses the deepsearch quiet-round machinery; `chat_tmpl` already has the
  reserved-role decode path (`tool_call`/`tool_result`) to extend.

## Phase 3 (optional, later) — KV surgery instead of clear+redecode
- Use `llama_memory_seq_rm` + `llama_memory_seq_add` (context-shift) to keep the
  system prefix + recent tail resident and avoid re-decoding them. Cheaper but
  fiddly (RoPE re-base, stale attention to dropped tokens). Not needed for
  correctness; a latency optimization once Phases 1–2 are proven.

---

## Edge cases
- **First user turn alone exceeds budget** (huge paste): truncate the user message
  to fit, with a visible `[Truncated input: N → M tokens]` notice.
- **Native-tool floor is high**: system prompt + 21 tool schemas ≈ 2.3k tokens on a
  4k ctx (measured). `RESERVE` + `KEEP` must leave room above that floor; on tiny
  contexts, reclamation may keep only the last 1–2 turns. Surface it, don't fail.
- **Plan mode**: an active plan's invariants live in `messages`/`.basi/`; ensure the
  summary preserves the current plan slug + phase (it's in the system/env block,
  which is pinned, so OK — verify).

## Test plan (deterministic, scriptable)
- Force compaction fast: `-c 2048 --yolo -p` is awkward (one-shot = one turn). Add a
  scripted multi-turn harness via stdin to the interactive REPL, OR a hidden
  `BASI_TEST_CTX` small-ctx + a loop of turns, asserting: (a) session survives past
  `n_ctx`, (b) the meter drops after `[Compacting]`, (c) a fact stated before
  compaction is still answerable after (Phase 2), (d) no `[Context limit reached]`.
- Unit-test `reclaim()` window selection offline (message list in → kept list out)
  with no model, like the edit-engine tests.

## Build order
1. **Phase 0** (resync helper + pre-turn room guard) — correctness, ~small diff.
2. **Phase 1** (trigger + deterministic window eviction) — session stays alive.
3. **Phase 2** (anchored summary) — stops forgetting.
4. Phase 3 only if latency of clear+redecode proves to matter.
