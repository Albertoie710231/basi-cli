# Server-backend migration — handoff for the next session

**Direction (decided 2026-07-14):** move BASI off the native in-process `libllama`
link to a **Pi-style `llama-server` backend**. Why: advanced features (MTP
spec-decode, etc.) are only correct through the server; linking libllama costs an
ABI tax (every llama.cpp rebuild → `free(): invalid pointer`); the server is the
maintained, faster path. BASI's identity (reuse pillar — llama-independent — plus
agent loop, tools, prompts, deterministic-first) lives ABOVE the generation layer,
so the pivot sharpens "its own thing," doesn't dilute it.

## STATE: the core works. Branch `spec-decode-mtp`, latest `521892e`.

**DONE + TESTED (all committed):**
- **M1** `c3a3a3e` — `src/srvgen.{c,h}`: spawn llama-server (fork/exec + `/health`
  poll + process-group kill), stream `/completion` over `curl -N` SSE, parse
  `content` chunks. Proved: coherent streaming + **MTP lossless (23.6 t/s)**.
- **M2** `d22f032` — `basi_tool_grammar_json()` in chat_tmpl.cpp serializes BASI's
  tool GBNF (grammar+lazy+triggers) as spliceable `/completion` fields. Proved:
  server honors sent grammar (`root ::= "yes"|"no"` → `yes`).
- **generate() routed** `521892e` — `BASI_SERVER=1`: main.c loads model **vocab_only**
  (cheap; for templating/grammar/tokenization), spawns the server (weights+gen,
  +MTP from `BASI_SPEC`), `atexit`-kills it. `model.c generate()` → `generate_server()`.
  **PROVED end-to-end: `--yolo -p "list files using bash…"` → grammar-constrained
  tool call → BASI parsed it → `bash ls` executed → real listing fed back through
  the server → model summarized the repo.** Full agent loop functional server-backed.

## HOW TO BUILD + TEST
```
make -j"$(nproc)"     # needs LLAMA_DIR=~/llama.cpp (build_vulkan). NOTE: rebuild
                      # basi-cli after ANY llama.cpp rebuild (ABI coupling) or it
                      # crashes with free(): invalid pointer at startup.

# end-to-end agent over the server (MTP on):
BASI_SERVER=1 BASI_SPEC=draft-mtp BASI_SPEC_NMAX=1 \
  ./basi-cli -m ~/models/Qwen3.6-35B-A3B-MTP-UD-Q6_K_XL.gguf -ngl 7 --yolo \
  -p "list files using bash, then count them"

# lower-level self-tests (fire before in-process load, server owns the model):
BASI_SERVER_SELFTEST=1 BASI_SPEC=draft-mtp BASI_SPEC_NMAX=1 ./basi-cli -m <model> -ngl 7   # M1
BASI_SERVER_SELFTEST=1 BASI_SRV_GRAMMAR='root ::= "yes" | "no"' ... ./basi-cli ...         # M2
```
Server logs → `/tmp/basi_srvgen.log`. Server binds `127.0.0.1:8181`.

## REMAINING (refinement/cleanup — no unknowns left)
1. **Display fidelity** (most visible). `generate_server()`'s `srv_display_emit` streams
   chunks raw → some `<think>` / `<tool_call>` markup shows. Native has a full
   per-char state machine (model.c generate(), STATE_* + openers[]). Rebuild an
   equivalent streaming state machine that hides thinking + tool markup live. Keep
   `strip_thinking_dup()` for the returned/stored text.
2. **Sampling parity** — `generate_server` sends temp=0.4 + the grammar only. Add
   BASI's repeat_penalty(1.1)/min_p(0.05)/top_k/seed to the request so quality
   matches native (fields: `repeat_penalty`, `min_p`, `top_k`, `seed`).
3. **Embeddings** — route `embed.c` to the server `/embedding` endpoint (RAG/reuse).
4. **deepsearch** — `ds_generate()` currently uses in-process ctx/sampler; route to srvgen.
5. **Lifecycle / model-switch** — `/model` re-execs today; make it restart the server.
   Harden the interactive REPL for the vocab_only/NULL-ctx case (statusbar ctx meter,
   `kv_resync_full`, `context_used_tokens` — the `-p` path is already clean).
6. **Drop the libllama link** once the above are done → kills the ABI-coupling gotcha
   for good. BASI would still link a *small* piece for vocab_only tokenization/template
   (or move templating to the server chat endpoint and drop it entirely).

## GOTCHAS / NOTES
- Branch `spec-decode-mtp` also carries the **PARKED native-MTP attempt** (`src/spec.{c,h}`,
  commits `6259d86`,`2d00946`) — a shim that drives MTP in-process but produces garbage
  (next-n logit-buffer internals; llama-cli embeds the server, there's no simple MTP
  driver to copy). Dead end; kept for the root-cause writeup. **Split the winning
  server-backend work onto a clean `server-backend` branch off master** as step 0.
- `src/cookbook.c` has an **uncommitted** in-flight change (the `/cookbook rm` multi-select
  menu) — unrelated; don't sweep it into server-backend commits.
- The server /completion API (verified) exposes everything: raw `prompt`, `grammar`+
  `grammar_lazy`+`grammar_triggers`, `stream`, all sampling knobs, `cache_prompt`, `/embedding`.
- Memory: `project_server_backend_pivot.md` (full context), `project_mtp_specdecode_arc.md`
  (the native-MTP wall + the working `--spec-draft-n-max 1` workaround).
