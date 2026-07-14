# Server-backend migration — handoff for the next session

**Direction (decided 2026-07-14):** move BASI off the native in-process `libllama`
link to a **Pi-style `llama-server` backend**. Why: advanced features (MTP
spec-decode, etc.) are only correct through the server; linking libllama costs an
ABI tax (every llama.cpp rebuild → `free(): invalid pointer`); the server is the
maintained, faster path. BASI's identity (reuse pillar — llama-independent — plus
agent loop, tools, prompts, deterministic-first) lives ABOVE the generation layer,
so the pivot sharpens "its own thing," doesn't dilute it.

## STATE: the core works. Branch `spec-decode-mtp`, latest `eaab47b`.

**2026-07-14 session added (committed `eaab47b`):**
- **Item 1 DONE — display fidelity.** `SrvDisplay` streaming state machine in
  model.c mirrors native generate()'s STATE_* filter over the SSE chunks: `<think>`
  collapses to the spinner box (or dim `[thinking]` under Ctrl+T), native
  `<tool_call>` markup is suppressed, answer text is markdown-rendered,
  forced-open `<think>` handled. Display only; returned text keeps markup for the
  parser. Verified: zero leaked think/tool markup, tool executed and fed back.
- **Item 2 DONE — sampling parity.** `SrvSampling` (srvgen.h) threads BASI's native
  knobs (temp, repeat_penalty 1.1 + repeat_last_n 256, min_p 0.05, top_k, top_p,
  seed) from main.c into the /completion request; temp==0 pins top_k:1. Verified:
  server accepts them, coherent output.
- **Item 3 REASSESSED — lower priority than the doc implied.** `embed.c` loads its
  OWN dedicated retrieval embedder (jina/bge) in-process, independent of the chat
  model, so RAG/reuse already work in server mode. Routing to the chat server's
  `/embedding` would give chat-model embeddings (worse for retrieval), so item 3 is
  only meaningful as a prerequisite for item 6 (drop libllama) — e.g. spawn a
  second small server for the embedder. Not a correctness fix.

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
1. ~~**Display fidelity**~~ ✅ DONE `eaab47b` (see above).
2. ~~**Sampling parity**~~ ✅ DONE `eaab47b` (see above).
3. **Embeddings** — reassessed (see above): only meaningful for item 6. Deprioritized;
   embed.c already works in server mode via its own in-process embedder.
4. **deepsearch** — `ds_generate()` currently uses in-process ctx/sampler; route to srvgen.
5. **Lifecycle / model-switch** — partially done:
   - ✅ **Interactive REPL context accounting** (`<next commit>`). ctx is a valid but
     empty vocab_only context in server mode (no crash), but the local KV never fills
     so the meter read `0/33k 0%`, compaction NEVER fired, and the model's
     "tokens remaining" hint was always wrong. Fixed: `srv_update_ctx_used()` tokenizes
     each rendered prompt and `context_used_tokens()` returns it when `basi_srv_port>0`
     — meter, reclaim trigger, and hint now honest. ALSO fixed a latent bug: the
     KV-delta feed (`formatted_buf+prev_len`) would send only the tail to the server
     (which prefix-caches the FULL prompt itself), losing history for non-thinking
     models — server mode now always feeds the full prompt. Verified: meter grows
     (2.5k→2.6k), turn-2 recalled "42" from turn-1, compaction triggers at used=2529.
   - ⬜ **`/model` switch** still re-execs (spawns a fresh process → fresh server).
     Could instead restart just the server in-process. Low priority — re-exec works.
   - Note: `summarize_head()` (COMPACT_SUMMARY/HYBRID only, NOT the default RETRIEVE)
     still decodes into the local vocab_only ctx and would break in server mode. Only
     matters if someone sets BASI_COMPACT=summary; route it through srvgen when addressed.
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
