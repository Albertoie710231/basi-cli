# Server-backend migration — handoff for the next session

**Direction (decided 2026-07-14):** move BASI off the native in-process `libllama`
link to a **Pi-style `llama-server` backend**. Why: advanced features (MTP
spec-decode, etc.) are only correct through the server; linking libllama costs an
ABI tax (every llama.cpp rebuild → `free(): invalid pointer`); the server is the
maintained, faster path. BASI's identity (reuse pillar — llama-independent — plus
agent loop, tools, prompts, deterministic-first) lives ABOVE the generation layer,
so the pivot sharpens "its own thing," doesn't dilute it.

## STATE: server-only rewrite well underway. Branch `spec-decode-mtp`, latest `b3bde9e`.
Items 1/2/4/5/5b DONE+verified. Item 6 (drop libllama) IN PROGRESS: (a) chat client +
(b) agent-loop-over-chat DONE. **(c) native teardown MOSTLY DONE: chat+server are the
DEFAULT/mandatory path (no env needed); spec.cpp + the 448-line in-process decode loop +
the ENTIRE /completion path (generate/generate_server/SrvDisplay) are DELETED; ALL
generation (agent loop, --no-tools, final-gen, summary, deepsearch) now flows through
generate_chat → /v1/chat/completions.** model.c 1870→~1050 lines. **common_chat GUTTED
(chat_tmpl.cpp nlohmann-only), sampler chain + all dead renders removed, deepsearch
dctx removed — chat_tmpl.cpp + deepsearch.c are now function-libllama-free.** Remaining
(c): main.c model+ctx load (swap statusbar off llama_n_ctx), model.c ggml gguf-reading,
embed.c→HTTP (net-new). Then (d) drop the link. See §6 for the precise list.

**Launch-script feature (`5791c05`,`b15517a`):** the model picker now configures the
llama-server launch (SPEC-DECODE + FLASH-ATTN sections, auto-following model MTP-ness)
and BASI writes/execs an editable `.basi/run-llama-server.sh` (gitignored, `# BASI-MODEL:`
marker → reuse-respecting-edits vs regenerate-on-model-change). Precedence: BASI_SPEC env
> picker > auto-detect "MTP" in the filename. `srvgen_{write_launch_script,script_matches,
spawn_script}` + `SrvLaunch`.

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
4. ~~**deepsearch**~~ ✅ DONE (`2d2fb4b`, `a6f5d04`). ds_generate() already routes through
   generate()→generate_server(), so it runs server-backed. Two fixes: (a) `basi_srv_suppress_grammar`
   keeps the main tool grammar out of deepsearch's rounds/synthesis (native deepsearch's
   sampler never had it); (b) deepsearch clears the main tool schemas for its own renders
   (`basi_tools_registered()` save/restore) so a native-tools model emits deepsearch's
   `<tool>` ReAct format instead of `<tool_call>`/`<function=>` — was "no action" every
   round in BOTH native+server before. Verified: full 3-round web_search/web_fetch loop
   runs over the server; native fixed too (no regression).
5. **Lifecycle / model-switch** — DONE:
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
   - ✅ **`/model` switch** (`4ee6ddf`). Re-exec via execv skips atexit(kill_srv), so the
     old llama-server was orphaned and the new process couldn't rebind :8181. Fixed:
     kill_srv() before execv. Verified (tmux): 9B→27B switch kills the old server, one
     server left (the new one) — no orphan/bind conflict. (Re-exec is kept; it's the
     clean way to release VRAM. Same-model /model still short-circuits, no re-exec.)
   - Note: `summarize_head()` (COMPACT_SUMMARY/HYBRID only, NOT the default RETRIEVE)
     still decodes into the local vocab_only ctx and would break in server mode. Only
     matters if someone sets BASI_COMPACT=summary; route it through srvgen when addressed.
6. **Drop the libllama link** — endgame. **FEASIBILITY ASSESSED 2026-07-14 (endpoint
   pre-check PASSED against llama-server 8182, Qwen3.5-9B):** every libllama use maps
   to a working server HTTP endpoint —
   - templating (`common_chat`/`apply_template`) → **`/apply-template`** — returns the
     IDENTICAL rendered prompt incl. forced-open `<think>`; with a `tools` array it
     injects the same `# Tools … <tools>{…}</tools>` block BASI builds.
   - tokenization (ctx meter, `srv_update_ctx_used`) → **`/tokenize`** / `/detokenize`.
   - tool grammar + `common_chat_parse` → **`/v1/chat/completions` with `tools`**: the
     server derives the grammar, applies it, AND returns STRUCTURED `tool_calls`
     (`{"name":"bash","arguments":"{\"command\":\"ls\"}"}`) — replaces
     basi_tool_grammar_json + grammar splicing + the PEG parser in one call.
   - n_ctx / chat_template / bos/eos → **`/props`**.
   - embeddings → `/embedding` (embed.c is already self-contained in-process).

   **User GREENLIT the full server-only rewrite (2026-07-14).** ~half the current
   libllama surface is the NATIVE generate() path — `llama_decode`/batch/samplers/
   `llama_get_memory` in model.c + deepsearch's own `dctx` — which gets DELETED
   (native in-process mode goes away). Phasing + PROGRESS:

   - ✅ **(a) chat client** (`8d1c3c7`). `src/srvchat.{h,cpp}` — `srvchat_complete()`
     POSTs `/v1/chat/completions` (stream=true, tools), parses SSE deltas (content,
     server-separated `reasoning_content`, incrementally-streamed `tool_calls`) +
     final `usage.prompt_tokens` + `timings` tok/s. nlohmann + curl only, NO libllama.
     Self-test `BASI_SRV_CHAT_SELFTEST=1`. Verified: reasoning streams, structured
     `bash({"command":"ls -la"})` assembled, prompt_tokens/tps captured.
   - ✅ **(b) pt1 serialization** (`a8fcc5b`). `basi_messages_to_json()` /
     `basi_tools_to_json()` (chat_tmpl.cpp, nlohmann-only): BASI messages+tools →
     OpenAI JSON. Roles map: tool_call→assistant+tool_calls[] (synthetic `call_N`
     id); tool_result→role:"tool"+`tool_call_id`. Round-trip `BASI_SRV_CHAT_MSGTEST=1`
     verified: model ANSWERS FROM an injected tool_result (id-pairing templates right).
   - ✅ **(b) pt2 WIRING DONE** (`b382498`). `generate_chat(messages, msg_count,
     &tc_out, &n_out)` in model.c: serialize → `srvchat_complete` (reasoning→thinking
     box, content→md answer; no `<think>` parsing) → answer text + STRUCTURED
     tool_calls (→BasiToolCall) + `prompt_tokens`. run_turn has a BASI_SERVER_CHAT
     branch: calls generate_chat, ctx-used from usage, feeds structured calls to the
     EXISTING native dispatch, skips apply_template grammar-reset + PEG parse.
     **Qwen3.x quirk handled**: a brief answer sometimes lands entirely in
     `reasoning_content` (content empty) — when no content + no tool_call, promote
     reasoning to the answer & reveal it. Verified: "count .c files in src/" → round1
     `bash find…|wc -l` executes, round2 "There are 21 .c files"; ctx honest 2.5k→3.4k.
     Diagnostics kept: BASI_DEBUG_CHAT. **REMAINING in (b):** (i) make BASI_SERVER_CHAT
     the DEFAULT once validated interactively (markdown path + multi-turn REPL);
     (ii) deepsearch — ds_generate() still hits /completion via generate_server; it
     clears g_tools so generate_chat would send no tools and the model would emit the
     `<tool>` ReAct format in `content` (deepsearch parses that) — route ds to
     generate_chat and confirm; (iii) legacy/non-native models in chat mode (today
     the branch assumes tool-capable — fine for the target models).
   - 🔄 **(c) delete native path — IN PROGRESS.** Done so far:
       - ✅ step1 (`918c7af`): chat mode DEFAULT in server mode (BASI_SERVER_COMPLETION=1 opts back to /completion).
       - ✅ step2 (`918c7af`… `use_server=true`): server mode DEFAULT/mandatory (was BASI_NATIVE hatch, now removed) — every launch spawns llama-server.
       - ✅ step3 (`…`): deleted `src/spec.{c,h}` (parked native-MTP dead-end) + its self-test hook.
       - ✅ step4 (`c22acf5`): DELETED the in-process decode loop in generate() (~448 lines: tokenize→llama_decode→sampler loop, inline ThinkingState, batch/KV, per-token UTF-8/Ctrl+T). generate() is now a thin `return generate_server(prompt)` wrapper. Display helpers (draw/clear_thinking_box, utf8_seq_len, strip_thinking_dup) stay (used by SrvDisplay + ChatDisplay). Verified: chat tool-call + --no-tools both work.
     MORE done (2026-07-14):
       - ✅ step5 (`…`): deepsearch → generate_chat (no more apply_template/generate()/dctx-gen).
       - ✅ step6 (`771d882`): moved --no-tools -p, tool-exhausted final-gen, and
         summarize_head onto generate_chat; removed run_turn's BASI_SERVER_COMPLETION
         branch + the legacy <tool>-extraction dispatch (server returns structured calls).
       - ✅ step7 (`0f58144`): DELETED the whole /completion path — generate(),
         generate_server(), the SrvDisplay STATE_* machine, strip_thinking_dup,
         ThinkingState, utf8_seq_len. model.c 1422→1108. srvgen_complete kept (self-test,
         libllama-free). generate_server was the last generation-path user of common_chat grammar.
     DONE (steps 8–11, `b382498`…`b3bde9e`): removed ALL run_turn dead renders +
     delta/kv machinery + srv_update_ctx_used; removed the in-process sampler chain +
     g_tool_grammar; **gutted common_chat** — chat_tmpl.cpp rewritten to keep ONLY the
     nlohmann serializers (2 llama symbols, TYPE-only), deleted apply_template/
     apply_chatml + basi_render_chat/tool_grammar_*/parse_tool_calls/thinking_tags/
     tools_active; native_tools hardcoded 1; removed deepsearch's dead dctx + samplers
     (now function-libllama-free). Verified at each step (chat tool loops + multi-turn
     recall + deepsearch).
     REMAINING (c) — the last function-level libllama users:
       - **main.c** (~11 fns): the vocab_only MODEL LOAD (llama_model_load_from_file/
         default_params/get_vocab/free) + CTX (llama_init_from_model/free/n_ctx/
         get_memory/memory_seq_pos_max/memory_clear/context_default_params). To drop:
         make `context_used_tokens` return basi_srv_ctx_used unconditionally; store the
         server ctx size in a global and swap `llama_n_ctx(ctx)`→it (statusbar +
         format_context_meter); make kv_resync_full just reset prev_len (drop
         llama_memory_clear); then delete the model+ctx load entirely (nothing uses
         model/vocab/ctx after — basi_srv_model was only for the deleted grammar).
       - **model.c** (3 fns): model_init (ggml_backend_load_all + llama_log_set) + the
         picker's read_gguf_arch (ggml_dtype_bytes_per_elem, VRAM estimate). The picker
         reads GGUF metadata off disk directly — reimplement the dtype-size lookup as a
         static table to drop ggml, and model_init becomes unnecessary once no model loads.
       - **embed.c** (15 fns, NET-NEW WORK): route to a SPAWNED embedder llama-server
         (`--embedding`) + `/embedding` HTTP. The one piece that's new code, not deletion.
       - `llama_chat_message` POD {role,content} → a local struct so headers drop llama.h.
     Surface now: main.c 19, embed.c 23, model.c 6, deepsearch.c 4 (TYPES only),
     chat_tmpl.cpp 2 (TYPE only); srvgen.c + srvchat.cpp ZERO. Then (d) drop the link.
   - ⬜ **(d) drop link** — remove `-lllama -lllama-common -lggml* -lvulkan` and the
     `-I$(LLAMA_DIR)/...` includes (keep a vendored nlohmann). Verify `ldd basi-cli`
     shows no libllama/libggml. ABI-coupling gotcha GONE.

   Endpoint evidence + serialization + client all PROVEN; remaining is plumbing.

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
