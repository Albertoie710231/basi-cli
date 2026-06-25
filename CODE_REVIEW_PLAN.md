# BASI-CLI — Code Review & Remediation Plan

> **For a fresh session:** This is an actionable plan produced by a full code review on
> **2026-06-23**. It is self-contained — you do not need the original conversation.
> Work the tasks in the **Order of work** table. Each task gives the problem, exact
> location, the fix (with code), an authoritative source, and how to verify.
>
> **Line numbers are as-of the review date and will drift as you edit.** Each task
> includes an anchor code snippet — `grep` for that snippet to relocate the site.
> Do the tasks **top-to-bottom**; later line numbers shift after earlier edits.

---

## Context

BASI-CLI is a local-LLM agentic CLI written in C on top of llama.cpp (~10.4k lines,
16 modules in `src/`). The project's design motto is **deterministic-first**: anything
implementable in C is, and there is a shared utility library `src/util.c` / `src/util.h`
(`StringBuf`, `tokenize_command`, `run_command`, `mkdir_p`, `url_encode`, `json_escape_into`,
and the `jx_*` JSON-path parser).

The review checked three things the maintainer asked for: **(1) no repeated code,
(2) correct C practices, (3) reuse our own libraries.** Verdict: reuse discipline is
**good** — `StringBuf`, `tokenize_command`, `mkdir_p`, `url_encode`, and the `jx_*` parser
have **zero competing reimplementations**. The issues below are the real remaining gaps,
each verified against the source (not taken from a tool's say-so).

Every best-practice claim cites the **SEI CERT C Coding Standard** (the reference standard
for C) or the DRY principle. Full links in **Sources** at the bottom.

---

## How to build & verify (do this once before and after each task)

```sh
# Build defaults point at the author's llama.cpp paths; override for this machine:
make LLAMA_DIR=/path/to/llama.cpp LLAMA_BUILD=/path/to/llama.cpp/build
# Output binary: ./basi-cli
```

- Compiler flags are already strict: `-Wall -Wextra -O2` (see `Makefile`). **A clean
  build with no new warnings is the baseline acceptance check for every task.**
- There is **no unit-test suite**. Verify behaviorally: build, then exercise the affected
  tool. Suggested smoke test for the tool loop:
  `./basi-cli -m <model.gguf> -p "read src/util.c and tell me how sb_ensure grows"`.
- Keep each task a separate commit so regressions are bisectable.

---

## ⚠️ False leads — do NOT chase these (already traced to source and refuted)

A prior adversarial pass raised two "critical" items that **do not hold up**. Recorded here
so nobody re-investigates them:

1. **"`grep` pattern → `rm -rf /` remote code execution" — FALSE.** `tooldefs.c` builds
   `grep -n "PATTERN" FILE`; it is then re-tokenized by `tokenize_command` and **re-quoted
   with proper `'\"'\"'` single-quote escaping at `main.c:788-804` before `popen`**. Shell
   metacharacters become *literal arguments*, not shell syntax. The real defect on this path
   is **tokenizer corruption** (Task 3), not RCE.
2. **"`web_search` query injection via `'`" — FALSE.** The query passes through `url_encode`
   (`util.c:175`), whose allow-set is `[A-Za-z0-9-_.~]` only — a `'` becomes `%27`. It cannot
   reach the shell unescaped.

The SSRF guard in `web.c` (`url_is_fetchable`: blocks loopback, RFC1918, `169.254/16`,
`100.64/10`, `.local/.internal`, `--max-redirs 0`) and the `'\"'\"'` escaping in
`bash`/`verify`/`docs_search` are **correct** — leave them as-is.

---

## Order of work

| # | Task | File(s) | Effort | Risk | Payoff |
|---|------|---------|--------|------|--------|
| 1 | Guard `realloc` in shared lib | `util.c` | tiny | low | hardens every module |
| 2 | Reuse robust file-slurp helper | `util.{c,h}`, `patch.c`, `lsp.c`, `kb.c` | small | low | removes dup + fixes latent overflow |
| 3 | Fix native-tool → shell round-trip | `main.c`, `tooldefs.c` | medium | med | stops silent tool corruption |
| 4 | Single JSON-string escaper | `lsp.c`, `main.c` → `util.c` | small | low | one correct copy |
| 5 | `run_command_status` for verify | `util.{c,h}`, `verify.c` | small | low | removes dup |
| 6 | `mkdir_p` truncation + `strncpy` term | `util.c`, `main.c` | tiny | low | robustness |
| 7 | (Optional) unchecked `malloc`/`strdup` sweep | many | medium | low | OOM-safety |

Tasks 1, 2, 4, 6 are low-risk and independent — safe to land first. Task 3 is the only
behavioral change worth careful testing.

---

## Task 1 — Guard `realloc` in `StringBuf` and the tokenizer  *(highest leverage)*

**Problem:** Two `p = realloc(p, n)` sites in the shared library are unchecked. On failure
`realloc` returns `NULL`, the original block **leaks**, and the next line dereferences `NULL`.
Because these are in `StringBuf` and `tokenize_command`, **every module** is exposed.

**Locations & anchors:**
- `src/util.c` ~line 29, in `sb_ensure`:
  ```c
  sb->data = realloc(sb->data, newcap);
  ```
- `src/util.c` ~line 87, in `tokenize_command`:
  ```c
  al.args = realloc(al.args, cap * sizeof(char *));
  ```

**Fix:** assign to a temporary first (so the original survives a failed `realloc`), then
handle NULL. For this codebase's style, aborting is acceptable since these are
infrastructure allocations:
```c
char *tmp = realloc(sb->data, newcap);
if (!tmp) { perror("realloc"); abort(); }   /* or: return + propagate */
sb->data = tmp;
```
Apply the same pattern to the `tokenize_command` site (`char **tmp = realloc(...)`).

**Source:** [CERT ERR33-C — detect and handle standard library errors][err33];
the `p = realloc(p,…)` leak is [PVS-Studio V701][v701] ("save the result into a different
variable; if NULL the original is still valid").

**Verify:** clean rebuild; run the read smoke test — `StringBuf` is on every hot path, so any
breakage shows immediately.

---

## Task 2 — Reuse one robust "read whole file" helper

**Problem:** `kb.c:195` `kb_read_file()` is the **robust canonical** slurp — it checks
`fseek`, guards `ftell() < 0`, and NULL-checks `malloc`. It is exported in `kb.h:55`. But two
modules reimplement the slurp **without those guards**, so the copies are *buggier than the
original*:
- `src/patch.c` ~line 431 (in `execute_edit`), anchor:
  ```c
  fseek(f, 0, SEEK_END); long fsize = ftell(f); fseek(f, 0, SEEK_SET);
  char *orig = malloc(fsize + 1);
  size_t nread = fread(orig, 1, fsize, f);
  ```
  If `ftell` returns `-1`, `fsize+1 == 0` → `malloc(0)`, then `fread(orig,1,(size_t)-1,f)` —
  latent buffer overflow that `kb_read_file` already prevents.
- `src/lsp.c` ~line 555 (in `execute_code_context`): same pattern, same missing guards.

**Fix (preferred — relocate the helper to the shared lib so layering is clean):**
1. Add to `src/util.h`:
   ```c
   char *read_file_all(const char *path, size_t *out_len);  /* malloc'd; NULL on error; caller frees */
   ```
2. Add to `src/util.c` the body of `kb_read_file` verbatim (it is already correct):
   ```c
   char *read_file_all(const char *path, size_t *out_len) {
       FILE *f = fopen(path, "r");
       if (!f) return NULL;
       if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
       long sz = ftell(f);
       if (sz < 0) { fclose(f); return NULL; }
       fseek(f, 0, SEEK_SET);
       char *buf = malloc((size_t)sz + 1);
       if (!buf) { fclose(f); return NULL; }
       size_t n = fread(buf, 1, (size_t)sz, f);
       buf[n] = '\0';
       fclose(f);
       if (out_len) *out_len = n;
       return buf;
   }
   ```
3. In `kb.c`, make `kb_read_file` a thin wrapper (`return read_file_all(path, out_len);`) to
   avoid churning its other call sites, **or** replace its body and keep the name.
4. In `patch.c:431`, replace the slurp block with:
   ```c
   size_t nread = 0;
   char *orig = read_file_all(path, &nread);
   if (!orig) { /* reuse the existing cannot-open error path */ }
   sb_append(&cur, orig, nread);
   free(orig);
   ```
   (Note: the existing `fopen` a few lines above becomes redundant — `read_file_all` opens
   the file itself; remove the now-dead `FILE *f`.)
5. In `lsp.c:555`, replace the slurp with `read_file_all`, preserving the existing
   `fsize > 8 MiB` guard by checking `out_len` after the call.

**Source:** [DRY principle][dry] — a single slurp routine means a single place to be correct.
The latent `ftell` overflow is exactly what the canonical copy already guards.

**Verify:** rebuild; `code_context` (`./basi-cli ... -p "code_context for sb_ensure in src/util.c"`)
exercises `lsp.c`; an `edit` tool call exercises `patch.c`. Confirm both still read files.

---

## Task 3 — Fix the native-tool → shell-string round-trip (silent corruption)

**Problem:** `tooldefs.c` `basi_build_command()` takes **already-parsed JSON args** and
**re-serializes** them into a flat command string, which `execute_tool` (`main.c:677`) then
**re-tokenizes**. `append_arg(..., quote=true)` wraps a value in `"..."` but does not (and,
through `tokenize_command`, *cannot*) preserve an embedded `"`:
- `src/tooldefs.c` ~line 71, anchor:
  ```c
  static void append_arg(StringBuf *sb, const char *json, const char *key, bool quote) {
      char *v = jx_get_string(json, key);
      ...
      if (quote) sb_append_char(sb, '"');
      sb_append_str(sb, v);          // raw — an embedded " ends the token early
      if (quote) sb_append_char(sb, '"');
  ```
**Concrete failure:** a `grep` pattern `foo"bar` becomes `grep -n "foo"bar FILE`;
`tokenize_command` yields tokens `foo` and `bar` → grep silently searches the wrong thing,
**no error**. Same for any `web_search` query / `web_fetch` URL / `readfile` regex containing
a `"`. This is **not** an injection (see False Leads) — it is **correctness loss**.

Why a backslash-escape band-aid won't work: `tokenize_command` has **no escape mechanism**
inside quotes, and the **legacy `<tool>` prose path** relies on that (e.g. a regex
`"\bword\b"` must keep its backslashes). Teaching the tokenizer to unescape would regress the
prose path. The clean fix is to **stop round-tripping** for native calls.

**Fix (preferred — structured dispatch, removes the bug class):**
Add a native entry point that dispatches parsed JSON args **directly** to the existing
handlers, bypassing `basi_build_command` + `tokenize_command` entirely.
1. In `main.c`, add `static char *execute_tool_native(const char *name, const char *args_json)`:
   - For `web_search`/`web_fetch`/`readfile`: pull args with `jx_get_string` and call the
     existing handlers directly — they already take separate string args:
     `execute_web_search(query, recency)` (see `main.c:748`), `execute_web_fetch(url)`
     (`main.c:761`), `execute_readfile(path, regex)` (`main.c:775`).
   - For `read`: call the same size-checked file read used at `main.c:703-736`.
   - For `head`/`tail`/`grep`/`wc`: build the shell command from the parsed args **using the
     existing `'...'`-escaping loop at `main.c:788-804`** (which is already injection-safe),
     never via `tokenize_command`.
   - For `edit`/`scaffold`/`code_context`/`plan_*`/`assumptions`/`spike_write`/`docs_*`: these
     take a string body/args; keep using `basi_build_command` + `execute_tool` for them (they
     don't suffer the quote problem) **or** call their handlers directly.
2. At the native call site (`main.c` ~line 1670, anchor
   `cmd_str = basi_build_command(ncalls[0].name, ncalls[0].arguments);`), replace the
   `basi_build_command` → `execute_tool` pair with `execute_tool_native(ncalls[0].name,
   ncalls[0].arguments)`.
3. Keep `basi_build_command` for any tool you did not move (and for the legacy prose path
   that still needs it).

**Fix (interim, if you defer the refactor):** in `append_arg`, document that values
containing `"` are unsupported on the native path and at minimum reject them with a clear
error rather than silently corrupting — but the structured dispatch above is the real fix.

**Source:** [CERT ENV33-C][env33] — keep command arguments structured; do not reserialize
through a shell-string layer. [DRY][dry] — the handlers already exist; call them once.

**Verify:** `./basi-cli ... -p 'search src/main.c for the text he said "hi"'` (a pattern with
a `"`). Before: corrupted grep. After: the literal `"` reaches grep intact. Also re-run a
plain `web_search` and `edit` to confirm no regression on the other tools.

---

## Task 4 — Collapse three JSON-string escapers into one

**Problem:** three separate implementations of JSON string escaping exist; the `lsp.c` one is
**already subtly wrong** (missing `\b`/`\f`), which is the textbook argument for DRY:
- `src/util.c:219` `json_escape_into(StringBuf*, s)` — the canonical one (adds quotes, full
  escape set). Currently only `session.c` uses it.
- `src/lsp.c:303` `lsp_json_escape(StringBuf*, s)` — near-dup, **no quotes, missing `\b`/`\f`**.
  Called at `lsp.c:344,346,357,359`.
- `src/main.c:1303-1310` — hand-rolled `fputs`/`fprintf` escaping in the `/transcript` handler.

**Fix:**
- `lsp.c`: delete `lsp_json_escape`; its callers add their own surrounding quotes, so either
  (a) call `json_escape_into` (which **adds** quotes — drop the manual quote chars around the
  call), or (b) add a `json_escape_body_into` variant in `util.c` that omits the quotes and
  have both modules use it. Prefer (a) unless a caller truly needs the no-quote form.
- `main.c` `/transcript`: escape into a `StringBuf` with `json_escape_into`, then
  `fwrite(sb.data, 1, sb.len, out)` — removes the third copy.

**Source:** [DRY][dry]. The bug-divergence (`lsp.c` missing escapes) is the concrete cost of
the duplication.

**Verify:** rebuild; run `/transcript` in an interactive session and confirm the JSONL is
valid (`python3 -c "import json,sys;[json.loads(l) for l in open('<file>')]"`). Exercise
`code_context` to confirm the LSP request JSON is still accepted by clangd.

---

## Task 5 — Extract `run_command_status` to remove the verify.c popen dup

**Problem:** `verify.c:144-167` reimplements the `popen` → `fread`-into-`StringBuf` → `pclose`
loop that `run_command` (`util.c:109`) already has. The difference is **legitimate**:
`verify.c` needs the exit status (`WIFEXITED`/`WEXITSTATUS`, `verify.c:169`) which
`run_command` discards.

**Fix:** add to `util.c`/`util.h`:
```c
char *run_command_status(const char *cmd, size_t max_output, int *exit_code);
```
Implement the capped read loop once; set `*exit_code` from `pclose`/`WEXITSTATUS`. Reimplement
`run_command` as `run_command_status(cmd, max, NULL)`, and have `verify.c` call the new
function instead of its private loop.

**Source:** [DRY][dry].

**Verify:** rebuild; run a `/plan` flow that reaches `plan_verify`, or any path that calls
`run_clause`, and confirm exit codes are still reported correctly (a deliberately failing
clause should report non-zero).

---

## Task 6 — Truncation & null-termination hardening

**Problem A — `mkdir_p` silent truncation.** `src/util.c:135`:
```c
char tmp[1024];
snprintf(tmp, sizeof(tmp), "%s", path);   // return value ignored
```
A path > 1023 bytes is silently truncated, then `mkdir`s the wrong path.
**Fix:** check the return: `int k = snprintf(...); if (k < 0 || (size_t)k >= sizeof(tmp)) return -1;`

**Problem B — `strncpy` may not null-terminate.** `src/main.c:937`:
```c
strncpy(picked_model, cfg.model_path, sizeof(picked_model) - 1);
```
Relies on the buffer being pre-zeroed.
**Fix:** add the explicit terminator immediately after:
`picked_model[sizeof(picked_model) - 1] = '\0';`

**Source:** [CERT STR31-C — sufficient space for data and the null terminator][str31]
(covers both truncation detection and `strncpy`'s no-null-terminate behavior).

**Verify:** rebuild (these are defensive; no behavior change on normal-length inputs).

---

## Task 7 (Optional) — Unchecked `malloc`/`strdup` sweep

**Problem:** allocation results are routinely used without a NULL check. Most fixed-size
`malloc(256/384/512)` error-message buffers are near-harmless on a desktop, but prioritize
the **size-from-input** allocations where NULL is realistic even under Linux overcommit:
- `main.c:730` `malloc(st.st_size + 1)` (file read), `deepsearch.c:433`
  `malloc(ol + strlen(synth) + 1)`, and the slurp sites fixed in Task 2.
- `strdup` results stored into arrays without a check: `web.c:681`, `kb.c:594`,
  `embed.c:466-468`.

**Fix:** check each and fail gracefully (`return strdup("Error: out of memory");` for the
tool-result paths, or propagate). Do the **size-from-input** ones first; the fixed-size
message buffers are low value.

**Source:** [CERT ERR33-C][err33] / [MEM32-C][v701].

**Verify:** rebuild; this is hardening — no normal-path behavior change.

---

## Definition of done

- [ ] All tasks 1–6 landed (7 optional), each its own commit.
- [ ] `make` is clean — **no new `-Wall -Wextra` warnings**.
- [ ] Smoke tests in each task pass.
- [ ] No duplication regressions: `grep -rn "SEEK_END" src/*.c` shows the slurp only in the
      shared helper (+ the distinct `main.c:1151` system-prompt append, which is intentionally
      a different bounded-buffer case — leave it).

## What was already good (do not "fix")

- `jx_*` JSON parser, `StringBuf`, `mkdir_p`, `tokenize_command`, `url_encode` — consolidated,
  single-source, no duplication.
- `web.c` SSRF guard and the `'\"'\"'` shell-escaping in `bash`/`verify`/`docs_search`.

---

## Sources

- [SEI CERT C — ERR33-C: Detect and handle standard library errors][err33]
- [PVS-Studio V701 — realloc leak (CERT MEM32-C pattern)][v701]
- [SEI CERT C — ENV33-C: Do not call system()][env33]
- [SEI CERT C — STR31-C: Guarantee sufficient space for data and the null terminator][str31]
- [DRY — Don't Repeat Yourself][dry]

[err33]: https://wiki.sei.cmu.edu/confluence/display/c/ERR33-C.+Detect+and+handle+standard+library+errors
[v701]: https://pvs-studio.com/en/docs/warnings/v701/
[env33]: https://wiki.sei.cmu.edu/confluence/display/c/ENV33-C.+Do+not+call+system
[str31]: https://wiki.sei.cmu.edu/confluence/display/c/STR31-C.+Guarantee+that+storage+for+strings+has+sufficient+space+for+character+data+and+the+null+terminator
[dry]: https://en.wikipedia.org/wiki/Don%27t_repeat_yourself
