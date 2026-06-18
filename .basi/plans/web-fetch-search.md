# Plan: rewrite web fetch + search (Odysseus-modeled, Firefox tier-3)

## Status (2026-06-16) — web_search now auto-fetches top 3 (comprehensive)
Root cause found via the version-table test: `web_search` returned *snippets only*,
and for many fact lookups the answer isn't in the snippet (e.g. Go's `1.26.4` is
only on the page; the snippet jumble even contained the stale `1.25` the 4B model
then echoed). Qwen3-4B-2507 is ~11 months stale, so its correct post-cutoff
answers (torch 2.12, Python 3.14.6, Rust 1.96.0, Debian 13) PROVE the web tool is
used — the one failure (Go) was the case where the live number was absent from the
snippet and the model fell back to its prior.
Fix (Odysseus `comprehensive_web_search` model): `web_search` now fork-parallel
fetches the **top 3** ranked URLs (`WEB_SEARCH_FETCH_COUNT`, ~3000 chars each via
`WEB_SEARCH_FETCH_CHARS`), inlines them under `==== FETCHED PAGE CONTENT ====`, and
the header tells the model to prefer page content over snippets/priors. Verified:
`1.26.4` now lands in context; 3 fetches ≈ 1.5 s wall-clock (parallel). Shared
`web_fetch_extract` core now backs both tools.
Follow-up (same day): auto-fetch first made Go *worse* — it pulled
`dev.golang.org/release` (Go's dev dashboard, full of the unreleased 1.27: "switch
to Go 1.27 as default inside Google"), so the model reported 1.27 (it appeared 4×
vs 1.26.4 2×). Fix: `is_preview_host()` sinks developer/preview hosts (`dev.`,
`tip.`, `beta.`, `next.`, `staging.`, `nightly`, `canary`) by −100 in the ranker —
they stay in the list but drop out of the fetched top-3. After: Go content = 1.26.4
×5, 1.27 ×0; Node/Python unaffected. Also added prompt rule #6: cite ONLY URLs that
appear in results (the 4B was fabricating citation links, e.g. a "Go 1.27 notes"
link pointing at the 1.21 docs). NEXT lever: strip w3m nav boilerplate to
main-content (Odysseus content-area heuristic) so the cap holds useful text.

## Status (2026-06-16) — Phase 1 DONE + verified live + legacy retired
- Verified `web_search` end-to-end against a **real local SearXNG** (native venv
  at `~/Documentos/searxng`, JSON enabled, port 8888): real ranked results, time
  filter, and the full search→fetch loop.
- **Legacy retired:** removed `execute_webfetch`, `fetch_and_extract`,
  `url_split_base`, the `MINI_BROWSER` dep, and `WEBFETCH_*` constants from
  web.c; updated main.c (allow-list, dispatch, prompt catalog, help, errors),
  plan.c (phase gating → `is_web` covers web_search/web_fetch), web.h. Binary
  shrank ~12 KB. `readfile` kept (now uses `EXTRACT_MIN_LINE_LEN`).
- **Seamless SearXNG:** default instance is now `localhost:8888`; `web_search`
  works with zero config. `web_ensure_searxng()` is called at basi-cli startup
  (before model load, warms up concurrently) — auto-launches the local SearXNG
  if it's not already up; non-blocking; opt out with `BASI_NO_SEARXNG=1`,
  relocate with `SEARXNG_HOME`.

## Status (2026-06-16) — Phase 1 DONE
Built + verified in `src/web.c`:
- `web_fetch(url)` — curl-first, manual per-hop redirect validation, SSRF guard
  (host_is_public: v4/v6 private/loopback/link-local/CGNAT/multicast + internal
  names), shell-injection-proof (rejects `'`/control chars), PDF via pdftotext,
  HTML via w3m + og/title thin fallback. Verified live (example.com, Wikipedia);
  guard blocks 169.254.169.254/localhost/192.168/::1/metadata.google.internal.
- `web_search(query,[time])` — **thin client to a SearXNG JSON API** (see pivot
  below), C-side ranking ported from Odysseus, verified end-to-end vs a mock
  SearXNG; degrades gracefully when no instance is reachable.
- `jx_*` JSON parser **lifted from lsp.c → util.c** (shared by web.c + LSP).
- Wired into main.c (allow-list, dispatch, prompt catalog); rule #1 amended to
  allow web_search→web_fetch chaining. Legacy `webfetch` kept until verified.
- Builds clean under `-Wall -Wextra`.

**Search-backend pivot (supersedes "in-process metasearch" below):** user
confirmed Odysseus uses self-hosted **SearXNG**, and chose to make BASI a *thin
client* to a configurable instance (`SEARXNG_INSTANCE`, default
`http://localhost:8080`) — NOT the Mojeek/Bing in-process fan-out from
[[project_web_search_redesign]]. BASI stays a single binary; the user runs
SearXNG (JSON format enabled). DDG-Lite is dead from this box (anomaly/block
page), confirming the move.

Remaining: Phase 2 (tune SearXNG params/engines, discourse/.json rewrites,
JSON-LD articleBody), Phase 3 (Firefox/Marionette tier-3, spawn-per-fetch,
retire mini_browser), Phase 4 (cache TTL/LRU).

---


## Goal
Replace the brittle single-tool `webfetch` in `src/web.c` with **Claude/Codex-style
web tooling**: two clean agent primitives the agent loop drives to "navigate and
investigate" the web. Odysseus is the **open-source blueprint** (it implements the
same capability as Claude/Codex `WebSearch`/`WebFetch`, which are closed) — port its
*structure*, keep BASI's own engine list and constraints.

Key fact that frames everything: **Claude/Codex web tools are NOT browsers.** Both are
server-side HTTP + HTML→markdown; JavaScript rendering is explicitly unsupported. The
"browsing" intelligence lives in the agent loop (search → read snippets → fetch → follow
link → synthesize), not in a rendering engine. Odysseus matches this — its only browser
is a *separate, optional* Playwright-MCP, not part of the fetch path.

→ To reach parity we need **no browser**. Firefox/Marionette is an **upgrade beyond**
Claude/Codex: a gated tier-3 JS renderer, mission-clean (Gecko/SpiderMonkey, not
Blink/V8 — see [[feedback_no_proprietary_no_google]]).

## Tool interface (split the coupled tool — mirror Odysseus)
Today: `execute_webfetch(search_query, grep_query)` fuses search+fetch+grep
(`web.c:284`). Replace with two tools (Odysseus `tool_schemas.py:55,70`):

- `web_search(query, time_filter?)` → ranked results `[{title, url, snippet, age}]`.
  Snippets only — NO content fetch. `time_filter ∈ {day,week,month,year}`.
- `web_fetch(url)` → clean extracted text of ONE page. Bare domain ok; http→https.

The old "search 5 + fetch all 5 + grep" becomes the agent's job: it searches, reads
snippets, then fetches the URLs it chooses. This is the Claude/Codex shape and the
"navigate" behavior the user asked for.

**Tradeoff to confirm:** the split costs more agent-loop turns than the one-shot combo,
but is far more navigable and matches the target UX. (Recommend: take the split.)

## Component map — Odysseus (Python) → BASI (`web.c`, C)
| Odysseus | What to port | BASI target |
|---|---|---|
| `providers.py` provider abstraction + **fallback chaining** | structure only; engines = BASI's own | `web_search` fan-out |
| `core.py` `searxng_search_results` (cache+retry+chain) | retry/chain/cache orchestration | search orchestrator |
| `content.py` `_public_http_url` + manual per-hop redirect re-validation (`:69-104`) | **SSRF guard — port verbatim in spirit** (BASI has none today) | shared fetch guard |
| `content.py` `fetch_webpage_content` tiering | curl-first layered fetch | `fetch_and_extract` rewrite |
| `ranking.py` `rank_search_results` (`:92`) | scoring formula, pure/deterministic — ideal for C | `rank_results()` |
| `cache.py` SHA-256 key + TTL + LRU (`:29-57`) | file cache (2h content; news 30m / ref 24h search) | `web_cache.c` |
| `query.py` `_cache_duration_for_query` (`:139`) | news-vs-reference TTL heuristic | cache TTL |
| `query.py` `enhance_query` OR/AND/`site:` boosts | **port selectively** — operators are SearXNG/Brave syntax; verify Mojeek/Bing/DDG support before using | (maybe skip v1) |
| `content.py` key_points/tldr/quotes/statistics | **skip** — model reads raw content (Claude/Codex return raw); deterministic-first doesn't mean re-deriving what the model does | — |

### Search engines: keep BASI's in-process metasearch, NOT SearXNG
Odysseus runs SearXNG as a Docker service. BASI's redesign ([[project_web_search_redesign]])
rejected that on single-binary/no-runtime-service grounds. Use the live-verified engine
set from that memory: **Mojeek** (primary; `<a class="title" href>`, clean URLs),
**Bing** (curl+UA; base64url-unwrap `bing.com/ck/a?...&u=a1<...>`), **DuckDuckGo** (keep
existing path), **Mwmbl** (no-key JSON, optional). Dedup = normalize URL + round-robin
interleave + bump hit_count≥2. Reuse `jx_*` JSON parser (lift from `lsp.c:182` → `util.c`).

## Fetch tiers (`web_fetch` / `fetch_and_extract` rewrite)
```
tier 0  host rewrites      reddit→old.reddit, discourse /t/<slug>/<id>→.json
tier 1  curl -sL --compressed -A <Chrome UA>     (UA is the ONLY load-bearing header)
tier 2  smart extract      w3m -dump → JSON-LD articleBody → og:title/og:description
        ── Claude/Codex parity ends here (deterministic, no browser) ──
tier 3  Firefox/Marionette gated; replaces mini_browser (QuickJS)
PDF     curl → pdftotext (keep existing branch, web.c:170)
```
All tiers run behind the **SSRF guard** (reject private/loopback/link-local/reserved IPs,
metadata hosts, `.local/.internal/.lan`; re-validate every redirect hop).

## Tier-3: Firefox via native Marionette (spawn-per-fetch)
- **Launch:** `firefox -headless -marionette -no-remote -profile <mktemp -d>`; listens on
  `127.0.0.1:2828`.
- **Protocol:** netstring-framed JSON `<bytelen>:<json>`; read handshake →
  `WebDriver:NewSession` → `WebDriver:Navigate{url}` → (wait for load) →
  `WebDriver:GetPageSource` → feed HTML into the tier-2 extractor. Reuse `jx_*` parser +
  a raw TCP socket. No geckodriver binary.
- **Lifecycle:** spawn-per-fetch (user's call) — fresh Firefox + throwaway profile per
  tier-3 page, killed after. Cost ~1–2 s cold start; acceptable because tier-3 is rare/gated.
- **Gating:** invoke only when curl is 2xx AND extracted text is thin AND not a known-walled
  signature (x.com SPA, CF-managed-challenge). Same condition as the redesign decision.
- **Retire** the hardcoded `MINI_BROWSER` path (`web.c:21`); Gecko renders the SPA/CF pages
  QuickJS couldn't (memory: x.com 510 B, Hashnode 0 B).
- **Honest cost:** Firefox is a heavyweight runtime dep launched as a subprocess — a real
  departure from single-binary, but gated, on-demand, FOSS, user-owned. Walled pages
  (TLS/JA3, heavy CF) stay unreachable for any mission-allowed tool → "walled/JS-required",
  skipped (non-fatal, multi-result design).

## Reuse existing BASI bones
fork-based parallel fetch (`web.c:343`), claim-dir PDF dedup (`web.c:334`), `StringBuf`,
`run_command`, `url_encode`/`url_decode`, the PDF→pdftotext branch.

## Phasing (deterministic-first; ship parity before the browser)
1. **Parity, no browser:** split into `web_search`+`web_fetch`; curl-first tiers 0–2;
   SSRF guard; ranking port. ← biggest UX win; matches Claude/Codex.
2. **Metasearch:** Mojeek/Bing/DDG fan-out + dedup, replacing DDG-Lite-only.
3. **Tier-3:** Firefox/Marionette spawn-per-fetch; retire mini_browser.
4. **Cache + polish:** SHA-256 file cache, TTLs, LRU; tune ranking.

## Open decisions
- [confirm] Tool split (more loop turns) vs keep a one-shot combo. Recommend split.
- [confirm] Drop `grep_query` coupling — `web_fetch` returns full clean content (Claude/Codex
  style), agent greps mentally. Recommend drop.
- [verify] Which search operators (`site:`, OR/AND) Mojeek/Bing/DDG honor before porting
  `enhance_query`.

Relates to [[project_basi_cli]], [[feedback_deterministic_first]],
[[feedback_optimize_for_reliability]], [[feedback_no_proprietary_no_google]].
