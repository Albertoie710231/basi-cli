# Knowledge Base + Planning — Design Plan

Status: brainstorm captured 2026-04-27. No implementation yet. Pick up next session.

## Why this matters

The pillar that makes BASI truly model-agnostic. Today BASI works for things the local model already knows. With a curated, navigable knowledge base feeding a planning phase, BASI works on **anything the user has docs for** — niche libraries, internal APIs, framework versions older or newer than the model's cutoff, languages the model was barely trained on (Godot/GDScript), and even projects where the official docs are insufficient and the user is building their own corpus.

The model's training cutoff stops being a ceiling. The user's curated knowledge base is.

This is the deterministic-first motto applied to *knowledge*: domain facts live in deterministic storage; the LLM is invoked only for synthesis on top of that store.

## Project thesis

**The architecture is the durable asset. The model is a replaceable component.**

If the substrate is right — planning phase, curated knowledge base, deterministic tools, structured artifacts, explicit contracts between roles — then any LLM with enough capability slots in and behaves well. Frontier models perform well because they have the best judgment for the planner role. Local 7Bs perform well because they're never asked to do work above their pay grade. A new model released next quarter is a drop-in upgrade. A model that regresses is detected fast because the artifacts are structured and the failures are legible.

Three earlier framings in this project all reduce to the same thesis:

- **Deterministic-first motto** — anything that can be implemented deterministically, must be. The LLM is the smallest part of the system, not the orchestrator.
- **Reliability over tokens** — what we optimize for is goal-completion-rate. Goal-completion comes from the substrate, not the model.
- **Manager/worker split** — split work along capability lines, define contracts between roles, swap either side independently.

Every design decision in this plan should be evaluated against this thesis: *does it make the architecture more durable, or more dependent on a particular model behaving a particular way?*

## Three pillars (interrelated)

### 1. Planning (the entry point)
Every non-trivial task starts with a planning phase. The LLM consumes the doc DB heavily, drafts a plan, validates assumptions against documentation, iterates with the user. Plan is the artifact that anchors the rest of the project.

### 2. Knowledge base / doc DB (the substrate)
A navigable, structured, multi-source corpus the user curates. The LLM doesn't decide whether to consult it — at planning time it's mandatory; during execution it's consulted whenever the plan breaks (which is often).

### 3. Manager/worker split (Tier 3, lands naturally here)
- Planner = paid frontier LLM (judgment-heavy, doc-heavy, low-volume).
- Executor = local 7B (well-specified, high-volume, deterministic-leaning).
- The phase model is the contract between them. Architect-mode pattern (Aider) but with a real structured doc DB underneath instead of relying on whatever the planner happens to know.

## Phase model

| Phase | Doc-DB usage | LLM role | User role |
|---|---|---|---|
| **Planning** | heavy / dense | judgment, drafts plan | in the loop, catches wrong reads |
| **Execution** | sparse / targeted | follows plan, edits code | reviews diffs |
| **Re-planning** | heavy again | redrafts when plan invalidates | confirms scope shift |

Re-planning is **frequent** — plans break in real projects, and fast targeted retrieval during execution is what makes recovery cheap. Earlier framing ("retrieval is rare during execution") was wrong; correction: retrieval is *low-frequency-by-default but high-importance-when-needed*. Ergonomics of the retrieval tools matter as much as their existence.

## Why prior RAG attempts failed

User's previous implementation: vector DB built from a folder of files, retrieves top-K (5) chunks per query, model decides when to retrieve.

Failure modes (in observed order of severity):

1. **The "decide whether to retrieve" step.** Pure judgment, given to the small model. Small models don't reliably notice when they don't know something — they hallucinate confidently instead of asking. This is a deterministic-first violation: probabilistic component doing a routing decision a deterministic mechanism could do.
2. **Top-K is arbitrary.** Five chunks is sometimes too broad (noise drowns the signal), sometimes too narrow (the relevant section is the 6th). No principled tuning.
3. **Vendor lock-in via embedding quality.** Best embedding model the user worked with was paid OpenAI. Open embeddings consistently weaker. Re-introduces exactly the corporate dependency BASI exists to avoid.

## Doc DB design — current direction

### Retrieval-as-navigation, not retrieval-as-similarity

Tool-calling navigation **beats** vector RAG for structured docs. Model walks the doc tree (TOC → section → cross-link) the way a careful human reads. The model gets exactly what it asks for and *knows* what it didn't get — vector search hides what was filtered out.

### The librarian's toolkit (the LLM-facing API)

Primary mechanism. Each is deterministic.

- `docs_toc()` — table of contents (titles + paths only, no bodies)
- `docs_get <path>` — fetch a specific section by title-path
- `docs_search <keyword>` — full-text grep across the corpus, returns matching paths
- `docs_followlink <ref>` — chase a cross-reference from one section to another
- `docs_recent_notes` — what the user has added during *this* project
- `docs_vector_search <query>` — fallback only, used when above all fail. Demoted from primary to last-resort.

This makes the doc DB look more like a **structured filesystem of markdown** than a vector store. Vectors aren't gone, but they're not the API the model talks to.

### Multi-source, with precedence

The corpus is layered. Each layer is a "shelf":

1. **User notes** (highest precedence) — `/note "GDScript closures don't capture by reference"` — these can grow to *be* the documentation when official docs are insufficient.
2. **Pinned external sources** — specific blog posts, forum answers, paper excerpts the user explicitly added.
3. **Official docs** (lowest precedence) — imported in bulk via URL or local path.

When sources disagree, user notes win. The model is told this explicitly in the system prompt.

### Ingestion is first-class and refresh-friendly

Doc sources are *versioned references*, not one-time imports. Godot 4.3 → 4.4 changes things; `docs refresh` is a normal operation, not a rebuild from scratch. Each ingested source remembers its origin (URL or path) and last-fetched timestamp.

### User curation during the project

When the LLM hits a gap mid-execution, the user can:
- `/note "..."` — add a one-line user note immediately.
- `/ingest <url>` — add an external source on the fly.
- `/edit <doc-path>` — open a section in $EDITOR to refine.

The DB grows over the project's life. By the end, the user has a project-specific corpus that's more useful than the original imported docs.

## Forcing-function case studies

These shape the design. If the system can't handle these, it doesn't deliver on agnosticism.

### Godot / GDScript
Niche language, no LLM well-trained on it, evolves significantly between versions. Official docs are well-structured (classes → methods → signals), so navigation tools work great. **Re-ingestion must be cheap** because Godot updates.

### Trash docs
Sometimes official docs are wrong, incomplete, or missing. The user fills gaps from forums, source code, experimentation. By project end, **user notes ARE the documentation**. The system has to support this gracefully — user notes can't be second-class.

## Connection to existing BASI features

- **Plan mode (#7 from the agent-mode milestone)**: currently just blocks bash/apply_patch/scaffold and lets the model emit a `<plan>` block. This vision evolves it into a *phase boundary* that produces a **structured plan artifact saved to disk**, referenced for the rest of the project.
- **`code_context` tool**: same architectural pattern as the doc tools (semantic-only, structured navigation, no surrounding-text noise). The doc-DB tools should follow the same template — terse, fenced output, named-target inputs, no free-form queries.
- **Deterministic-first motto**: this feature is the defining example of the principle. Knowledge → deterministic storage. Navigation → deterministic tools. Synthesis → LLM.
- **Manager/worker (Tier 3)**: the phase model gives manager/worker a real contract. Without phases, manager/worker is just "split prompts." With phases, planner ↔ executor have well-defined inputs and outputs.

## Open design questions (pick up here)

1. **Plan artifact format.** Markdown with frontmatter? Structured JSON? Free text with a few mandatory sections (goals, constraints, file list, step list)? Trade-off: machine-parseable vs. human-friendly. **Front-runner: Toyota A3 format** (background / current state / goal / root cause / countermeasures / implementation plan / follow-up) — battle-tested in operations for decades, fits one page, parses easily. See Planning literature section below.
2. **Where does the plan live.** `./.basi/plan.md`? Per-feature plans in `./.basi/plans/<name>.md`? Single rolling plan vs. plan-per-task?
3. **Doc DB storage layout.** `./.basi/knowledge/` with subdirs per shelf (`/user-notes/`, `/external/`, `/official/`)? Single flat directory with metadata in frontmatter? Index file?
4. **Ingestion mechanism.** URL fetcher (libcurl?), local file copy, both? Markdown-only or also HTML→markdown via pandoc/readability?
5. **Refresh semantics.** What happens when an upstream source changes — diff-and-merge, blow-away-and-replace, version-history?
6. **Tool interface details.** Exact arg shape, output format, error messages (Codex-style "<tool> not allowed: <reason>" mold).
7. **How the LLM signals "I need more info during execution."** Slash command (user-driven), tool call (model-driven), or "plan break" detection that triggers re-planning?
8. **Integration with plan mode.** Does plan mode auto-load doc-DB tools? Auto-inject the TOC into the system prompt during planning only?
9. **Vector-search fallback details.** Which embedding model? Where stored? How to keep it from being the lazy default?

## Next session action items

In rough order:

- [ ] Read the planning-literature primary sources listed below (Polya 4-stage, A3, pre-mortem, WBS, spike, DoD). Mine concrete prompts and structural patterns we can port — same pattern we ran for Codex / Claude Code earlier.
- [ ] Decide doc-DB storage layout (#3 above) — concrete enough to build against.
- [ ] Decide plan artifact format (#1) — likely A3 unless something better surfaces from the literature mine.
- [ ] Sketch the librarian's toolkit interfaces (#6) — argument shapes, output shapes, sample model-facing prose.
- [ ] Decide how plan mode evolves from "tool blocker" to "produces plan artifact." Probably: enter planning → load TOC into system prompt + enable docs_* tools, exit planning → save artifact + disable docs by default during execution.
- [ ] Decide ingestion path (#4) — start with local files only? Add URL ingestion as v2?
- [ ] Mock up one full session (planning a real Godot feature) on paper to find rough edges before coding.

## Footnotes / corrections from brainstorm

1. **Open embedding models have caught up.** The "best embed model is paid OpenAI's" intuition was accurate ~2022–2023 but no longer structural. As of late 2025, **BGE-M3** (567M params, open), **jina-embeddings-v3**, **gte-Qwen2-7B-instruct** (7B, open), and **NV-Embed-v2** are competitive with or beating `text-embedding-3-large` on MTEB. The vector-search fallback path doesn't have to be vendor-locked.

2. **Embedding-model size vs. consumer hardware.** Three tiers: small (<200M, <500MB VRAM, "good"), medium (500M–1B, ~1–1.5GB VRAM, "competitive with OpenAI"), large (7B, 4–8GB VRAM, "top of MTEB"). **BGE-M3 is the pragmatic pick** for BASI's target user — 567M params, ~1GB VRAM quantized, quality on par with OpenAI's, runs alongside a 7B coding model on an 8GB GPU without strain. **Critical infrastructure win:** llama.cpp already supports embedding mode (`--embeddings`, `llama-embedding` binary), and GGUF builds of BGE-M3 / nomic-embed / e5 exist. No new inference backend required. The 7B embedding tier is off-limits for our target user (would need 14GB+ VRAM total alongside the coding model) — skip.

3. **The doc-DB benefits all model tiers.** Frontier models (Claude, GPT, Gemini) also gain from current/curated docs — they're just less *dependent* on it than 7B locals. The plan should hold "doc DB is mandatory infra for small models, valuable infra for big ones." This compounds with the manager/worker pattern: the planner (frontier) reads the DB to write a precise spec; the executor (local) reads the DB to verify mid-execution. Both need it, both benefit.

4. **Two distinct user scenarios coexist.** The plan above assumes the **per-project, single-curator** scenario (one user, tens-to-hundreds of docs, navigation-first). A second valid scenario exists: **corporate cross-project vector DB** (team-curated, thousands of docs, many projects, sensitive data, primary access via vector search not navigation). For corporate-scale shared knowledge, navigation tools don't fit (no single curator can build a TOC over 1000+ heterogeneous docs); vector search becomes primary. Local open-weight embeddings exist precisely to enable this without leaking data to a cloud vendor. **BASI's design accommodates both naturally** — same `docs_vector_search` tool, two configurations: individuals barely touch it (navigation handles ~all queries), corporate users wire it to a shared local index and lean on it heavily.

## Planning literature worth mining

Same pattern as the Codex / Claude Code research earlier: harvest battle-tested patterns from a discipline that has thought about this for decades. Reinventing planning from first principles when there's a century of operational experience is a deterministic-first violation.

Bodies of knowledge most directly relevant:

1. **Polya, "How to Solve It" (1945)** — four-stage heuristic: *understand the problem, devise a plan, carry it out, look back*. Identical structure to our phase model. **What to port:** the specific self-interrogation prompts ("Can you restate the problem in your own words?", "Is there a related problem you've solved before?", "Can you derive the result differently?") — cheap to encode in the planner's system prompt, high yield for small models that otherwise jump straight to coding.

2. **Work Breakdown Structure (PMBOK)** — discipline of decomposing work into the smallest verifiable tasks. **What to port:** the structural rule that a plan is a tree where every leaf is small enough to execute in one tool-call cycle. Solves "vague plans" by enforcing decomposition at the artifact level, not asking the model to know what fine-grained means.

3. **A3 problem-solving format (Toyota)** — single-page plan template with mandatory sections: *background / current state / goal / root cause / countermeasures / implementation plan / follow-up*. **Front-runner for the plan artifact format.** Tested against decades of operational use, fits one page, parses easily. Forces the planner to fill the right boxes rather than meander.

4. **Pre-mortem (Gary Klein, 1998)** — "imagine the project has already failed; explain why." Extraordinarily strong for surfacing assumption gaps and risks. **What to port:** a `/premortem` slash command that runs the planner adversarially against the current plan. Cheap to add, high signal — and exactly the kind of judgment-flip a frontier-model planner is well-suited to.

5. **Spike (Scrum)** — time-boxed exploration to reduce uncertainty *before* committing to a plan. This is **literally what the user already does as a personal workflow** ("read docs before writing a single line of code"). **What to port:** the formal name and the time-boxing discipline. Suggests a Phase 0 (spike) before Phase 1 (planning) where the agent is read-only against the doc DB and can't even draft a plan yet.

6. **Definition of Done / Acceptance Criteria (Scrum)** — every task has testable completion conditions. **What to port:** every plan step in the A3 implementation-plan section gets a `verify:` field — a concrete check (test command, signature query, file existence) the executor runs before claiming done. Connects to "verification-through-analysis" (post-edit signature delta check) from the earlier code_context discussion.

**Optional but worth a skim:**

- **Critical Path Method (CPM)** — task dependency ordering. Useful for the executor to know which steps unblock others.
- **Means-Ends Analysis (Newell & Simon)** — recursive subgoaling. Mostly subsumed by WBS for our needs.
- **OODA loop (Boyd)** — *observe / orient / decide / act*. Maps to the execute-then-replan loop. Adds the "orient" step (situation assessment) explicitly between observation and action — useful for the re-planning trigger.

## Inheritance from settled work

- The deterministic-first motto applies and is the design's main constraint. See `feedback_deterministic_first.md`.
- The reliability-over-tokens metric applies. Doc DB is justified by goal-completion-rate, not token economics. See `feedback_optimize_for_reliability.md`.
- The manager/worker idea is documented in `project_manager_worker_idea.md` (Tier 3) and connects here as the natural execution model.
- The Codex/Claude-Code patterns harvested earlier (Tier 2 references) inform tool-description style for the doc-DB tools. See `reference_codex_patterns.md`, `reference_claude_code_patterns.md`.

---

# Design Decisions (locked 2026-04-28)

Action items #1–#7 from the brainstorm are resolved below. The brainstorm above stays as historical context; this section is what we build against.

Lit-mine summary lives at `reference_planning_literature.md`. The decisions in this section are derived from that mine.

## Decision #1 — Planning literature (item #1, done)

Mined Polya / A3 / WBS / DoD / pre-mortem / spike. Results saved to `reference_planning_literature.md`. Notable specific outputs that drive everything below:

- **A3 confirmed.** Use the Proposal A3 variant (forward-looking) over problem-solving A3.
- **Leaf-task rule confirmed and tightened:** single named artifact + one shell-checkable `verify:` + one tool-call cycle.
- **Polya prompts** drop verbatim into the planning system prompt as stage-checklists.
- **Pre-mortem framing locked:** Klein's past-tense "the project has failed" wording; +30% reason-identification benefit (Mitchell/Russo/Pennington 1989) is the load-bearing claim.
- **Spike phase rules locked:** read-only tools, 8k token / 12-call budget, exit-artifact triple Question/Findings/Decision.

## Decision #2 — Doc-DB storage layout

Filesystem hierarchy under `./.basi/`:

```
.basi/
├── BASI.md                  # CLAUDE.md analogue (milestone #4 of agent mode)
├── plans/                   # plan artifacts (this section)
│   ├── <slug>.md            # Proposal A3 plan
│   └── <slug>.spike.md      # spike artifact (when phase 0 ran)
└── knowledge/               # the doc DB
    ├── notes/               # user notes — HIGHEST precedence
    ├── pinned/              # explicitly added external sources
    └── docs/                # bulk-imported official docs — LOWEST precedence
```

Every file under `knowledge/` is markdown with frontmatter:

```yaml
---
source: <url|path|user-note>
shelf: notes|pinned|docs
title: <human-readable title>
fetched: 2026-04-28               # ISO date; absent for user-note
upstream_version: <optional>      # e.g. "godot-4.4.1"
---
```

Section addressing: full path + optional anchor. Example: `docs/godot-4.4/classes/Node.md#methods/queue_free`. Anchors are derived deterministically from the heading path — no hand-maintained anchor maps.

Precedence rule (encoded in code): when sources disagree, `notes/` > `pinned/` > `docs/`. The model is told this explicitly in the planning system prompt; the precedence value is also surfaced in `docs_search` output so the model can see which shelf a hit came from without a separate call.

**Why this layout** — it's a structured filesystem of markdown (the brainstorm's framing), navigable by `find`/`ls`/`grep` deterministically, no DB engine required. The shelf hierarchy matches the precedence semantics directly so the code never has to translate. Frontmatter survives `git mv` for refresh semantics later.

**Rejected alternative:** flat directory with shelf-in-frontmatter only. Rejected because precedence-by-directory is cheaper to enforce and humans read trees better than they read frontmatter.

## Decision #3 — Plan artifact format

Proposal A3 in markdown with YAML frontmatter, hard-capped at 200 lines:

```markdown
---
slug: add-knowledge-base
status: drafting | spike | premortem | active | done | abandoned
created: 2026-04-28
goal: <one-line theme>
---

# <human title>

## Theme
<one sentence — what we're trying to do>

## Background
<context, importance, why-now>

## Current Condition
<measures, not anecdote — failing test, reproducer cmd, missing file, etc.>

## Cause Analysis
<5-Why chain to root cause>

## Target Condition
<quantified — signature exists / test passes / file at path>

## Implementation Plan

| id  | title              | deliverable           | depends_on | touches              | verify                                         | status  |
|-----|--------------------|-----------------------|------------|----------------------|------------------------------------------------|---------|
| 1.1 | parse plan header  | src/plan.c:parse_hdr  | -          | src/plan.c, plan.h   | `cc -c src/plan.c && grep -q parse_hdr plan.h` | pending |
| 1.2 | ...                | ...                   | 1.1        | ...                  | ...                                            | pending |

## Follow-Up
<verification plan: how/when we'll check effects after execution>

## Pre-mortem
<populated by /premortem after draft, before active>

### Failure modes
<numbered list, distinct, past-tense framing>

### Plan revisions
<bullet diff — what changed in the plan body in response>

### Unaddressed risks
<failure modes accepted as residual>
```

**Hard cap:** 200 lines / ~6 KB. Validation rejects overflow; the model is forced to compress rather than truncate (Shook's "constraints are load-bearing" rule).

**Spike artifact** at `<slug>.spike.md`:

```markdown
---
slug: add-knowledge-base
phase: spike
created: 2026-04-28
---

## Question
<the specific uncertainty that triggered the spike>

## Findings
- <bullet — each cites a path or URL>
- ...

## Decision
PROCEED-TO-PLAN | NEED-ANOTHER-SPIKE | ABANDON
```

Hard cap: 80 lines.

## Decision #4 — Librarian's toolkit interfaces

Tool prose follows Codex style (3–4 sentences, namespaced names, error mold `<tool> not allowed: <reason>`).

```
docs_toc                        — no args
  Returns the table of contents of this project's knowledge base as a flat
  list of "<shelf>/<path>: <title>" entries with no bodies. Call this FIRST
  when you don't know what's in the corpus. Then call docs_get on a
  specific path to read a section. Hits across all three shelves
  (notes, pinned, docs) in precedence order.

docs_get <path>                 — path: string (required)
  Returns the body of a single document or section by path
  (e.g., "docs/godot-4.4/classes/Node.md#methods/queue_free"). Use after
  docs_toc has shown you the path. Returns plain markdown, capped at
  4 KB; if the section is larger, the response says so and asks you to
  request a more specific anchor.

docs_search <keyword>           — keyword: string (required)
  Full-text grep across the entire knowledge base. Returns up to 50
  matches as "<shelf>/<path>:<line>: <surrounding-line>". Literal
  substring match — NOT a semantic search. Use when you know a specific
  term but not where it lives.

docs_followlink <ref>           — ref: string (required), context: path
  Resolves a markdown cross-reference link (e.g.,
  "[queue_free](#methods/queue_free)" or "[Signal](Signal.md)") against
  the supplied source-doc context, returning the linked section's body.
  Use to chase references the way you'd click a link.

docs_recent_notes               — no args
  Returns all user notes added during this project session, in insertion
  order, with timestamps. Notes win over imported docs by precedence —
  call this BEFORE answering a question where the user might have
  corrected a prior assumption.

docs_vector_search <query>      — query: string (required)  [DEFERRED — v2]
  Last-resort semantic search across the corpus via BGE-M3 embeddings.
  Use ONLY when docs_toc, docs_search, and docs_recent_notes have all
  failed. Returns top-3 matches by cosine similarity. Does NOT replace
  navigation.
```

All tools are blocked outside plan mode and execute mode (i.e., always available — they're read-only, no approval needed).

**Output discipline** — every tool returns plain markdown or plain `path:line: ...` lines. No JSON wrappers. The model's prompt-template renders these directly into the conversation.

## Decision #5 — Plan-mode evolution

Today's plan mode is a tool-blocker that lets the model emit a free-text `<plan>` sentinel block. Replace with a phase machine driven by deterministic state transitions:

```
                +-----------+    assumptions >= 3      +----------+
  /plan <slug>  |           |  --------------------->  |          |
  ---------->   | drafting  |                          |  spike   |
                |           |  <---------------------  |          |
                +-----+-----+    decision = PROCEED    +----------+
                      |
                      | draft saved
                      v
                +-----------+    /premortem            +-----------+
                | drafting  |  -------------------->   | premortem |
                +-----------+                          +-----------+
                                                            |
                                                            | revisions saved
                                                            v
                                                       +----------+
                                                       |  active  |  -> exit plan mode
                                                       +----------+
```

State stored in the plan file's frontmatter `status:` field. Each transition is a deterministic check in C, not an LLM judgment:

- `drafting` → `spike`: model emits `{unverified: N}` JSON before plan-write; if N ≥ 3, code routes to spike.
- `spike` → `drafting`: spike artifact written with `Decision: PROCEED-TO-PLAN`.
- `drafting` → `premortem`: user invokes `/premortem` slash command.
- `premortem` → `active`: user invokes `/plan accept` (final user-in-the-loop gate).
- `active` → exits plan mode; executor takes over against the plan.

**Tools available per phase:**

| Phase     | docs_*  | code_context | readfile | webfetch | plan_write | bash | apply_patch | scaffold |
|-----------|---------|--------------|----------|----------|------------|------|-------------|----------|
| spike     | ✓       | ✓            | ✓        | ✓ (GET)  | ✗          | ✗    | ✗           | ✗        |
| drafting  | ✓       | ✓            | ✓        | ✓ (GET)  | ✓          | ✗    | ✗           | ✗        |
| premortem | ✓ (read) | ✓            | ✓        | ✗        | ✓ (append) | ✗    | ✗           | ✗        |
| active    | ✓       | ✓            | ✓        | ✓        | ✓ (status) | ✓    | ✓           | ✓        |

`plan_write` is a new tool — writes the plan artifact with validation (A3 sections present, leaf rules satisfied, banned words absent, 200-line cap).

## Decision #6 — Ingestion path

Three tiers, only v1 ships now:

**v1 (this milestone):**
- `basi-cli docs add <path> [--shelf=pinned|docs]` — copies a local markdown file into `./.basi/knowledge/<shelf>/`. Stamps frontmatter (`source`, `fetched`, `shelf`, `title` from H1).
- `/note "..."` — slash command appends a single-line entry to `./.basi/knowledge/notes/session-<date>.md` (one file per day).
- `/edit <doc-path>` — opens the doc in `$EDITOR` (defaults to whatever `EDITOR` is set; if unset, refuse with "<edit> not allowed: $EDITOR not set").

**v2 (deferred):**
- URL ingestion via libcurl. HTML→markdown via pandoc shell-out (most users have it; refuse cleanly if missing).
- `docs refresh <path>` — re-fetches by stored `source` URL, diffs, prompts user before replacing.

**v3 (deferred):**
- Bulk URL crawls (sitemap-driven).
- `docs_vector_search` backed by BGE-M3 via llama.cpp embedding mode.

Rationale: v1 covers the forcing-function case (Godot user pre-downloads docs locally, runs `basi-cli docs add`). v2 is convenience. v3 is the corporate-scale scenario from footnote #4.

## Decision #7 — Mock session (paper walk-through)

Walking the design through one full session for the Godot forcing-function case to surface rough edges before coding.

**Scenario:** user is working in a Godot 4.4 project. Asks BASI to add a `queue_free()` call to an enemy node when its HP reaches 0.

**Pre-state:** user has previously run:
- `basi-cli docs add /path/to/godot-docs/ --shelf=docs` (bulk Godot 4.4 docs)
- `/note "GDScript closures don't capture by reference — confirmed via experiment 2026-04-26"` (project-specific gotcha)

**Session walk:**

1. User: `/plan kill-on-zero-hp` — enters plan mode, status=`drafting`.
2. Model first emits `{unverified: [...], unverified: 4}` — it doesn't know whether `queue_free()` is on `Node` or `Node2D`, doesn't know GDScript syntax for HP comparison in this codebase, doesn't know if the existing enemy script has a `_process` or signal-driven update, doesn't know the project's lifecycle conventions.
3. Code routes to `spike` — 4 ≥ 3.
4. Model uses `docs_toc` → sees `docs/godot-4.4/classes/Node.md`, `docs/godot-4.4/classes/Node2D.md`, etc.
5. Model uses `docs_get docs/godot-4.4/classes/Node.md#methods/queue_free` → confirms `queue_free()` is on `Node` (parent of all).
6. Model uses `code_context` to find the existing enemy script's `take_damage` function.
7. Model uses `docs_recent_notes` → reads the "closures don't capture by reference" note (irrelevant here, but the discipline ran).
8. Model writes `kill-on-zero-hp.spike.md` — Question: "where to call queue_free in existing damage flow", Findings: "queue_free is on Node, take_damage exists at scripts/Enemy.gd:42, project uses signals not _process for HP updates", Decision: `PROCEED-TO-PLAN`. ~6 tool calls, ~3 KB I/O — well under budget.
9. Code transitions to `drafting`. Model writes the A3 with one leaf:
   - `1.1` — title: "queue_free on HP zero", deliverable: `scripts/Enemy.gd:on_hp_changed`, touches: `scripts/Enemy.gd`, verify: `grep -q queue_free scripts/Enemy.gd && godot --check-only scripts/Enemy.gd` (assumes godot CLI available).
10. User runs `/premortem`. Model emits failure modes: "queue_free called twice if signal fires twice", "free during physics step crashes engine", "no audio cue before despawn so player loses feedback". Plan revisions: add 1.2 (guard against double-fire), add 1.3 (call_deferred wrapper). Unaddressed risk: audio cue (out of scope).
11. User: `/plan accept`. Status → `active`. Plan mode exits.
12. Executor takes over: edits `scripts/Enemy.gd`, runs `verify:` after each leaf, marks done.

**Rough edges surfaced by the walk:**

- **Anchor extraction needs care.** `#methods/queue_free` requires that the bulk-imported Godot docs use a consistent anchor convention. If they're flat (one file per class with no method anchors), we either need to split during ingestion or rely on `docs_search` for in-doc lookups. **Action:** add to v1 ingestion: optional `--split-by-heading=H2` flag.
- **`docs_get` 4 KB cap is tight for whole-class pages.** A full Godot `Node.md` is ~30 KB. Without anchor splitting we choke on first read. **Action:** `--split-by-heading` is now mandatory-default for the `docs` shelf.
- **`verify:` for non-shell ecosystems.** `godot --check-only` may not exist on all setups. The `verify:` system needs a fallback: if `cmd` exits with `127` (command not found), the executor should report "verify-tool-missing" not "verify-failed" and prompt the user.
- **`/note` filename collision.** Two notes on the same day go into the same file — fine. But what about a note with backticks or quotes? **Action:** notes are appended as raw markdown; no escaping needed if we always wrap in a fenced block. Decide: append as bullet `- 14:32 <text>` — simpler.
- **Spike artifact lifecycle on re-spike.** If `Decision: NEED-ANOTHER-SPIKE`, do we overwrite or append? **Action:** append a numbered second `## Question`/`## Findings`/`## Decision` group to the same `.spike.md` file. Cap at 3 cycles; on the 4th, force `ABANDON`.

These five edges are now folded into the relevant decisions above:
- Storage decision #2 gets `--split-by-heading=H2` ingestion default for `docs/` shelf.
- Toolkit decision #4: `docs_get` overflow message must include the available anchors at the requested level.
- Phase decision #5: executor's verify-runner distinguishes exit code 127 (tool missing) from non-zero (failure).
- Ingestion decision #6: notes use `- HH:MM <text>` bullet format in `notes/session-<date>.md`.
- Plan-mode decision #5: spike re-entry caps at 3 cycles.

## Implementation order (after design)

Now that #1–#7 are locked, code in this order:

1. **Storage scaffolding** — directory creation, frontmatter parser, slug-name validation. Smallest concrete piece.
2. **Librarian toolkit (read-only)** — `docs_toc`, `docs_get`, `docs_search`, `docs_recent_notes`. `docs_followlink` last (parses links). Defer `docs_vector_search`.
3. **Ingestion v1** — `basi-cli docs add`, `--split-by-heading=H2`, `/note`, `/edit`.
4. **Plan artifact validator** — A3 section presence, leaf-rule check, banned-word lint, 200-line cap.
5. **`plan_write` tool** — uses the validator.
6. **Phase machine** — state transitions, tool gates per phase.
7. **`/premortem` slash command** — runs the protocol, appends section.
8. **Spike phase** — assumption gate, time-box, `.spike.md` write, re-entry cap.
9. **Executor verify-runner** — runs `cmd:`, distinguishes 127 vs. non-zero.
10. **`docs_vector_search`** — last; requires BGE-M3 GGUF + llama.cpp embedding mode wiring.

Each step has a single deliverable, a verify clause, and lands in one cycle — i.e. these are the leaf tasks of the *meta-plan* for building the doc-DB feature, by the same WBS rule we just defined. We dogfood from step 1.
