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
