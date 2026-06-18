---
slug: note-cli
status: active
created: 2026-05-01
goal: Build a tiny Python note-taking CLI ("noted") in three independent phases, each step executable by a clean-context worker LLM.
---
# Note CLI — stress test for self-contained plan steps

## Theme
Tiny note-taking CLI ("noted"), written in Python, structured as a manager/worker test: each step in this plan is a self-contained directive a worker LLM can execute without having read any prior step.

## Background
The plan-phase pillar in BASI-CLI today produces a terse implementation table where rows reference each other implicitly ("extend the function from row 2"). That works for human readers but is brittle for small worker LLMs running steps in isolation. This plan is a hand-authored mock — written by a frontier model — that BASI's in-tree 7B will execute one step at a time. If the worker completes it, the manager/worker split is worth implementing. If not, the experiment is dead.

## Current Condition
No `noted` package exists. The test runs from the BASI-CLI repo root (BASI's cwd convention) — the worker creates `noted/` and `pyproject.toml` at the repo root; both are gitignored as mock artifacts. Python 3.11 is available; `~/.noted/notes.json` should not exist at start (verify clauses that need a clean slate run `rm -f ~/.noted/notes.json` themselves). The worker has BASI-CLI's tool surface (read, edit, bash).

## Cause Analysis
Small models lose context across long tool-call chains. They also struggle to reconcile "the function I wrote in step 2" with "the call site I'm now editing in step 5" because the intermediate file state may not match their mental model. Self-contained steps remove that coupling — the only thing left is on-disk file state, which the worker can inspect directly.

## Target Condition
A `noted` CLI installable via `pip install -e .`, supporting: `add <body> [--tag T...]`, `list`, `show <id>`, `delete <id>`, `search <query>`, `tag <id> <tag>`. Storage at `~/.noted/notes.json` (single JSON array, created lazily). All 8 verify clauses pass when run sequentially from a clean working directory.

## Implementation Plan

### Phase 1 — Storage layer
Goal: `noted/store.py` exposes pure functions for JSON-backed note storage. No CLI in this phase. Phase ends with a smoke test that exercises every function.

| id | title | deliverable | depends_on | touches | verify | status |
|----|-------|-------------|------------|---------|--------|--------|
| 1.1 | Scaffold package | Create `noted/__init__.py` (empty file), `noted/store.py` (empty for now), and `pyproject.toml` declaring package `noted`, `requires-python = ">=3.11"`, no runtime deps, and entry point `noted = "noted.cli:main"` (the cli module is created in phase 2 — declare the entry point now). | — | `noted/__init__.py`, `noted/store.py`, `pyproject.toml` | `test -f noted/__init__.py && test -f noted/store.py && test -f pyproject.toml` | draft |
| 1.2 | Storage path + load | In `noted/store.py`, add `import os, json` and `from datetime import datetime`. Define `_storage_path() -> str` returning `os.path.expanduser("~/.noted/notes.json")` (use `os.path.expanduser` literally — not `pathlib.Path.home()`). Define `_load() -> list[dict]` which creates the parent directory via `os.makedirs(os.path.dirname(_storage_path()), exist_ok=True)`, returns `[]` if the file is missing or has size 0, otherwise returns `json.load(open(path))`. | 1.1 | `noted/store.py` | `python -c "from noted.store import _storage_path, _load; print(_storage_path()); print(_load())"` | draft |
| 1.3 | CRUD functions | In `noted/store.py`, append: `_save(notes: list[dict]) -> None` writes `json.dump(notes, fp, indent=2)` to `_storage_path()`. `add_note(body: str, tags: list[str]) -> dict` assigns `id = max((n["id"] for n in existing), default=0) + 1`, sets `created_at = datetime.now().isoformat()`, appends to the loaded list, saves, returns the new note dict. `list_notes() -> list[dict]` returns `_load()`. `get_note(note_id: int) -> dict | None` returns the matching note or `None`. `delete_note(note_id: int) -> bool` removes the matching note, saves, returns `True` if found, else `False`. Do not modify functions defined in 1.2. | 1.2 | `noted/store.py` | `python -c "from noted import store; n = store.add_note('hello', ['x']); assert store.get_note(n['id']) == n; assert store.delete_note(n['id'])"` | draft |
| 1.4 | Phase-1 smoke | No new code. Run a smoke test that adds 3 notes, lists them, deletes all 3, and verifies the list is empty. | 1.3 | (none) | `rm -f ~/.noted/notes.json && python -c "from noted import store; [store.add_note(f'n{i}', []) for i in range(3)]; ids=[n['id'] for n in store.list_notes()]; assert len(ids)==3; [store.delete_note(i) for i in ids]; assert store.list_notes()==[]"` | draft |

### Phase 2 — CLI scaffold + basic commands
Goal: `noted/cli.py` with argparse subcommands `add`, `list`, `show`, `delete`, wired to the store functions from phase 1.

| id | title | deliverable | depends_on | touches | verify | status |
|----|-------|-------------|------------|---------|--------|--------|
| 2.1 | CLI scaffold | Create `noted/cli.py`. Add `import argparse, json, sys` and `from noted import store`. Define `main()`: build `argparse.ArgumentParser`, call `subparsers = p.add_subparsers(dest="cmd", required=True)`. Register four subcommands — `add` (positional `body`, optional repeatable `--tag` declared with `action="append"` so it collects into a list), `list` (no args), `show` (positional `id` typed as `int`), `delete` (positional `id` typed as `int`). Each subparser does `set_defaults(func=cmd_<name>)`. `main()` ends with `args = p.parse_args(); args.func(args)`. Define empty stub handlers `cmd_add`, `cmd_list`, `cmd_show`, `cmd_delete` that take `args` and `pass`. | 1.4 | `noted/cli.py` | `pip install -e . >/dev/null 2>&1 && noted --help | grep -qE '\badd\b' && noted --help | grep -qE '\blist\b'` | draft |
| 2.2 | CLI handlers | In `noted/cli.py`, replace the stub bodies. `cmd_add(args)`: `n = store.add_note(args.body, args.tag or []); print(f"added note {n['id']}")`. `cmd_list(args)`: for each `n` in `store.list_notes()` print `f"{n['id']:>3}  {n['body'][:60]}  [{','.join(n['tags'])}]"`. `cmd_show(args)`: `n = store.get_note(args.id)`; if `n is None` write `f"note {args.id} not found"` to `sys.stderr` and `sys.exit(1)`, else `print(json.dumps(n, indent=2))`. `cmd_delete(args)`: if `store.delete_note(args.id)` print `f"deleted note {args.id}"`, else write error to stderr and `sys.exit(1)`. Do not change `main()` or the parser. | 2.1 | `noted/cli.py` | `rm -f ~/.noted/notes.json && noted add 'first note' --tag a --tag b && noted list | grep -q 'first note' && nid=$(noted list | head -1 | awk '{print $1}') && noted show $nid | grep -q '"body": "first note"' && noted delete $nid && [ -z "$(noted list)" ]` | draft |

### Phase 3 — Search and tags
Goal: extend the store with `search_notes` and `add_tag`; add matching `search` and `tag` subcommands to the CLI.

| id | title | deliverable | depends_on | touches | verify | status |
|----|-------|-------------|------------|---------|--------|--------|
| 3.1 | Store: search + tag | In `noted/store.py`, append two functions at the end of the file. `search_notes(query: str) -> list[dict]` returns the list of notes whose `body` contains `query` case-insensitively (compare via `query.lower() in n["body"].lower()`). `add_tag(note_id: int, tag: str) -> bool` loads notes, finds the matching note, appends `tag` to its `tags` list only if not already in the list, saves, returns `True` on found, `False` otherwise. Do not modify any existing function. | 1.3 | `noted/store.py` | `rm -f ~/.noted/notes.json && python -c "from noted import store; n = store.add_note('hello world', []); ns = store.search_notes('WORLD'); assert len(ns)==1; assert store.add_tag(ns[0]['id'], 'greeting'); assert 'greeting' in store.get_note(ns[0]['id'])['tags']"` | draft |
| 3.2 | CLI: search + tag | In `noted/cli.py`, inside the existing `main()` function, register two more subcommands alongside the existing four — `search` (positional `query`) with `set_defaults(func=cmd_search)`, `tag` (positional `id` typed as `int`, positional `tag`) with `set_defaults(func=cmd_tag)`. Add `cmd_search(args)`: print each match from `store.search_notes(args.query)` using the same one-line format as `cmd_list`. Add `cmd_tag(args)`: if `store.add_tag(args.id, args.tag)` print `f"tagged note {args.id} with {args.tag}"`, else write error to stderr and `sys.exit(1)`. Do not modify the existing four handlers. | 2.2, 3.1 | `noted/cli.py` | `rm -f ~/.noted/notes.json && noted add 'find me' && nid=$(noted list | head -1 | awk '{print $1}') && noted search 'FIND' | grep -q 'find me' && noted tag $nid important && noted show $nid | grep -q 'important'` | draft |

## Follow-Up
After all 8 verify clauses pass, evaluate: how many steps did the worker complete on first try? Where did it need retries? Were retries due to step ambiguity (manager's fault) or tool-format failures (unrelated to the experiment)? On pass, design the manager/worker tool surface — `plan_verify <phase>` filtering, per-phase context window, manager invocation point. On fail, document the specific failure modes in `project_basi_cli.md`; do not implement manager/worker.

## Pre-mortem

### Failure modes
1. Worker hallucinates a function name not declared in its step (e.g. calls `store.find_notes` instead of `search_notes`). Mitigation: each step repeats every required identifier verbatim.
2. Worker writes the import as `from store import ...` (omitting the package). Mitigation: every importing step says `from noted import store` or `from noted.store import ...` literally.
3. Worker conflates `--tag` (repeatable list) with `--tags` (single comma-separated). Mitigation: step 2.1 specifies `action="append"` literally.
4. Worker forgets to `pip install -e .` between phase 1 and phase 2. Mitigation: phase 2.1's verify clause runs the install itself.
5. Worker uses `pathlib.Path.home()` instead of `os.path.expanduser`. Mitigation: step 1.2 names `os.path.expanduser` and explicitly forbids `pathlib`.

### Plan revisions
- Phase 2 split into scaffold + handlers so a structural argparse error fails 2.1 before handler logic muddies diagnosis.
- `pyproject.toml` merged into 1.1 (not its own step) — short content, and isolating it would cost the worker an extra context-rebuild for no gain.

### Unaddressed risks
- Worker may produce valid Python that still doesn't put `noted` on PATH (entry-point pinning quirks). If 2.1's verify fails, fallback is to invoke as `python -m noted.cli` and amend the plan accordingly.
- `~/.noted/notes.json` lives in `$HOME`, not the per-test directory. A failed run leaves residue; verify clauses that need a clean slate begin with `rm -f ~/.noted/notes.json`.
