# Kernel discovery factory (prototype)

A phased, hard-scoped agent pipeline that turns a vague optimization question into
measured code changes. The LLM does the creative step (generate theories); the driver
does the mechanical step (apply → build → benchmark → rank) — deterministic-first.

- `factory-fanout.sh` — the fan-out: one LLM theory phase emits N candidates in a
  parseable `@@THEORY@@ … @@SEARCH@@/@@REPLACE@@ … @@END@@` format; the driver counts
  them (`parse_theories.py`), applies each (`apply_theory.py`), builds+benchmarks each
  vs one baseline (server killed first so the GPU is free/uncontaminated), and prints a
  ranked conclusion.
- `factory.sh` — the earlier linear pipeline (investigate → implement → experiment).
- Each phase is a fresh `basi-cli -p --tools <subset>` process: clean session, hard tool
  scope, one directive, file handoff. No nudges — the phase boundary is the convergence.

PROTOTYPE: paths to llama.cpp, the GGUF model, and the server are hardcoded for the dev
box; adapt before reuse. This drove the 2026-07 B580 kernel investigation (see the
project memory): 5 theories generated, all measured, none a real win, one a −4.14%
regression → no decode headroom, and the existing kernel choices are correct.
