# BASI-CLI

A local-LLM **agentic CLI written in C**, built directly on [llama.cpp](https://github.com/ggml-org/llama.cpp).
BASI runs a model locally and drives it through a tool loop — read files, search and read the web,
consult a local knowledge base, query C code structure, and run multi-round **deep research** — all
from a single terminal binary.

**Design principle — *deterministic-first*:** anything that can be implemented deterministically (in
C) is; the LLM is used only for the irreducible reasoning/pattern-matching. The agent loop, tool
dispatch, ranking, guards, and parsing are plain C — the model just decides and synthesizes.

> Personal research project. The build defaults point at the author's paths — override them (below).

## Features

- **Agentic tool loop** — the model requests tools with `<tool>…</tool>`; results feed back as
  `<tool_result>` until it answers.
- **Deep research** (`--deepsearch` / `/deepsearch`) — a multi-round ReAct loop (search → read →
  synthesize) that runs in its own isolated context, distills each page to goal-relevant evidence,
  and won't answer until it has actually searched. Produces a cited answer.
- **Web** — `web_search` (thin client to a local [SearXNG](https://github.com/searxng/searxng);
  ranked, auto-fetches the top results) and `web_fetch` (curl-first, SSRF-guarded, extracts HTML via
  w3m and PDFs via pdftotext).
- **Knowledge base** — `docs_*` tools over a local `.basi/knowledge/` corpus, including semantic
  vector search backed by a GGUF embedding model.
- **Code context** — `code_context` surfaces C symbol info via a clangd LSP.
- **Editing & scaffolding** — `apply_patch` (structured diffs) and `scaffold` (templates), gated by
  an approval prompt.
- **Planning pillar** — an A3 / spike / pre-mortem plan workflow with phase-gated tools.
- **Native chat templates** — each model is driven in its *own* chat format via llama.cpp's jinja
  engine (Qwen, Gemma, DeepSeek, custom merges…), not a one-size-fits-all fallback.

## Requirements

- A C/C++ toolchain (`gcc`/`g++`, C17/C++17).
- **llama.cpp built as shared libraries.** BASI links `-lllama`, `-lggml`, `-lggml-base`, and
  `-lllama-common` (the same shared common lib `llama-cli` uses). The Makefile assumes a Vulkan build
  (`build_vulkan/`, `-lvulkan`) — adjust `LLAMA_BUILD` and the backend lib for your setup.
- A model in **GGUF** format. Instruct models follow the tool format most reliably.
- Optional runtime tools: `curl`, `w3m`, `pdftotext` (poppler), `unzip`, and a local **SearXNG**
  instance with the JSON format enabled (for `web_search`).

## Build

```sh
make LLAMA_DIR=/path/to/llama.cpp LLAMA_BUILD=/path/to/llama.cpp/build
```

`LLAMA_DIR`/`LLAMA_BUILD` default to the author's paths; override them for your machine. The headers
come from `$(LLAMA_DIR)/include`, `$(LLAMA_DIR)/common`, and `$(LLAMA_DIR)/vendor`; the libs from
`$(LLAMA_BUILD)/bin`.

## Usage

Interactive:

```sh
./basi-cli -m /path/to/model.gguf
```

Non-interactive (one-shot, prints and exits):

```sh
./basi-cli -m model.gguf -p "your prompt"                  # single agent turn (with tools)
./basi-cli -m model.gguf -p "your prompt" --no-tools       # flat completion (no tools)
./basi-cli -m model.gguf --deepsearch "your question"      # multi-round deep research
```

`--no-tools` turns `-p` into a single plain completion: no agent loop, no tool-instruction
system prompt, and **only the completion is printed to stdout** (load chatter goes to stderr) —
clean to capture in a pipe or use as a local teacher for data generation. It uses a minimal
"helpful assistant" system prompt by default; override it with `-s "<system>"`, or pass `-s ""`
for a pure completion of the prompt alone.

### Model & sampling flags

These apply to every mode (interactive, `-p`, `--no-tools`, `--deepsearch`):

| Flag | Default | Purpose |
|---|---|---|
| `-ngl`, `--ngl <n>` | `99` | Model layers to offload to GPU. **`0` = CPU only.** |
| `-c`, `--ctx <n>` | `32768` | Context size in tokens. |
| `-t`, `--temp <f>` | `0.4` | Sampling temperature (`0` = greedy). |
| `--seed <n>` | random | RNG seed. Fix it for reproducible output; vary it for diverse data-gen. |

Explicit flags win over the model picker's chosen values. Example — reproducible CPU-only
completion: `./basi-cli -m model.gguf -ngl 0 --seed 42 -p "..." --no-tools`.

Interactive slash commands include `/help`, `/deepsearch <q>`, `/plan`, `/permissions`, `/clear`,
`/cost`, `/model`.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `BASI_MODEL` | — | Default model path if `-m` is omitted |
| `SEARXNG_INSTANCE` | `http://localhost:8888` | SearXNG endpoint for `web_search` |
| `SEARXNG_HOME` | `~/Documentos/searxng` | Local SearXNG to auto-launch at startup |
| `BASI_NO_SEARXNG` | — | Set to disable the SearXNG auto-launch |
| `BASI_DEEPSEARCH_ROUNDS` | `5` | Max deep-research rounds |
| `BASI_DEEPSEARCH_CTX` | `32768` | Deep-research context size (lower for interactive `/deepsearch` on a single GPU) |

## License

[MIT](LICENSE).
