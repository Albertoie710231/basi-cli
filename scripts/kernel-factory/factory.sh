#!/usr/bin/env bash
# Factory: a phased clean-session pipeline for a code-optimization task.
# Each phase is a FRESH `basi-cli -p` process (a genuinely clean session — new
# process, no shared context), HARD-SCOPED to only its phase's tools (--tools), given
# ONE directive plus the prior phase's output. The conveyor belt between phases is a
# file (.basi/findings.md). No nudges, no in-loop convergence: the phase boundary
# (a fresh process + a round cap) IS the hard convergence.
set -u
BASI=/home/alberto/Documentos/BASI-CLI/basi-cli
export BASI_ATTACH=1 BASI_SERVER_PORT=8181
export BASI_BASH_TIMEOUT=1800   # experiment phase builds llama.cpp (minutes)
MODEL=/home/alberto/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q6_K.gguf
REPO=/home/alberto/llama.cpp
SC="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
mkdir -p .basi
rm -f .basi/findings.md

phase() {  # name tools iters directive
  local name="$1" tools="$2" iters="$3" directive="$4"
  echo "════════ PHASE: $name  [tools: $tools, cap: $iters] ════════"
  BASI_MAX_TOOL_ITERS="$iters" "$BASI" -p "$directive" --tools "$tools" --yolo \
    > "$SC/factory-$name.log" 2>&1
  echo "  (phase $name done; log: factory-$name.log)"
}

# ── Phase 1: INVESTIGATE (navigation + write findings) → .basi/findings.md ──
phase investigate "read,head,tail,grep,wc,symbols,edit" 30 \
"You are the INVESTIGATION phase of a factory pipeline. GOAL: find the SINGLE most
promising, low-risk change to speed up LLM token-generation (decode) on the Intel Arc
B580 in llama.cpp's VULKAN backend (ggml/src/ggml-vulkan/ggml-vulkan.cpp). The B580 is
detected as INTEL_XE2. METHOD: this backend has branches that check the GPU vendor and
architecture and do different things per GPU. Find how the code checks
vendor/architecture, read what it does DIFFERENTLY for this GPU vs others, and decide
FOR YOURSELF which of those choices looks suboptimal, overly conservative, or wrong for
a B580 — that is your candidate. Do NOT assume; verify against the code.

YOUR ONLY DELIVERABLE is the file .basi/findings.md. Write it with the edit tool (leave
search empty, put the report in replace). Do a FEW targeted reads/greps, then WRITE it.
It MUST end with a ready-to-apply EDIT SPEC in EXACTLY this format (so the next phase can
apply it mechanically — no interpretation):

EDIT-SPEC
file: <relative path, e.g. ggml/src/ggml-vulkan/ggml-vulkan.cpp>
<<<<<<< SEARCH
<the EXACT current lines from the file, copied VERBATIM — read the file first so this
matches character-for-character, including indentation>
=======
<the EXACT replacement lines>
>>>>>>> REPLACE

The SEARCH block MUST be copied verbatim from the real file (read it to get it exact) or
the next phase cannot apply it. Keep it small — just the lines that change plus one or
two for context. Do not modify any source file yourself — only write .basi/findings.md."

echo "── findings.md written? ──"
if [ ! -s "$REPO/.basi/findings.md" ]; then
  # Fallback: capture the phase's final text answer as the findings.
  echo "(no findings.md; falling back to the phase's final answer)"
  sed 's/\x1b\[[0-9;]*m//g' "$SC/factory-investigate.log" \
    | grep -avE '^┌|^└|^│|thinking|\[K|\[2A|^\[ (Prompt|Generation)|^● |Goodbye|tool budget|attached to' \
    | awk 'NF' | tail -60 > "$REPO/.basi/findings.md"
fi
if [ -s "$REPO/.basi/findings.md" ]; then
  echo "YES ($(wc -l < "$REPO/.basi/findings.md") lines)"; echo "----"; cat "$REPO/.basi/findings.md"; echo "----"
else
  echo "NO — investigation produced nothing usable; stopping."; exit 1
fi

# ── Phase 2: IMPLEMENT (read + edit only) → edits the source file ──
git -C "$REPO" stash list >/dev/null 2>&1
BEFORE=$(git -C "$REPO" status --short | grep -vE '\.basi' | wc -l)
phase implement "read,edit" 8 \
"You are the IMPLEMENTATION phase of a factory pipeline. Read .basi/findings.md ONCE — it
ends with an EDIT-SPEC block: a file path, then a SEARCH block (exact current text) and a
REPLACE block. Your job is to APPLY it, in ONE edit call:
  edit <file from the EDIT-SPEC>, with search = the SEARCH block text and
  replace = the REPLACE block text.
Then STOP. If the edit tool reports the SEARCH text was not found, read the target file
ONCE at that location to get the exact current text, fix the search text, and edit again.
Do NOT re-read findings.md repeatedly, do NOT investigate, do NOT modify anything else,
do NOT touch .basi/findings.md. Making the one edit is your entire job."

AFTER=$(git -C "$REPO" status --short | grep -vE '\.basi' | wc -l)
echo "── source files modified: before=$BEFORE after=$AFTER ──"
git -C "$REPO" status --short | grep -vE '\.basi'
echo "── the diff the factory produced ──"
git -C "$REPO" diff --stat

# ── Phase 3: EXPERIMENT (bash only) → build patched vs baseline, .so-swap, benchmark ──
if [ "$AFTER" -eq 0 ]; then
  echo "── no edit produced by implement; nothing to measure. ──"; exit 0
fi
phase experiment "bash" 25 \
"You are the EXPERIMENT phase of a factory. A code change was applied to
ggml/src/ggml-vulkan/ggml-vulkan.cpp (see: git -C $REPO diff) to try to speed up LLM
token-generation on the Intel Arc B580. Your ONLY job: MEASURE whether it helps, and
report the verdict FROM THE NUMBERS. Do not investigate the code.

HOW THE BUILD WORKS: llama-bench loads libggml-vulkan.so.0 at RUNTIME, so to compare the
change you must build and SWAP THE .so, not the launcher. The real .so is
$REPO/build_vulkan/bin/libggml-vulkan.so.0.13.1 (libggml-vulkan.so.0 is a symlink to it).
Builds are slow but have a long timeout. Do exactly this (a script is fine):
1. PATCHED (edit already in the tree):
   cmake --build $REPO/build_vulkan --target ggml-vulkan -j 8
   cp -L $REPO/build_vulkan/bin/libggml-vulkan.so.0 /tmp/vk-patched.so
2. BASELINE (revert, rebuild, restore):
   git -C $REPO stash
   cmake --build $REPO/build_vulkan --target ggml-vulkan -j 8
   cp -L $REPO/build_vulkan/bin/libggml-vulkan.so.0 /tmp/vk-baseline.so
   git -C $REPO stash pop
3. Benchmark BOTH, swapping the .so each time, model = $MODEL :
   cp /tmp/vk-baseline.so $REPO/build_vulkan/bin/libggml-vulkan.so.0.13.1
   $REPO/build_vulkan/bin/llama-bench -m $MODEL -ngl 99 -n 64 -r 5
   cp /tmp/vk-patched.so  $REPO/build_vulkan/bin/libggml-vulkan.so.0.13.1
   $REPO/build_vulkan/bin/llama-bench -m $MODEL -ngl 99 -n 64 -r 5
4. Read the tg64 (token-generation) tok/s for each (ignore the pp512 prefill row).
   Report: baseline tg, patched tg, and the VERDICT — meaningfully faster, or within
   noise? A no-change result is a valid, honest finding; report it as such, do not inflate."

echo "════════ factory complete ════════"
