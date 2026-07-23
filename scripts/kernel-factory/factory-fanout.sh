#!/usr/bin/env bash
# Fan-out factory: ONE LLM theory phase generates N candidate changes in a parseable
# format; then a DETERMINISTIC loop applies + builds + benchmarks EACH against a single
# baseline, and ranks them by measured tok/s. The creative step is the LLM's; trying
# each theory and concluding is mechanical (deterministic-first).
set -u
BASI=/home/alberto/Documentos/BASI-CLI/basi-cli
export BASI_ATTACH=1 BASI_SERVER_PORT=8181
REPO=/home/alberto/llama.cpp
SC="$(cd "$(dirname "$0")" && pwd)"
MODEL=/home/alberto/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q6_K.gguf
SO="$REPO/build_vulkan/bin/libggml-vulkan.so.0.13.1"
REPS=8
cd "$REPO"
git checkout -- . 2>/dev/null; git stash clear 2>/dev/null; rm -rf .basi; mkdir -p .basi

bench_tg() {  # print the tg64 tokens/sec, or empty on failure
  "$REPO/build_vulkan/bin/llama-bench" -m "$MODEL" -ngl 99 -n 64 -r "$REPS" 2>/dev/null \
    | awk -F'|' '/tg64/{v=$(NF-1); split(v,a,"±"); gsub(/ /,"",a[1]); print a[1]; exit}'
}
build_so() { cmake --build "$REPO/build_vulkan" --target ggml-vulkan -j 8 >/tmp/fanout-build.log 2>&1; }

# ── Phase 1: THEORY GENERATION (LLM, scoped, neutral directive) ──
echo "════════ PHASE: theories ════════"
BASI_MAX_TOOL_ITERS=30 BASI_MAX_ELISION_RESETS=1 "$BASI" -p \
"You are the THEORY phase of a factory. GOAL: find EVERY promising, low-risk change to
speed up LLM token-generation (decode) on the Intel Arc B580 in llama.cpp's VULKAN
backend (ggml/src/ggml-vulkan/ggml-vulkan.cpp). The B580 is detected as INTEL_XE2.
METHOD: find how the code branches on GPU vendor/architecture and does things differently
per GPU; read those sections and judge which choices look suboptimal or overly
conservative for a B580. Find MULTIPLE candidates (aim for 3-6) — do NOT stop at one.

Write ALL your theories to .basi/theories.md with the edit tool. Each theory MUST be one
block in EXACTLY this format (the markers are parsed programmatically — do not alter them):

@@THEORY@@
desc: <one short line: what changes and why it could speed up the B580>
file: ggml/src/ggml-vulkan/ggml-vulkan.cpp
@@SEARCH@@
<the EXACT current lines from the file, copied VERBATIM including indentation>
@@REPLACE@@
<the EXACT replacement lines>
@@END@@

Read the file to copy each SEARCH verbatim (it must match character-for-character). Keep
each SEARCH small (the changed lines plus 1-2 for context). Output as many real,
independent theories as you can find. Do not modify any source file — only write
.basi/theories.md." \
  --tools "read,head,tail,grep,wc,symbols,edit" --yolo > "$SC/fanout-theories.log" 2>&1

# fallback: if the model didn't write theories.md, salvage its final answer
if [ ! -s "$REPO/.basi/theories.md" ]; then
  sed 's/\x1b\[[0-9;]*m//g' "$SC/fanout-theories.log" | grep -a '@@' -A0 >/dev/null 2>&1
  sed 's/\x1b\[[0-9;]*m//g' "$SC/fanout-theories.log" > "$REPO/.basi/theories.md"
fi

# ── Parse ──
N=$(python3 "$SC/parse_theories.py" "$REPO/.basi/theories.md" "$REPO/.basi/parsed")
echo "════════ parsed $N theories ════════"
[ "$N" -ge 1 ] || { echo "no parseable theories; stopping."; exit 1; }
for i in $(seq 1 "$N"); do echo "  T$i: $(cat "$REPO/.basi/parsed/theory-$i.desc")"; done

# theory phase (the only LLM step) is done — free the GPU so benchmarks run on a clean,
# uncontaminated card (no resident 35B skewing tok/s, and full VRAM for the bench model).
echo "── freeing GPU: stopping the llama-server before benchmarks ──"
for p in $(pgrep -x llama-server); do kill "$p" 2>/dev/null; done; sleep 3

# ── Baseline: build clean, benchmark once ──
echo "════════ building + benchmarking BASELINE ════════"
git checkout -- . 2>/dev/null
build_so || { echo "baseline build failed"; exit 1; }
cp -L "$SO" /tmp/vk-baseline.so
BASE_TG=$(bench_tg)
echo "  baseline tg64 = ${BASE_TG:-FAIL} tok/s"

# ── Loop: apply + build + benchmark EACH theory ──
declare -a RES
for i in $(seq 1 "$N"); do
  echo "════════ THEORY $i / $N ════════"
  f=$(cat "$REPO/.basi/parsed/theory-$i.file")
  git checkout -- . 2>/dev/null
  ap=$(python3 "$SC/apply_theory.py" "$REPO/$f" "$REPO/.basi/parsed/theory-$i.search" "$REPO/.basi/parsed/theory-$i.replace" 2>&1)
  echo "  apply: $ap"
  if [[ "$ap" != APPLIED* ]]; then RES[$i]="T$i  SEARCH_NOT_FOUND  (could not apply)"; continue; fi
  if ! build_so; then RES[$i]="T$i  BUILD_FAILED  (did not compile)"; git checkout -- . 2>/dev/null; continue; fi
  cp -L "$SO" "/tmp/vk-theory-$i.so"
  git checkout -- . 2>/dev/null
  TG=$(bench_tg)
  DELTA=$(python3 -c "print(f'{(($TG-$BASE_TG)/$BASE_TG*100):+.2f}%')" 2>/dev/null || echo "?")
  RES[$i]="T$i  tg64=${TG:-FAIL}  ($DELTA vs $BASE_TG)  ::  $(cat "$REPO/.basi/parsed/theory-$i.desc")"
  echo "  ${RES[$i]}"
done

# ── restore baseline lib + tree ──
git checkout -- . 2>/dev/null; cp /tmp/vk-baseline.so "$SO" 2>/dev/null

# ── Overall conclusion (ranked by measured tg) ──
echo ""
echo "════════════════ OVERALL CONCLUSION ════════════════"
echo "baseline tg64 = $BASE_TG tok/s   (Qwen3-4B-Q6, B580, r=$REPS)"
echo "theories tried: $N"
printf '%s\n' "${RES[@]:1}" | sort -t= -k2 -rn
echo "─────────────────────────────────────────────────────"
echo "Rule of thumb: a delta under ~0.3% is within measurement noise (no real win)."
