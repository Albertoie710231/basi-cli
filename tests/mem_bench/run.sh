#!/usr/bin/env bash
# Phase 4 A/B benchmark: compare context-compaction strategies on fact recall.
#
# Builds ONE realistic session (real DeepSeek-V4 paper text) with unique facts
# planted at spread positions, runs it under each BASI_COMPACT mode, and grades
# recall by grepping the model's output for each fact token. With the getline
# stdin path, piped input is NOT echoed, so a token in the output was *generated*
# by the model — i.e. recalled.
#
# Usage: tests/mem_bench/run.sh [chat_model.gguf] [embed_model.gguf]
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/basi-cli"
SRC="/home/alberto/Documentos/GPT/standard_gpt/deepseek_database/dsv4_extracted.txt"
CTX=8192
SEED=42

CHAT="${1:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q6_K.gguf}"
EMB="${2:-$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-small-retrieval-GGUF/snapshots/78b0ebcb4c870fdfef409e578b65288b49a4fa90/v5-small-retrieval-Q4_K_M.gguf}"

WORK="$(mktemp -d)"
IN="$WORK/session.txt"
RESULTS="$ROOT/tests/mem_bench/RESULTS.md"

# Facts planted at chunk positions 1/3/5/7 (F1 earliest ... F4 latest). DISTINCT
# topics + unique tokens, so embedding retrieval can disambiguate them (the v1
# near-identical "vault N" facts were adversarial for retrieval, favourable for
# summary). All end up compacted before the probes.
POS=(1 3 5 7)
FACT_SENT=(
  "the lead architect of the CSA attention module is engineer ZORRILLA_7741"
  "the internal training cluster is codenamed BLUEFIN_9931"
  "the ablation study deadline is set to milestone GLACIER_0414"
  "the total pretraining compute budget is recorded as TUNGSTEN_4271 GPU-hours"
)
PROBE_Q=(
  "Earlier I named the lead architect of the CSA attention module. Who is it? Answer with only the name token."
  "Earlier I gave the internal training cluster's codename. What is it? Answer with only the codename token."
  "Earlier I gave the ablation study deadline milestone. What is it? Answer with only the milestone token."
  "Earlier I gave the total pretraining compute budget. What is it? Answer with only the value token."
)
TOKEN=(ZORRILLA_7741 BLUEFIN_9931 GLACIER_0414 TUNGSTEN_4271)

# --- build the session: 8 real paper chunks (~1500 tok), facts at 1,3,5,7 ---
tr '\n' ' ' < "$SRC" | tr -cd '\40-\176' | tr -s ' ' | fold -w 6000 -s > "$WORK/chunks.txt"
: > "$IN"
n=0; fi=0
while IFS= read -r chunk && [ $n -lt 8 ]; do
  n=$((n+1))
  line="Paper part $n (read and remember, reply only OK): $chunk"
  if [ "$n" = "${POS[$fi]:-x}" ]; then
    line="$line  IMPORTANT SESSION FACT: ${FACT_SENT[$fi]} — remember it exactly."
    fi=$((fi+1))
  fi
  echo "$line Reply with only OK." >> "$IN"
done < "$WORK/chunks.txt"
# probes (after all content -> everything early is compacted)
for i in 0 1 2 3; do echo "${PROBE_Q[$i]}" >> "$IN"; done
echo "In one short sentence, what is the paper I shared about?" >> "$IN"

echo "Session: $(wc -l < "$IN") turns (8 content incl. 4 facts, 4 fact-probes, 1 gist-probe)"

# --- run each mode, grade ---
{
  echo "# Phase 4 A/B — compaction strategy recall"
  echo
  echo "Model: \`$(basename "$CHAT")\` · Embedder: \`$(basename "$EMB")\` · ctx=$CTX · seed=$SEED"
  echo "Facts planted at chunks 1/3/5/7 (F1 earliest → F4 latest); probed after all 8 chunks."
  echo "Recall = fact token appears in the model's answer (input is not echoed)."
  echo
  echo "| mode | F1 | F2 | F3 | F4 | recall | gist | compactions | wall | crash |"
  echo "|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"
} > "$RESULTS"

for MODE in off summary retrieve hybrid; do
  RUN=$(mktemp -d)
  OUT="$WORK/out_$MODE.txt"
  BASI_EMBED_MODEL="$EMB" BASI_COMPACT="$MODE" \
    timeout 600 env -C "$RUN" "$BIN" -m "$CHAT" -c "$CTX" --seed "$SEED" --yolo \
    < "$IN" > "$OUT" 2>&1
  S="$(sed 's/\x1b\[[0-9;]*m//g' "$OUT")"

  row="| $MODE |"; hit=0
  for i in 0 1 2 3; do
    tok="${TOKEN[$i]}"
    if printf '%s' "$S" | grep -q "$tok"; then row="$row ✅ |"; hit=$((hit+1)); else row="$row ❌ |"; fi
  done
  comp=$(printf '%s\n' "$S" | grep -c 'Compacted:')
  wall=$(printf '%s\n' "$S" | grep -c 'Context limit reached')
  crash=$(grep -aciE 'DeviceLost|volcado|core dump' "$OUT")
  gist=$(printf '%s' "$S" | grep -ciE 'deepseek|long.context|million.token|mixture.of.experts|MoE' )
  gmark=$([ "$gist" -gt 0 ] && echo "✅" || echo "❌")
  cmark=$([ "$crash" -gt 0 ] && echo "💥" || echo "—")
  echo "$row $hit/4 | $gmark | $comp | $wall | $cmark |" >> "$RESULTS"
  echo "  $MODE: recall $hit/4, compactions=$comp, wall=$wall, crash=$crash"
done

echo
echo "=== RESULTS ($RESULTS) ==="
cat "$RESULTS"
