#!/usr/bin/env bash
# Compare retrieval SCORING methods (dense / bm25 / hybrid) on two fact regimes:
#   distinct  — different topics + unique tokens (dense already does well here)
#   neardup   — near-identical "vault N" facts (dense FAILS: embeds them alike)
# Hypothesis under test: BM25/hybrid help on neardup (exact keyword "one" vs "two"),
# but may NOT beat dense on distinct in this local small-model / small-corpus regime.
# Recall = fact token appears in the model's answer (piped input isn't echoed).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/basi-cli"
SRC="/home/alberto/Documentos/GPT/standard_gpt/deepseek_database/dsv4_extracted.txt"
CTX=8192; SEED=42
CHAT="${1:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q6_K.gguf}"
EMB="${2:-$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-small-retrieval-GGUF/snapshots/78b0ebcb4c870fdfef409e578b65288b49a4fa90/v5-small-retrieval-Q4_K_M.gguf}"
WORK="$(mktemp -d)"
RESULTS="$ROOT/tests/mem_bench/SCORE_RESULTS.md"

tr '\n' ' ' < "$SRC" | tr -cd '\40-\176' | tr -s ' ' | fold -w 6000 -s > "$WORK/chunks.txt"

# build a session file for a given fact regime into $1
build_session() {
  local regime="$1" out="$2"
  local -a sent q tok
  if [ "$regime" = "distinct" ]; then
    sent=("the lead architect of the CSA attention module is engineer ZORRILLA_7741"
          "the internal training cluster is codenamed BLUEFIN_9931"
          "the ablation study deadline is set to milestone GLACIER_0414"
          "the total pretraining compute budget is recorded as TUNGSTEN_4271 GPU-hours")
    q=("Earlier I named the lead architect of the CSA attention module. Who is it? Answer with only the name token."
       "Earlier I gave the internal training cluster's codename. What is it? Answer with only the codename token."
       "Earlier I gave the ablation study deadline milestone. What is it? Answer with only the milestone token."
       "Earlier I gave the total pretraining compute budget. What is it? Answer with only the value token.")
    tok=(ZORRILLA_7741 BLUEFIN_9931 GLACIER_0414 TUNGSTEN_4271)
  else  # neardup
    sent=("the access code for vault one is KESTREL_4417"
          "the access code for vault two is MARMOT_8823"
          "the access code for vault three is AXOLOTL_5591"
          "the access code for vault four is NARWHAL_7234")
    q=("What is the access code for vault one that I gave earlier? Answer with only the code token."
       "What is the access code for vault two that I gave earlier? Answer with only the code token."
       "What is the access code for vault three that I gave earlier? Answer with only the code token."
       "What is the access code for vault four that I gave earlier? Answer with only the code token.")
    tok=(KESTREL_4417 MARMOT_8823 AXOLOTL_5591 NARWHAL_7234)
  fi
  : > "$out"
  local n=0 fi=0
  local -a POS=(1 3 5 7)
  while IFS= read -r chunk && [ $n -lt 8 ]; do
    n=$((n+1))
    local line="Paper part $n (read and remember, reply only OK): $chunk"
    if [ "$n" = "${POS[$fi]:-x}" ]; then line="$line  IMPORTANT SESSION FACT: ${sent[$fi]} — remember it exactly."; fi=$((fi+1)); fi
    echo "$line Reply with only OK." >> "$out"
  done < "$WORK/chunks.txt"
  for i in 0 1 2 3; do echo "${q[$i]}" >> "$out"; done
  printf '%s\n' "${tok[@]}" > "$out.tok"
}

{
  echo "# Retrieval scoring comparison (dense vs bm25 vs hybrid)"
  echo
  echo "Model: \`$(basename "$CHAT")\` · Embedder: \`$(basename "$EMB")\` · ctx=$CTX · seed=$SEED · mode=retrieve"
  echo "Recall over 4 facts planted at chunks 1/3/5/7, probed after all are compacted."
  echo
  echo "| fact regime | score | F1 | F2 | F3 | F4 | recall |"
  echo "|---|---|:--:|:--:|:--:|:--:|:--:|"
} > "$RESULTS"

for REGIME in distinct neardup; do
  IN="$WORK/$REGIME.txt"; build_session "$REGIME" "$IN"
  mapfile -t TOK < "$IN.tok"
  for SCORE in dense bm25 hybrid; do
    RUN=$(mktemp -d); OUT="$WORK/out_${REGIME}_${SCORE}.txt"
    BASI_EMBED_MODEL="$EMB" BASI_COMPACT=retrieve BASI_RETRIEVE_SCORE="$SCORE" \
      timeout 600 env -C "$RUN" "$BIN" -m "$CHAT" -c "$CTX" --seed "$SEED" --yolo \
      < "$IN" > "$OUT" 2>&1
    S="$(sed 's/\x1b\[[0-9;]*m//g' "$OUT")"
    row="| $REGIME | $SCORE |"; hit=0
    for i in 0 1 2 3; do
      if printf '%s' "$S" | grep -q "${TOK[$i]}"; then row="$row ✅ |"; hit=$((hit+1)); else row="$row ❌ |"; fi
    done
    echo "$row $hit/4 |" >> "$RESULTS"
    echo "  $REGIME/$SCORE: $hit/4"
  done
done

echo; echo "=== $RESULTS ==="; cat "$RESULTS"
