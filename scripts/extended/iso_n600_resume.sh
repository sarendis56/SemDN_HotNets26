#!/usr/bin/env bash
# Resume the n=600 iso-budget run after interruption: pools are already built, so re-run the
# 5 readers (parallel across the 3 A100s, 3+2 waves) then judge. e4_answer opens preds with
# "w", so a partial reader file is cleanly overwritten. Gates on preds completeness before judging.
set -u
cd "$(dirname "$0")/../.."
G=data/e4
M32=Qwen/Qwen3-32B
TAG=t3b_
ARMS=(chunk chunk_ce page_rel page_seq snippet_full)
GPUS=(0 1 2)
NG=${#GPUS[@]}

reader() {  # $1=arm $2=gpu
  CUDA_VISIBLE_DEVICES="$2" uv run python -m semdn.e4_answer --model "$M32" \
    --pool "$G/pool_iso_${TAG}$1.jsonl" --out "$G/ans_iso_${TAG}$1_q3.jsonl" \
    > "$G/reader_${TAG}$1.log" 2>&1
}

echo "=== readers in parallel across $NG GPUs ==="
i=0
for arm in "${ARMS[@]}"; do
  gpu=${GPUS[$((i % NG))]}
  echo "  launch reader $arm on GPU $gpu"
  reader "$arm" "$gpu" &
  i=$((i + 1))
  (( i % NG == 0 )) && wait
done
wait

echo "=== verify preds complete (6000 lines = 600 q x 10 budgets) ==="
ok=1
for arm in "${ARMS[@]}"; do
  n=$(wc -l < "$G/ans_iso_${TAG}${arm}_q3.preds.jsonl" 2>/dev/null || echo 0)
  echo "  $arm: $n"
  [ "$n" -ge 6000 ] || { ok=0; echo "    !! incomplete ($arm)"; }
done
[ "$ok" = 1 ] || { echo "READERS INCOMPLETE — not judging"; exit 1; }

echo "=== judge Llama-3.1-70B (TP=2, GPU0,1) ==="
PREDS=""; TAGS=""; OUTS=""
for arm in "${ARMS[@]}"; do
  PREDS="$PREDS,$G/ans_iso_${TAG}${arm}_q3.preds.jsonl"
  TAGS="$TAGS,iso_${TAG}${arm}"
  OUTS="$OUTS,$G/judge_iso_${TAG}${arm}_q3.jsonl"
done
bash scripts/lib/judge_robust.sh "${PREDS#,}" "${TAGS#,}" "${OUTS#,}"
echo "=== ISO N600 RESUME DONE ==="
