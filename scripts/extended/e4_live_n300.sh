#!/usr/bin/env bash
# n=300 LIVE-WEB RETEST. Expand the 147-question live set toward 300 (credit-capped SERP, no
# Serper overspend), rebuild all 7 live arms, read with Qwen3-32B, judge with Llama-70B.
# Every stage is resumable (serp/fetch skip cached; select/rerank append; readers/judge overwrite).
set -u
cd "$(dirname "$0")/../.."
source scripts/lib/gpus.sh   # on_gpus: pick free GPUs at launch (shared machine)
export HF_HUB_OFFLINE=1
M32=Qwen/Qwen3-32B
G=data/e4
N=300

echo "########## PHASE 1: SERP (DONE — 287q/842cr already spent; cap=1 prevents further spend) ##########"
uv run python -m semdn.e4_granularity --stage serp --n $N --depth 50 --credit-cap 1
uv run python -c "
import json
r=[json.loads(l) for l in open('$G/serp_deep.jsonl')]
print(f'  serp_deep now {len(r)} questions; session-new credits (beyond first 147): {sum(x.get(\"credits\",0) for x in r[147:])}')"

echo "########## PHASE 2: RENDER page fetch (headless Chromium + generous timeouts; offline, no credits) ##########"
bash scripts/extended/e4_fetch_render.sh

echo "########## PHASE 3: ranking builds (GPU) ##########"
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder e5      --chunker structnt --n $N
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder e5      --chunker structnt --hybrid --n $N
on_gpus 1 uv run python -m semdn.e4_granularity --stage rerank                   --chunker structnt --n $N
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder qwen3   --chunker structnt --n $N
on_gpus 1 uv run python -m semdn.e4_granularity --stage select --encoder qwen3rr --chunker coherent  --n $N
uv run python -m semdn.e4_live_baselines --mode snippet
uv run python -m semdn.e4_live_baselines --mode fullpage

echo "########## PHASE 4: readers (Qwen3-32B) ##########"
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_structnt.jsonl         --out $G/granularity_quality_structnt_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_structnt_hybrid.jsonl  --out $G/gq_live_hybrid_structnt_e5_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_ce_structnt.jsonl      --out $G/granularity_quality_ce_structnt_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_qwen3_structnt.jsonl   --out $G/granularity_quality_qwen3_structnt_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_qwen3rr_coherent.jsonl --out $G/gq_live_coherent_qwen3rr_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_live_snippet.jsonl     --out $G/gq_pool_live_snippet_q3.jsonl
on_gpus 1 uv run python -m semdn.e4_answer --model "$M32" --pool $G/pool_live_fullpage.jsonl    --out $G/gq_pool_live_fullpage_q3.jsonl

echo "########## PHASE 5: judge (Llama-70B), all 7 arms in one load ##########"
bash scripts/lib/judge_robust.sh \
  "$G/granularity_quality_structnt_q3.preds.jsonl,$G/gq_live_hybrid_structnt_e5_q3.preds.jsonl,$G/granularity_quality_ce_structnt_q3.preds.jsonl,$G/granularity_quality_qwen3_structnt_q3.preds.jsonl,$G/gq_live_coherent_qwen3rr_q3.preds.jsonl,$G/gq_pool_live_snippet_q3.preds.jsonl,$G/gq_pool_live_fullpage_q3.preds.jsonl" \
  "structnt,hybrid,ce-structnt,qwen3-structnt,coh-qwen3rr,snippet,fullpage" \
  "$G/judge_q3_structnt.jsonl,$G/judge_live_hybrid_q3.jsonl,$G/judge_cestructnt_q3.jsonl,$G/judge_qwen3structnt_q3.jsonl,$G/judge_live_coherent_qwen3rr_q3.jsonl,$G/judge_pool_live_snippet_q3.jsonl,$G/judge_pool_live_fullpage_q3.jsonl"
echo "########## N300 LIVE RETEST DONE ##########"
