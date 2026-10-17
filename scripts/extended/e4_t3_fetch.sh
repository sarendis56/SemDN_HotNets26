#!/usr/bin/env bash
# Download + extract CRAG Task-3 (50 pages/question web corpus, 7.7GB bz2), Xet
# disabled. Then peek at the schema.
set -u
cd "$(dirname "$0")/../.."
D=data/datasets/crag
echo "=== downloading crag_task_3_dev_v4.tar.bz2 (7.7GB) ==="
HF_HUB_DISABLE_XET=1 hf download Rickyoung0221/crag crag_task_3_dev_v4.tar.bz2 \
  --repo-type dataset --local-dir "$D"
echo "=== extracting (bz2 -> ~tens of GB jsonl) ==="
tar -xjf "$D/crag_task_3_dev_v4.tar.bz2" -C "$D"
echo "=== extracted ==="
ls -la "$D"/*task_3* 2>/dev/null
F=$(ls "$D"/crag_task_3_dev_v4.jsonl 2>/dev/null || ls "$D"/*task_3*.jsonl 2>/dev/null | head -1)
echo "inner jsonl: $F"
uv run python -c "
import json
with open('$F') as fh:
    r=json.loads(fh.readline())
print('fields:', [k for k in r if k!='search_results'])
print('n search_results (pages):', len(r.get('search_results') or []))
print('qid:', r.get('interaction_id'))
print('page_result html chars (1st):', len((r.get('search_results') or [{}])[0].get('page_result','') or ''))
"
echo "=== T3 FETCH DONE ==="
