#!/usr/bin/env bash
# Render-enabled resilient fetch: same restart-until-cached loop as e4_fetch_resilient.sh,
# but with a headless Chromium fallback and GENEROUS timeouts. Recovers reachable-but-thin
# pages (JS shells / 403 bot-blocks) that the light fetch missed. Entirely offline — uses NO
# Serper credits. Network-level connection failures stay blocked (render shares the host network).
set -u
cd "$(dirname "$0")/../.."
CACHE=data/e4/pages_cache.jsonl

# generous render budgets (ms / s) — slower but higher recovery
export E1_GOTO_MS=25000 E1_IDLE_MS=6000 E1_INNER_S=12 E1_BUDGET_S=40

TOTAL=$(uv run python -c "
import json
s=[json.loads(l) for l in open('data/e4/serp_deep.jsonl')]
u={r['url'] for rec in s for r in rec['results']}
print(len(u))")
echo "target unique urls: $TOTAL (render on, generous timeouts)"

for i in $(seq 1 90); do
  before=$(wc -l < "$CACHE" 2>/dev/null || echo 0)
  if [ "$before" -ge "$TOTAL" ]; then echo "all $TOTAL urls cached"; break; fi
  echo "--- attempt $i: $before/$TOTAL cached ---"
  uv run python -m semdn.e4_granularity --stage fetch --render --fetch-conc 8 --fetch-timeout 25 \
    >> data/e4/logs/e4g_fetch_render.log 2>&1
  after=$(wc -l < "$CACHE" 2>/dev/null || echo 0)
  if [ "$after" -le "$before" ]; then
    echo "no progress at $after -> marking first un-cached url blocked (crasher)"
    uv run python -c "
import json
cache={json.loads(l)['url'] for l in open('$CACHE')}
s=[json.loads(l) for l in open('data/e4/serp_deep.jsonl')]
for rec in s:
    for r in rec['results']:
        if r['url'] not in cache:
            with open('$CACHE','a') as f:
                f.write(json.dumps({'url':r['url'],'ok':False,'method':'blocked',
                    'status':'crasher','n_chars':0,'text':''})+chr(10))
            print('blocked crasher:', r['url']); raise SystemExit
"
  fi
done
echo "=== RENDER FETCH DONE: $(wc -l < $CACHE)/$TOTAL cached ==="
uv run python -c "
import json
rows=[json.loads(l) for l in open('$CACHE')]
ok=sum(r['ok'] for r in rows)
rend=sum(1 for r in rows if r.get('method')=='render')
print(f'usable {ok}/{len(rows)} ({ok/len(rows):.0%}); recovered via render: {rend}')"
