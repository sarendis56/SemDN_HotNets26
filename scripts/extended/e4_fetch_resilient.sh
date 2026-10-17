#!/usr/bin/env bash
# Resilient E4 fetch: light_fetch via trafilatura/lxml segfaults (~free(): invalid
# pointer) every ~200 urls under thread concurrency. The fetch stage is resumable
# (every attempted url is cached), so we just restart it until all target urls are
# cached. Stuck-breaker: if a restart makes ZERO progress, the first un-cached url
# is a deterministic crasher -> mark it blocked and move on, so we never spin.
set -u
cd "$(dirname "$0")/../.."
CACHE=data/e4/pages_cache.jsonl

TOTAL=$(uv run python -c "
import json
s=[json.loads(l) for l in open('data/e4/serp_deep.jsonl')]
u={r['url'] for rec in s for r in rec['results']}
print(len(u))")
echo "target unique urls: $TOTAL"

for i in $(seq 1 60); do
  before=$(wc -l < "$CACHE" 2>/dev/null || echo 0)
  if [ "$before" -ge "$TOTAL" ]; then echo "all $TOTAL urls cached"; break; fi
  echo "--- attempt $i: $before/$TOTAL cached ---"
  uv run python -m semdn.e4_granularity --stage fetch --fetch-conc 8 --fetch-timeout 8 \
    >> data/e4/logs/e4g_fetch_loop.log 2>&1
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
echo "=== RESILIENT FETCH DONE: $(wc -l < $CACHE)/$TOTAL cached ==="
uv run python -c "
import json
rows=[json.loads(l) for l in open('$CACHE')]
ok=sum(r['ok'] for r in rows)
print(f'usable {ok}/{len(rows)} ({ok/len(rows):.0%})')"
