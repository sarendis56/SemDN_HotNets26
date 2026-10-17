#!/usr/bin/env bash
# Serial GPU queue — runs the pending GPU experiments ONE BY ONE (never two at once, since
# each needs both GPUs for the Llama-70B judge). Launch this as a single background job once
# the current GPU job (cf_compare + summary_v2) has finished.
set -u
cd "$(dirname "$0")/../.."

echo "########## QUEUE: nochunk_extra (full-page CRAG + live snippet/full-page) ##########"
bash scripts/extended/e4_nochunk_extra.sh

echo "########## QUEUE: wait for the n=600 page extraction to finish ##########"
until grep -q "cache (t3b/coherent) done" data/e4/t3b_cache.log 2>/dev/null; do sleep 60; done

echo "########## QUEUE: n=600 comprehensive ##########"
bash scripts/extended/e4_n600.sh

echo "########## QUEUE: Cloudflare aug re-run (context-overflow fix) ##########"
bash scripts/extended/e4_cf_aug_fix.sh

echo "########## GPU QUEUE DONE ##########"
