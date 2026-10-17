#!/usr/bin/env bash
# Print the index of the first GPU that is sustained-idle (util<10%, mem<2GiB for
# N consecutive 60s checks). Blocks until one is free. Used to gate GPU jobs.
set -u
need=${1:-3}
while true; do
    for g in 0 1; do
        ok=1
        for _ in $(seq 1 "$need"); do
            line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                   --format=csv,noheader,nounits -i "$g" 2>/dev/null)
            util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
            mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
            if [ -z "$util" ] || [ "$util" -ge 10 ] || [ "$mem" -ge 2048 ]; then
                ok=0; break
            fi
            sleep 60
        done
        if [ "$ok" -eq 1 ]; then echo "$g"; exit 0; fi
    done
done
