#!/usr/bin/env bash
# Block until GPU $1 is sustained-idle: util < 10% and mem < 2 GiB for
# $2 (default 5) consecutive 60 s checks. Exits 0 when idle.
set -u
gpu=${1:?usage: wait_gpu.sh <gpu-index> [consecutive-checks]}
need=${2:-5}
ok=0
until [ "$ok" -ge "$need" ]; do
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$gpu")
    util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
    mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
    if [ "$util" -lt 10 ] && [ "$mem" -lt 2048 ]; then
        ok=$((ok + 1))
    else
        ok=0
    fi
    sleep 60
done
echo "GPU $gpu sustained idle ($need consecutive checks)"
