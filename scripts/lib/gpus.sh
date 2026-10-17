# Shared-machine GPU launcher. Source it, then run a GPU stage with:
#   on_gpus N [MIN_FREE_MIB] cmd args...        e.g.  on_gpus 2 78000 uv run python -m ...
# Runs cmd on the N GPUs with the most free memory, each with >= MIN_FREE MiB free (default
# 74000: what vLLM at gpu_memory_utilization 0.90 needs on an 80GB card). Returns 1
# without running cmd if not enough GPUs are free. GPUS=0,1,2 restricts the candidates.
# Picked at each launch, so a long pipeline re-checks before every stage.
# ON_GPUS_WAIT=1: instead of failing, re-check every 5 minutes until enough GPUs are free.
# ON_GPUS_IDLE=1: also require < 10% compute use (skip cards another user is computing on).
# ponytail: free memory is a snapshot -- another user can grab the card between pick and load.
on_gpus() {
  local n=$1 min=74000 picked
  shift
  if [[ ${1:-} =~ ^[0-9]+$ ]]; then min=$1; shift; fi
  while :; do
    picked=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', ' -v allow=",${GPUS:-}," -v min="$min" -v idle="${ON_GPUS_IDLE:-0}" \
          '(allow == ",," || index(allow, "," $1 ",")) && $2 >= min && (idle != 1 || $3 < 10) {print $2, $1}' \
      | sort -rn | head -n "$n" | awk '{print $2}' | paste -sd, -)
    [ "$(echo "$picked" | tr ',' '\n' | grep -c .)" -ge "$n" ] || [ "${ON_GPUS_WAIT:-0}" != 1 ] && break
    echo "on_gpus: waiting for $n GPU(s) with >=${min} MiB free ($(date +%H:%M))" >&2
    sleep 300
  done
  if [ "$(echo "$picked" | tr ',' '\n' | grep -c .)" -lt "$n" ]; then
    echo "on_gpus: need $n GPU(s) with >=${min} MiB free${GPUS:+ among $GPUS}; got '${picked}'" >&2
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv >&2
    return 1
  fi
  echo "on_gpus: GPU(s) $picked -> $*" >&2
  # PCI_BUS_ID: make CUDA number GPUs like nvidia-smi (default FASTEST_FIRST may differ on mixed boxes)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$picked "$@"
}

# Kill our own processes started with SCDN_RUN_TAG=<tag> (children inherit it). Use this,
# never pkill/pgrep-by-name: other jobs on the shared box may run the same program.
kill_tagged() {
  local p
  for p in $(pgrep -u "$(id -u)" .); do
    tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | grep -qx "SCDN_RUN_TAG=$1" && kill -9 "$p" 2>/dev/null
  done
  return 0
}
