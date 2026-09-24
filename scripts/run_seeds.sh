#!/usr/bin/env bash
# Seed sweep: run every (target, seed) notebook one after another on one GPU.
#
# Each run writes to results/<target>/seed<N>/: checkpoint.pkl, log.txt,
# figures/, stdout.log (the script's full output) and one <run_id>/metrics.json
# per evaluated sample set. A run that already finished (exit 0, recorded in
# results/<target>/seeds_done.txt) is skipped, so re-launching the sweep after
# an interruption picks up where it stopped. A failed run is logged and the
# sweep moves on to the next one.
#
# Each run is started in its own session (process group). When it ends, the
# whole group is terminated, so no leftover worker (e.g. the Geometric-W2
# process pool) survives into the next run, and the next run starts only once
# the group is gone. Before each run the sweep waits until the GPU has no more
# than MAX_FOREIGN_MB in use, so it does not start into a GPU another job holds.
#
# Usage (from the repo root) — detached, survives closing the editor or
# disconnecting:
#   setsid nohup scripts/run_seeds.sh > results/sweep.log 2>&1 < /dev/null &
# Stop the whole sweep (the current run included):
#   kill -TERM -- -$(cat results/sweep.pgid)
#
# Overrides (environment variables):
#   TARGETS="gmm dw4 gmm50 lj13"   targets, in run order
#   SEEDS="0 1 2"                  seeds per target
#   GPU=1                          physical GPU index (CUDA_VISIBLE_DEVICES)
#   PYTHON=python                  interpreter with progressive_is_tempering installed
#   FORCE=1                        re-run seeds already recorded as done
#   MAX_FOREIGN_MB=2000            wait while the GPU has more than this in use
#   MEM_FRACTION=0.95              share of the GPU JAX preallocates per run

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS="${TARGETS:-gmm dw4 gmm50 lj13}"
SEEDS="${SEEDS:-0 1 2}"
GPU="${GPU:-1}"
PYTHON="${PYTHON:-python}"
FORCE="${FORCE:-0}"
MAX_FOREIGN_MB="${MAX_FOREIGN_MB:-2000}"
# lj13's train_step asks for one 15.33 GiB scratch buffer (8 rungs x 512 rows
# through the 169-edge EGNN backward pass), which does not fit in JAX's default
# pool of 75% of the card once params, optimiser state and buffers are in it.
MEM_FRACTION="${MEM_FRACTION:-0.95}"

# The sweep's own process group: `kill -TERM -- -<pgid>` stops everything.
mkdir -p "$ROOT/results"
ps -o pgid= $$ | tr -d ' ' > "$ROOT/results/sweep.pgid"
trap '' HUP   # a closed terminal or dropped connection does not stop the sweep

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

gpu_used_mb() {
    nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' || echo 0
}

wait_for_gpu() {
    local used waited=0
    while used=$(gpu_used_mb); (( used > MAX_FOREIGN_MB )); do
        if (( waited % 600 == 0 )); then
            echo "[$(stamp)] GPU $GPU has ${used} MiB in use by other jobs; waiting"
        fi
        sleep 30; waited=$(( waited + 30 ))
    done
}

# Terminate every process left in run group $1, then wait until it is empty.
reap_group() {
    local pgid=$1
    kill -TERM -- "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -g "$pgid" > /dev/null || return 0
        sleep 1
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
    while pgrep -g "$pgid" > /dev/null; do sleep 1; done
}

# Stopping the sweep stops the current run too: it lives in its own group.
run_pid=""
on_stop() {
    echo "[$(stamp)] sweep stopped by signal"
    [[ -n "$run_pid" ]] && reap_group "$run_pid"
    exit 143
}
trap on_stop TERM INT

echo "[$(stamp)] sweep start: targets=($TARGETS) seeds=($SEEDS) gpu=$GPU python=$PYTHON"

for target in $TARGETS; do
    nb_dir="$ROOT/notebooks/$target"
    if [[ ! -f "$nb_dir/$target.py" ]]; then
        echo "[$(stamp)] $target: no notebooks/$target/$target.py, skipping"
        continue
    fi
    done_file="$ROOT/results/$target/seeds_done.txt"
    mkdir -p "$ROOT/results/$target"
    touch "$done_file"

    for seed in $SEEDS; do
        run="$target seed$seed"
        if [[ "$FORCE" != "1" ]] && grep -qx "$run exit=0" "$done_file"; then
            echo "[$(stamp)] $run: already done, skipping"
            continue
        fi
        out_dir="$ROOT/results/$target/seed$seed"
        mkdir -p "$out_dir"
        wait_for_gpu
        echo "[$(stamp)] $run: start"
        t0=$SECONDS
        # setsid: the run gets its own process group, reaped as a whole below.
        # JAX preallocation is forced on. wait_for_gpu already holds each run
        # until the GPU is nearly empty, and without one contiguous pool the
        # allocator cannot serve lj13's 15.33 GiB train_step scratch (8 rungs x
        # 512 rows through the 169-edge EGNN backward pass) — every lj13 seed
        # died on its first gradient step with RESOURCE_EXHAUSTED. It must be set
        # explicitly: the notebooks setdefault it to "false" when it is unset.
        ( cd "$nb_dir" && exec setsid env \
            CUDA_VISIBLE_DEVICES="$GPU" RUN_SEED="$seed" MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
            XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION" \
            "$PYTHON" "$target.py" ) > "$out_dir/stdout.log" 2>&1 < /dev/null &
        run_pid=$!
        wait "$run_pid"
        code=$?
        reap_group "$run_pid"   # setsid makes the run's pid its group id
        run_pid=""
        # Keep only the latest outcome per run, so a later success replaces a failure.
        grep -vx "$run exit=[0-9]*" "$done_file" > "$done_file.tmp" || true
        echo "$run exit=$code" >> "$done_file.tmp"
        mv "$done_file.tmp" "$done_file"
        echo "[$(stamp)] $run: exit=$code after $(( (SECONDS - t0) / 60 )) min"
    done
done

echo "[$(stamp)] sweep end"
