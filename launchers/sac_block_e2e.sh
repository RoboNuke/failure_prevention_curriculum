#!/usr/bin/env bash
# launchers/sac_block_e2e.sh — full train -> save -> load -> eval smoke test.
#
# Usage:
#   sac_block_e2e.sh <config_path> <experiment_name> [--no_eval]
#
# Reads task / num_envs / num_agents / total_timesteps / eval_timesteps / memory_size
# from runner_cfg in the supplied YAML. Override anything one-off via runner CLI flags
# in the python invocations below.
#
# Flags:
#   --no_eval   Skip the post-training eval pass (still verifies checkpoints exist).
#
# Fail loud, fail fast: any silent miss is a bug, not an expected outcome.
set -Eeuo pipefail
trap 'echo "[launcher] FAILED at ${BASH_SOURCE[0]}:${LINENO} (exit $?)" >&2' ERR

# ===== Args =====
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <config_path> <experiment_name> [--no_eval]" >&2
    echo "  e.g. $0 configs/exp_cfgs/cartpole.yaml cartpole_run1" >&2
    exit 2
fi
CONFIG_PATH="$1"
EXPERIMENT_NAME="$2"
shift 2
RUN_EVAL=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_eval) RUN_EVAL=0 ;;
        *) echo "[launcher] unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ===== Derived paths =====
# Resolve PROJECT_ROOT from the script's own location so this works in any
# clone path (HPC home != local home). LOGDIR follows project root by default;
# override via env var if needed (LOGDIR=... ./launchers/sac_block_e2e.sh ...).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOGDIR="${LOGDIR:-$PROJECT_ROOT/runs}"

RUNNER="$PROJECT_ROOT/learning/runner.py"
EXP_DIR="$LOGDIR/$EXPERIMENT_NAME"
EVAL_EXP_NAME="${EXPERIMENT_NAME}_eval"

# Resolve config to absolute (allow caller to pass a project-root-relative path).
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$PROJECT_ROOT/$CONFIG_PATH"
fi

# ===== Sanity =====
# We assume the caller has already activated the right python env (conda env,
# apptainer shell, venv, etc.) — the launcher does NOT manage environments.
[[ -f "$RUNNER" ]] || { echo "[launcher] runner not found: $RUNNER" >&2; exit 1; }
[[ -f "$CONFIG_PATH" ]] || { echo "[launcher] config not found: $CONFIG_PATH" >&2; exit 1; }
# Resolve python: PYTHON env var (e.g. PYTHON=/isaac-sim/python.sh) wins,
# else fall back to `python` on PATH. Set in your shell or sbatch script to
# point at the container's python wrapper.
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null \
    || { echo "[launcher] python interpreter '$PYTHON' not found — set PYTHON=/path/to/python (e.g. /isaac-sim/python.sh) or put one on PATH" >&2; exit 1; }

# ===== Read num_agents from YAML for the post-train checkpoint check =====
# All other runner_cfg fields (task, num_envs, etc.) flow through to runner.py
# implicitly via --config; only num_agents is needed bash-side to walk per-agent
# checkpoint dirs.
NUM_AGENTS="$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG_PATH'))['runner_cfg']['num_agents'])")"
[[ "$NUM_AGENTS" =~ ^[0-9]+$ ]] \
    || { echo "[launcher] could not read runner_cfg.num_agents from $CONFIG_PATH (got '$NUM_AGENTS')" >&2; exit 1; }

echo "[launcher] python=$(command -v "$PYTHON")  config=$CONFIG_PATH  experiment=$EXPERIMENT_NAME  num_agents=$NUM_AGENTS"

# ===== Train =====
# Ctrl-C (SIGINT, exit 130) is treated as "interrupted, proceed to eval with whatever
# was last flushed to disk". Any other nonzero exit (OOM=137, segfault=139, ValueError
# from runner, etc.) is still a hard failure. The `|| TRAIN_RC=$?` form neutralizes
# `set -e` and the ERR trap for this one command so we can branch on the code.
echo "[launcher] === TRAIN (config=$CONFIG_PATH) ==="
TRAIN_RC=0
"$PYTHON" "$RUNNER" \
    --config "$CONFIG_PATH" \
    --experiment_name "$EXPERIMENT_NAME" \
    --logdir "$LOGDIR" \
    --mode train \
    --headless || TRAIN_RC=$?

case "$TRAIN_RC" in
    0)   echo "[launcher] training completed normally" ;;
    130) echo "[launcher] training interrupted by Ctrl-C (exit 130); proceeding to eval with last saved checkpoints" ;;
    *)   echo "[launcher] training failed with exit $TRAIN_RC (not Ctrl-C); aborting" >&2; exit "$TRAIN_RC" ;;
esac

# ===== Verify checkpoints exist before attempting eval =====
# sac.write_checkpoint writes one file per agent at:
#   $EXP_DIR/<i>/checkpoints/ckpt_<step>.pt   for i in 0..N-1
# If skrl's auto checkpoint_interval ever resolves to "never", training would exit
# 0 with no .pt files written — that's exactly the silent failure we need to catch.
echo "[launcher] verifying per-agent checkpoints under $EXP_DIR"
[[ -d "$EXP_DIR" ]] || { echo "[launcher] experiment dir was not created: $EXP_DIR" >&2; exit 1; }
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    agent_ckpt_dir="$EXP_DIR/$i/checkpoints"
    [[ -d "$agent_ckpt_dir" ]] \
        || { echo "[launcher] missing checkpoint dir for agent $i: $agent_ckpt_dir" >&2; exit 1; }
    if ! compgen -G "$agent_ckpt_dir/ckpt_*.pt" >/dev/null; then
        echo "[launcher] no ckpt_*.pt files for agent $i in $agent_ckpt_dir" >&2
        exit 1
    fi
    latest_for_agent="$(ls -1 "$agent_ckpt_dir"/ckpt_*.pt | tail -1)"
    echo "[launcher]   agent $i: $latest_for_agent"
done

# ===== Eval =====
# Pass the experiment dir as --checkpoint; the runner walks 0/, 1/, ... internally
# and resolves the latest ckpt_<step>.pt per agent (omit --checkpoint_step => latest).
# Use a fresh experiment name for eval so its tensorboard events don't land in the
# training agent dirs (which would mix train + eval scalars on the same plots).
# `--mode eval` makes the runner use runner_cfg.eval_timesteps instead of total_timesteps.
if [[ "$RUN_EVAL" -eq 1 ]]; then
    echo "[launcher] === EVAL (config=$CONFIG_PATH, checkpoint=$EXP_DIR) ==="
    "$PYTHON" "$RUNNER" \
        --config "$CONFIG_PATH" \
        --experiment_name "$EVAL_EXP_NAME" \
        --logdir "$LOGDIR" \
        --checkpoint "$EXP_DIR" \
        --mode eval \
        --headless

    echo "[launcher] done. train=$EXP_DIR  eval=$LOGDIR/$EVAL_EXP_NAME"
else
    echo "[launcher] === EVAL skipped (--no_eval) ==="
    echo "[launcher] done. train=$EXP_DIR"
fi
