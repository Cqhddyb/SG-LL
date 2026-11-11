#!/usr/bin/env bash
# Evaluate one inference-only Route A delta suffix-residual scale on two suites.

set -euo pipefail

REPO_ROOT="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${REPO_ROOT}"

OPENPI_PYTHON="${OPENPI_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
FUSION_SCALE="${FUSION_SCALE:?FUSION_SCALE must be explicitly set}"
GPU="${GPU:?GPU must be set}"
PORT_BASE="${PORT_BASE:?PORT_BASE must be set}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR must name the immutable Route A checkpoint}"
RUN_PREFIX="${RUN_PREFIX:?RUN_PREFIX must be a fresh immutable output prefix}"
LIBERO_ROOT="${LIBERO_ROOT:?LIBERO_ROOT must point to LIBERO-Plus}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero_plus_geo}"
FUTURE_OFFSET="${FUTURE_OFFSET:-10}"
EVAL_SEED="${EVAL_SEED:-7}"
EVAL_WORKERS="${EVAL_WORKERS:-3}"
RESULT_ROOT="${RESULT_ROOT:-data/libero/results}"
LOG_ROOT="${LOG_ROOT:-logs/eval}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-}"
DIAGNOSTICS_ENABLED="${DIAGNOSTICS_ENABLED:-0}"
FUSION_SCALE_SUITES="${FUSION_SCALE_SUITES:-libero_spatial,libero_10}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${POLICY_DIR}/assets/Sylvest/libero_plus_lerobot/norm_stats.json}"
CLASSIFICATION_JSON="${CLASSIFICATION_JSON:-${LIBERO_ROOT}/libero/libero/benchmark/task_classification.json}"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "${OPENPI_PYTHON}" ]] || { echo "OPENPI_PYTHON is not executable: ${OPENPI_PYTHON}" >&2; exit 2; }
[[ "${GPU}" =~ ^[0-9]+$ && "${PORT_BASE}" =~ ^[0-9]+$ ]] || { echo "GPU and PORT_BASE must be non-negative integers" >&2; exit 2; }
[[ "${EVAL_SEED}" =~ ^[0-9]+$ && "${EVAL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid EVAL_SEED or EVAL_WORKERS" >&2; exit 2; }
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${DIAGNOSTICS_ENABLED}" == "0" || "${DIAGNOSTICS_ENABLED}" == "1" ]] || { echo "DIAGNOSTICS_ENABLED must be 0 or 1" >&2; exit 2; }
if [[ "${DIAGNOSTICS_ENABLED}" == "1" ]]; then
  [[ -n "${DIAGNOSTICS_ROOT}" ]] || { echo "DIAGNOSTICS_ROOT is required when diagnostics are enabled" >&2; exit 2; }
elif [[ -n "${DIAGNOSTICS_ROOT}" ]]; then
  echo "DIAGNOSTICS_ROOT requires DIAGNOSTICS_ENABLED=1" >&2
  exit 2
fi
"${OPENPI_PYTHON}" - "${FUSION_SCALE}" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or value < 0.0:
    raise SystemExit("FUSION_SCALE must be finite and non-negative")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Route A fusion-scale dry-run: alpha=${FUSION_SCALE} suites=${FUSION_SCALE_SUITES} diagnostics=${DIAGNOSTICS_ENABLED}"
  exit 0
fi

for required_path in "${POLICY_DIR}/model.safetensors" "${NORM_STATS_PATH}" "${CLASSIFICATION_JSON}"; do
  [[ -f "${required_path}" ]] || { echo "Missing required file: ${required_path}" >&2; exit 3; }
done
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
if [[ "${DIAGNOSTICS_ENABLED}" == "1" ]]; then
  mkdir -p "${DIAGNOSTICS_ROOT}"
fi

suite_tasks() {
  case "$1" in
    libero_spatial) printf '%s\n' 2402 ;;
    libero_10) printf '%s\n' 2519 ;;
    *) echo "Unsupported Hypothesis-C suite: $1" >&2; return 1 ;;
  esac
}

IFS=',' read -r -a requested_suites <<<"${FUSION_SCALE_SUITES}"
(( ${#requested_suites[@]} > 0 )) || {
  echo "FUSION_SCALE_SUITES must contain at least one suite" >&2
  exit 2
}
declare -A seen_suites=()
for suite in "${requested_suites[@]}"; do
  [[ -n "${suite}" ]] || {
    echo "FUSION_SCALE_SUITES must not contain an empty suite" >&2
    exit 2
  }
  suite_tasks "${suite}" >/dev/null
  [[ -z "${seen_suites[${suite}]+x}" ]] || {
    echo "FUSION_SCALE_SUITES must not repeat ${suite}" >&2
    exit 2
  }
  seen_suites["${suite}"]=1
done

run_suite() {
  local suite="$1"
  local port="$2"
  local num_tasks
  num_tasks="$(suite_tasks "${suite}")"
  local run_name="${RUN_PREFIX}_${suite}_full"
  local result_dir="${RESULT_ROOT}/${run_name}"
  local summary_csv="${RESULT_ROOT}/${run_name}_summary.csv"
  local pipeline_log="${LOG_ROOT}/${run_name}_pipeline.log"
  local diagnostics_dir="${DIAGNOSTICS_ROOT}/${run_name}"
  [[ ! -e "${result_dir}" && ! -e "${summary_csv}" ]] || {
    echo "Refusing to overwrite immutable fusion-scale output: ${run_name}" >&2
    exit 4
  }
  if [[ "${DIAGNOSTICS_ENABLED}" == "1" && -e "${diagnostics_dir}" ]]; then
    echo "Refusing to overwrite immutable diagnostics output: ${diagnostics_dir}" >&2
    exit 4
  fi
  local -a diagnostics_args=()
  if [[ "${DIAGNOSTICS_ENABLED}" == "1" ]]; then
    diagnostics_args=(
      --policy-scheme4-inference-diagnostics-enabled true
      --diagnostics-dir "${DIAGNOSTICS_ROOT}"
    )
  fi
  "${OPENPI_PYTHON}" examples/libero/run_parallel_eval.py \
    --run-name "${run_name}" \
    --task-suite-name "${suite}" \
    --num-tasks "${num_tasks}" \
    --num-workers "${EVAL_WORKERS}" \
    --num-trials-per-task 1 \
    --seed "${EVAL_SEED}" \
    --gpu "${GPU}" \
    --port "${port}" \
    --policy-config "${POLICY_CONFIG}" \
    --policy-dir "${POLICY_DIR}" \
    --proposal-variant scheme4_lac_double_align \
    --policy-data-future-teacher-obs-offset "${FUTURE_OFFSET}" \
    --policy-scheme4-delta-enabled true \
    --policy-scheme4-inference-delta-control normal \
    --policy-scheme4-inference-fusion-scale "${FUSION_SCALE}" \
    "${diagnostics_args[@]}" \
    --mujoco-egl-device-id "${GPU}" \
    --libero-pythonpath "${LIBERO_ROOT}" \
    --result-dir "${RESULT_ROOT}" \
    --log-dir "${LOG_ROOT}" \
    --server-ready-timeout-sec 900 \
    --no-save-videos >"${pipeline_log}" 2>&1
  "${OPENPI_PYTHON}" examples/libero/summarize_jsonl_results.py \
    --results-jsonl "${result_dir}" \
    --task-classification "${CLASSIFICATION_JSON}" \
    --suite "${suite}" \
    --out-summary-csv "${summary_csv}" >>"${pipeline_log}" 2>&1
}

for index in "${!requested_suites[@]}"; do
  run_suite "${requested_suites[${index}]}" "$((PORT_BASE + index))"
done
