#!/bin/bash
# Install the locked environment and create portable local defaults once.
# Pass uv sync options, e.g. --extra crossdocked for the docking toolchain.
set -e

MORPH_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MORPH_OUTPUT_ROOT="${MORPH_SCRATCH:-${MORPH_PROJECT_DIR}/outputs}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [ ! -f "${MORPH_PROJECT_DIR}/.env" ]; then
    cat > "${MORPH_PROJECT_DIR}/.env" <<EOL
export PROJECT_ROOT="${MORPH_PROJECT_DIR}"
export HYDRA_JOBS="${MORPH_PROJECT_DIR}"
export MORPH_SCRATCH="${MORPH_OUTPUT_ROOT}"
export MORPH_RUN_ROOT="${MORPH_RUN_ROOT:-${MORPH_OUTPUT_ROOT}/runs}"
export WANDB_DIR="\${MORPH_RUN_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
EOL
fi

# uv creates .venv, installs Morph editable, and checks uv.lock against pyproject.toml.
uv sync --locked --project "${MORPH_PROJECT_DIR}" "$@"

# Native docking programs are opt-in, just like their Python dependencies.
MORPH_INSTALL_CROSSDOCKED=false
MORPH_PREVIOUS_ARG=""
for MORPH_ARG in "$@"; do
    if [[ "${MORPH_ARG}" == "--extra=crossdocked" || "${MORPH_ARG}" == "--all-extras" ||
          ( "${MORPH_PREVIOUS_ARG}" == "--extra" && "${MORPH_ARG}" == "crossdocked" ) ]]; then
        MORPH_INSTALL_CROSSDOCKED=true
    fi
    if [[ "${MORPH_ARG}" == "--dry-run" || "${MORPH_ARG}" == "--no-extra=crossdocked" ||
          ( "${MORPH_PREVIOUS_ARG}" == "--no-extra" && "${MORPH_ARG}" == "crossdocked" ) ]]; then
        MORPH_INSTALL_CROSSDOCKED=false
        break
    fi
    MORPH_PREVIOUS_ARG="${MORPH_ARG}"
done
if ${MORPH_INSTALL_CROSSDOCKED}; then
    "${MORPH_PROJECT_DIR}/.venv/bin/python" \
        "${MORPH_PROJECT_DIR}/scripts/install_crossdocked_tools.py"
fi

echo "Environment ready. Activate it and load the run paths:"
echo "  source ${MORPH_PROJECT_DIR}/.venv/bin/activate"
echo "  source ${MORPH_PROJECT_DIR}/.env"
