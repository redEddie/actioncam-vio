#!/usr/bin/env bash
# Create the 'umi' conda environment for the recording / dataset side.
# pyrealsense2 is pinned to match the system librealsense (2.58.2) so the
# Python bindings and the C++ SDK / ORB-SLAM3 build stay ABI/feature consistent.
#   Run:  bash env/setup_umi_env.sh
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="umi"
PY_VER="3.12"   # cp312 wheel of pyrealsense2 2.58.2 is available

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "==> conda env '${ENV_NAME}' already exists"
else
  echo "==> creating conda env '${ENV_NAME}' (python ${PY_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

echo "==> installing python deps"
conda run -n "${ENV_NAME}" pip install \
  "pyrealsense2==2.58.2.10647" \
  numpy \
  opencv-python \
  pyyaml

echo "==> verify"
conda run -n "${ENV_NAME}" python -c \
  "import pyrealsense2 as rs, cv2, numpy, yaml; print('OK: pyrealsense2', getattr(rs,'__version__','?'), '| cv2', cv2.__version__)"

echo
echo "Activate with:  conda activate ${ENV_NAME}"
