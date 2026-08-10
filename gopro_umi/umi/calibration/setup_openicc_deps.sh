#!/usr/bin/env bash
# [sudo] System deps for building OpenICC (OpenImuCameraCalibrator) + Ceres + pyTheiaSfM.
# opencv-contrib + eigen are already present on this machine; this adds the rest.
#   Run:  sudo bash calibration/setup_openicc_deps.sh
set -euo pipefail
apt-get update
apt-get install -y \
  build-essential cmake pkg-config git gfortran \
  libgoogle-glog-dev libgflags-dev \
  libatlas-base-dev libsuitesparse-dev \
  libeigen3-dev libopencv-dev libopencv-contrib-dev
echo
echo "OK. Next (no sudo):  bash calibration/build_openicc.sh"
