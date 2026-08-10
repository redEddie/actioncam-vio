#!/bin/bash
cd /home/kimminje/Desktop/project/gopro_umi
python 4_deploy/deploy_smolvla_yawfree_microstep.py \
  --run \
  --port /dev/ttyACM0 \
  --camera-index 0 \
  --max-decisions 200 \
  --duration-s 0.1 \
  --interpolation-steps 1 \
  --start-duration-s 3.0 \
  --start-interpolation-steps 10
