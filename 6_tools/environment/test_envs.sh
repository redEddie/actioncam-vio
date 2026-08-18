
for E in "gopro_env" "openvla_env" "env_isaaclab_vla" "lerobot_env2"; do
    echo "Testing $E"
    /home/kimminje/miniconda3/envs/$E/bin/python -c "import draccus; print('draccus OK')" 2>/dev/null
    /home/kimminje/miniconda3/envs/$E/bin/python -c "import torch; print('torch OK')" 2>/dev/null
done
