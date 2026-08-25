#!/usr/bin/env python3
"""lerobot_train with RelativeWaypointActionStep injected. Same CLI as `python -m lerobot.scripts.lerobot_train`."""
import sys
sys.path.insert(0, '/home/jeonchanwook/work/smolvla/variants')
import relwp_step; relwp_step.install()
from lerobot.scripts.lerobot_train import main
if __name__ == "__main__":
    main()
