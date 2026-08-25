#!/usr/bin/env python3
"""lerobot_train + SE(3) relative-waypoint action step + strong augmentation."""
import sys
sys.path.insert(0, '/home/jeonchanwook/work/smolvla/variants')
import relwp_se3_step; relwp_se3_step.install()
import lerobot.scripts.lerobot_train as T
from lerobot.transforms.transforms import ImageTransformConfig
_orig = T.make_train_eval_datasets
def patched(cfg, *a, **k):
    it = cfg.dataset.image_transforms
    it.enable = True; it.max_num_transforms = 6; it.random_order = True
    it.tfs["brightness"] = ImageTransformConfig(1.0, "ColorJitter", {"brightness": (0.6, 1.4)})
    it.tfs["contrast"]   = ImageTransformConfig(1.0, "ColorJitter", {"contrast": (0.6, 1.4)})
    it.tfs["saturation"] = ImageTransformConfig(1.0, "ColorJitter", {"saturation": (0.5, 1.5)})
    it.tfs["hue"]        = ImageTransformConfig(1.0, "ColorJitter", {"hue": (-0.08, 0.08)})
    it.tfs["sharpness"]  = ImageTransformConfig(1.0, "SharpnessJitter", {"sharpness": (0.5, 1.5)})
    it.tfs["affine"]     = ImageTransformConfig(1.0, "RandomAffine", {"degrees": (-8.0, 8.0), "translate": (0.08, 0.08), "scale": (0.9, 1.1)})
    print("[strongaug] on", flush=True)
    return _orig(cfg, *a, **k)
T.make_train_eval_datasets = patched
from lerobot.scripts.lerobot_train import main
if __name__ == "__main__":
    main()
