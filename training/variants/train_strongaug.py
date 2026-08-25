#!/usr/bin/env python3
"""lerobot_train with strong image augmentation injected (CLI can't set nested tfs kwargs)."""
import sys
import lerobot.scripts.lerobot_train as T
from lerobot.transforms.transforms import ImageTransformConfig

_orig = T.make_train_eval_datasets
def patched(cfg, *a, **k):
    it = cfg.dataset.image_transforms
    it.enable = True; it.max_num_transforms = 6; it.random_order = True
    it.tfs["brightness"] = ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"brightness": (0.6, 1.4)})
    it.tfs["contrast"]   = ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"contrast": (0.6, 1.4)})
    it.tfs["saturation"] = ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"saturation": (0.5, 1.5)})
    it.tfs["hue"]        = ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"hue": (-0.08, 0.08)})
    it.tfs["sharpness"]  = ImageTransformConfig(weight=1.0, type="SharpnessJitter", kwargs={"sharpness": (0.5, 1.5)})
    it.tfs["affine"]     = ImageTransformConfig(weight=1.0, type="RandomAffine", kwargs={"degrees": (-8.0, 8.0), "translate": (0.08, 0.08), "scale": (0.9, 1.1)})
    print("[strongaug] applied:", {n: (c.type, c.kwargs) for n, c in it.tfs.items()}, flush=True)
    return _orig(cfg, *a, **k)
T.make_train_eval_datasets = patched
from lerobot.scripts.lerobot_train import main
if __name__ == "__main__":
    main()
