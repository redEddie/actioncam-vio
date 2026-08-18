"""STEP 3 Vision Ablation and Sensitivity Diagnostic."""

from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_condition_images(base_rgb: np.ndarray) -> dict[str, np.ndarray]:
    h, w, c = base_rgb.shape
    assert (h, w, c) == (256, 256, 3)

    # Condition A: Center
    cond_a = base_rgb.copy()

    # Condition B: Left (shift image content left by 35 px)
    shift_b = -35
    M_b = np.float32([[1, 0, shift_b], [0, 1, 0]])
    cond_b = cv2.warpAffine(base_rgb, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Condition C: Right (shift image content right by 35 px)
    shift_c = +35
    M_c = np.float32([[1, 0, shift_c], [0, 1, 0]])
    cond_c = cv2.warpAffine(base_rgb, M_c, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Condition D: No Cup (inpaint center area with table background)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h//3: 4*h//5, w//4: 3*w//4] = 255
    cond_d = cv2.inpaint(base_rgb, mask, 7, cv2.INPAINT_TELEA)

    # Condition E: Black
    cond_e = np.zeros((256, 256, 3), dtype=np.uint8)

    return {
        "A_center": cond_a,
        "B_left": cond_b,
        "C_right": cond_c,
        "D_nocup": cond_d,
        "E_black": cond_e,
    }

def main():
    img_path = PROJECT_ROOT / "debug_current_camera_scene.png"
    if not img_path.exists():
        raise FileNotFoundError("debug_current_camera_scene.png not found")
    
    bgr = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape != (256, 256, 3):
        rgb = cv2.resize(rgb, (256, 256))

    conditions = create_condition_images(rgb)

    for name, img_rgb in conditions.items():
        out_name = f"step3_{name}.png"
        cv2.imwrite(str(PROJECT_ROOT / out_name), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
