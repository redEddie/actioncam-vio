"""STEP 4 Dataset Distribution and Visual-to-Action Correlation Audit Script."""

import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"

def detect_cup(img_bgr: np.ndarray) -> tuple[float, float, float, float, float]:
    """Detect cup in 224x224 RGB image, returns (cx_norm, cy_norm, w_norm, h_norm, confidence)."""
    h_img, w_img, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Search workspace area
        if 200 < area < 12000 and 0.2 < w/h < 5.0 and y > 25:
            candidates.append((area, (x, y, w, h)))
    
    candidates.sort(reverse=True)
    if candidates:
        top_area, (x, y, w, h) = candidates[0]
        cx = (x + w / 2.0) / w_img
        cy = (y + h / 2.0) / h_img
        w_norm = w / w_img
        h_norm = h / h_img
        conf = min(1.0, top_area / 4000.0)
        return cx, cy, w_norm, h_norm, conf
    else:
        return 0.5, 0.5, 0.2, 0.2, 0.0

def main():
    print("==================================================")
    print(" STEP 4 DATASET DISTRIBUTION & CORRELATION AUDIT")
    print("==================================================\n")

    files = sorted(list((DATASET_PATH / "data").glob("**/*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATASET_PATH}")

    df = pq.read_table(files).to_pandas().sort_values("index").reset_index(drop=True)
    num_transitions = len(df)
    num_episodes = df["episode_index"].nunique()
    print(f"Loaded dataset: {num_episodes} episodes, {num_transitions} transitions.")

    # 1. Process each frame: extract state, action, future action chunk sum, and cup detection
    debug_dir = PROJECT_ROOT / "cup_detection_debug"
    debug_dir.mkdir(exist_ok=True)

    records = []
    episode_starts = []
    episode_reaches = []

    # Map raw action chunks of size 15 for each frame
    actions_arr = np.stack(df["action"].values) # [N, 6]
    states_arr = np.stack(df["observation.state"].values) # [N, 6]
    ep_indices = df["episode_index"].values
    frame_indices = df["frame_index"].values

    saved_debug_count = 0
    valid_detections = 0

    print("Analyzing frames and detecting cup positions...")
    for i in range(num_transitions):
        ep = ep_indices[i]
        fi = frame_indices[i]
        
        # Check future 15 steps within the same episode
        end_idx = min(num_transitions, i + 15)
        # Verify all belong to same episode
        same_ep_mask = (ep_indices[i:end_idx] == ep)
        valid_len = same_ep_mask.sum()
        if valid_len < 15:
            # Pad with last action if near episode boundary
            chunk = np.zeros((15, 6), dtype=np.float32)
            chunk[:valid_len] = actions_arr[i:i+valid_len]
            if valid_len > 0:
                chunk[valid_len:] = actions_arr[i+valid_len-1]
        else:
            chunk = actions_arr[i:i+15]

        sum_dx = chunk[:, 0].sum()
        sum_dy = chunk[:, 1].sum()
        sum_dz = chunk[:, 2].sum()
        early_sum_dy = chunk[:5, 1].sum()
        late_sum_dy = chunk[5:, 1].sum()
        a0_dy = chunk[0, 1]

        # Decode image
        img_bytes = df.at[i, "observation.images.top"]["bytes"]
        img_pil = Image.open(io.BytesIO(img_bytes))
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        cx, cy, w_box, h_box, conf = detect_cup(img_bgr)
        if conf > 0.1:
            valid_detections += 1

        # Save 25 sample debug visualizations
        if fi == 0 and saved_debug_count < 25:
            vis = img_bgr.copy()
            x1 = int((cx - w_box/2) * 224)
            y1 = int((cy - h_box/2) * 224)
            x2 = int((cx + w_box/2) * 224)
            y2 = int((cy + h_box/2) * 224)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx*224), int(cy*224)), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"Ep{ep} cx={cx:.2f} sum_dy={sum_dy:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(str(debug_dir / f"audit_ep{ep:02d}.png"), vis)
            saved_debug_count += 1

        records.append({
            "episode_index": ep,
            "frame_index": fi,
            "state": states_arr[i],
            "cup_x": cx,
            "cup_y": cy,
            "conf": conf,
            "sum_dx": sum_dx,
            "sum_dy": sum_dy,
            "sum_dz": sum_dz,
            "a0_dy": a0_dy,
            "early_sum_dy": early_sum_dy,
            "late_sum_dy": late_sum_dy,
        })

    # Group by episode start
    ep_starts = {}
    for r in records:
        ep = r["episode_index"]
        if ep not in ep_starts:
            ep_starts[ep] = r

    start_list = list(ep_starts.values())

    # 2. Cup Position Distribution Statistics (at episode starts)
    start_cup_x = np.array([r["cup_x"] for r in start_list])
    start_cup_y = np.array([r["cup_y"] for r in start_list])
    all_cup_x = np.array([r["cup_x"] for r in records if r["frame_index"] <= 15]) # Pre-grasp phase

    print("\n--- 1. CUP POSITION DISTRIBUTION (Episode Starts) ---")
    print(f"Mean: {start_cup_x.mean():.4f}")
    print(f"Std:  {start_cup_x.std():.4f}")
    print(f"Min:  {start_cup_x.min():.4f}")
    print(f"Max:  {start_cup_x.max():.4f}")
    print(f"P5:   {np.percentile(start_cup_x, 5):.4f}")
    print(f"P25:  {np.percentile(start_cup_x, 25):.4f}")
    print(f"P50:  {np.median(start_cup_x):.4f}")
    print(f"P75:  {np.percentile(start_cup_x, 75):.4f}")
    print(f"P95:  {np.percentile(start_cup_x, 95):.4f}")

    # Bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["Far Left (0~0.2)", "Left (0.2~0.4)", "Center (0.4~0.6)", "Right (0.6~0.8)", "Far Right (0.8~1.0)"]
    hist, _ = np.histogram(start_cup_x, bins=bins)
    print("\n--- 2. CUP POSITION BINS (Episode Starts, N=99) ---")
    for label, count in zip(bin_labels, hist):
        print(f" {label:<22}: {count:>3} episodes ({count/len(start_list)*100:>5.1f}%)")

    # 3. Cup X vs GT Reaching Trajectory Correlation
    pre_grasp_records = [r for r in records if r["frame_index"] <= 10]
    pg_cup_x = np.array([r["cup_x"] for r in pre_grasp_records])
    pg_sum_dy = np.array([r["sum_dy"] for r in pre_grasp_records])
    pg_sum_dx = np.array([r["sum_dx"] for r in pre_grasp_records])
    pg_sum_dz = np.array([r["sum_dz"] for r in pre_grasp_records])

    p_dy, _ = pearsonr(pg_cup_x, pg_sum_dy)
    s_dy, _ = spearmanr(pg_cup_x, pg_sum_dy)
    p_dx, _ = pearsonr(pg_cup_x, pg_sum_dx)
    p_dz, _ = pearsonr(pg_cup_x, pg_sum_dz)

    print("\n--- 3. PRE-GRASP CORRELATION (Cup X vs Future 1.5s Trajectory) ---")
    print(f"Pearson(Cup X, Sum dY):  {p_dy:.4f}")
    print(f"Spearman(Cup X, Sum dY): {s_dy:.4f}")
    print(f"Pearson(Cup X, Sum dX):  {p_dx:.4f}")
    print(f"Pearson(Cup X, Sum dZ):  {p_dz:.4f}")

    # 4. Position Group Motion Comparison (Left vs Center vs Right)
    left_mask = pg_cup_x < 0.32
    center_mask = (pg_cup_x >= 0.32) & (pg_cup_x <= 0.42)
    right_mask = pg_cup_x > 0.42

    print("\n--- 4. POSITION GROUP MOTION COMPARISON ---")
    print(f"{'Group':<10} | {'N':>5} | {'ΣdX mean±std (m)':>20} | {'ΣdY mean±std (m)':>20} | {'ΣdZ mean±std (m)':>20}")
    print("-" * 85)
    for name, m in [("Left", left_mask), ("Center", center_mask), ("Right", right_mask)]:
        if m.sum() > 0:
            mdx, sdx = pg_sum_dx[m].mean(), pg_sum_dx[m].std()
            mdy, sdy = pg_sum_dy[m].mean(), pg_sum_dy[m].std()
            mdz, sdz = pg_sum_dz[m].mean(), pg_sum_dz[m].std()
            print(f"{name:<10} | {m.sum():>5} | {mdx:>8.5f} ± {sdx:>7.5f} | {mdy:>8.5f} ± {sdy:>7.5f} | {mdz:>8.5f} ± {sdz:>7.5f}")

    # 5. Episode-Level Correlation
    ep_cup_x = np.array([r["cup_x"] for r in start_list])
    ep_sum_dy = np.array([r["sum_dy"] for r in start_list])
    ep_p_dy, _ = pearsonr(ep_cup_x, ep_sum_dy)
    ep_s_dy, _ = spearmanr(ep_cup_x, ep_sum_dy)
    print("\n--- 5. EPISODE-LEVEL CORRELATION (N=99) ---")
    print(f"Episode-Level Pearson(Cup X, Reaching Sum dY):  {ep_p_dy:.4f}")
    print(f"Episode-Level Spearman(Cup X, Reaching Sum dY): {ep_s_dy:.4f}")

    # 6. Start State Distribution
    start_states = np.stack([r["state"] for r in start_list])
    state_names = ["X (m)", "Y (m)", "Z (m)", "Roll (deg)", "Pitch (deg)", "Gripper"]
    print("\n--- 6. START STATE DISTRIBUTION (Episode Starts) ---")
    print(f"{'Axis':<15} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 65)
    for idx, name in enumerate(state_names):
        vals = start_states[:, idx]
        print(f"{name:<15} | {vals.mean():>10.4f} | {vals.std():>10.4f} | {vals.min():>10.4f} | {vals.max():>10.4f}")

    # 7. Predictive Shortcut Comparison (5-fold Episode-level Cross Validation)
    print("\n--- 7. PREDICTIVE SHORTCUT COMPARISON (5-Fold Episode CV) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    episodes = np.array(list(range(num_episodes)))

    X_cup = np.array([[r["cup_x"]] for r in start_list])
    X_state = start_states
    X_both = np.hstack([X_state, X_cup])
    y_dy = ep_sum_dy

    def eval_cv(X, y):
        r2_list, mae_list = [], []
        for train_idx, val_idx in kf.split(episodes):
            reg = Ridge(alpha=1.0)
            reg.fit(X[train_idx], y[train_idx])
            pred = reg.predict(X[val_idx])
            ss_tot = ((y[val_idx] - y[val_idx].mean()) ** 2).sum()
            ss_res = ((y[val_idx] - pred) ** 2).sum()
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            mae = np.abs(y[val_idx] - pred).mean()
            r2_list.append(r2)
            mae_list.append(mae)
        return np.mean(r2_list), np.mean(mae_list)

    r2_cup, mae_cup = eval_cv(X_cup, y_dy)
    r2_state, mae_state = eval_cv(X_state, y_dy)
    r2_both, mae_both = eval_cv(X_both, y_dy)

    print(f"Target = Future Reaching ΣdY")
    print(f"1. Cup X only:      CV R² = {r2_cup:>8.4f}, CV MAE = {mae_cup:.5f} m")
    print(f"2. State only:      CV R² = {r2_state:>8.4f}, CV MAE = {mae_state:.5f} m")
    print(f"3. State + Cup X:   CV R² = {r2_both:>8.4f}, CV MAE = {mae_both:.5f} m")

    # 8. Early vs Late Visual Signal
    ep_a0_dy = np.array([r["a0_dy"] for r in start_list])
    ep_early_dy = np.array([r["early_sum_dy"] for r in start_list])
    ep_late_dy = np.array([r["late_sum_dy"] for r in start_list])

    p_a0, _ = pearsonr(ep_cup_x, ep_a0_dy)
    p_early, _ = pearsonr(ep_cup_x, ep_early_dy)
    p_late, _ = pearsonr(ep_cup_x, ep_late_dy)

    print("\n--- 8. EARLY VS LATE VISUAL SIGNAL ---")
    print(f"Cup X vs A0 dY:       Pearson = {p_a0:.4f}")
    print(f"Cup X vs A0~A4 ΣdY:   Pearson = {p_early:.4f}")
    print(f"Cup X vs A5~A14 ΣdY:  Pearson = {p_late:.4f}")

    # 9. Generate Figures
    print("\nGenerating audit figures...")
    # Fig 1: Cup X Distribution
    plt.figure(figsize=(7, 5))
    plt.hist(start_cup_x, bins=15, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title("Step 4.1: Initial Cup X Position Distribution (99 Episodes)")
    plt.xlabel("Cup Center X (Normalized [0.0 = Left, 1.0 = Right])")
    plt.ylabel("Number of Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "step4_cup_x_distribution.png", dpi=150)
    plt.close()

    # Fig 2: Cup X vs Future Sum dY
    plt.figure(figsize=(7, 5))
    plt.scatter(ep_cup_x, ep_sum_dy, color="#e6550d", alpha=0.8, edgecolor="black", s=50)
    plt.title("Step 4.2: Cup X Position vs Future Reaching ΣdY (99 Episodes)")
    plt.xlabel("Cup Center X (Normalized)")
    plt.ylabel("Future Reaching ΣdY (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "step4_cup_x_vs_sum_dY.png", dpi=150)
    plt.close()

    # Fig 3: Start State Distribution
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 3, 1)
    plt.hist(start_states[:, 0], bins=10, color="#31a354", edgecolor="black")
    plt.title("Start X (m)")
    plt.subplot(1, 3, 2)
    plt.hist(start_states[:, 1], bins=10, color="#31a354", edgecolor="black")
    plt.title("Start Y (m)")
    plt.subplot(1, 3, 3)
    plt.hist(start_states[:, 2], bins=10, color="#31a354", edgecolor="black")
    plt.title("Start Z (m)")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "step4_start_state_distribution.png", dpi=150)
    plt.close()

    # Fig 4: Sum dY Distribution
    plt.figure(figsize=(7, 5))
    plt.hist(ep_sum_dy, bins=15, color="#756bb1", edgecolor="black", alpha=0.8)
    plt.title("Step 4.4: Future Reaching ΣdY Distribution (99 Episodes)")
    plt.xlabel("ΣdY (meters)")
    plt.ylabel("Number of Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "step4_sum_dY_distribution.png", dpi=150)
    plt.close()

    # Fig 5: Position Bin vs Sum dY Boxplot
    plt.figure(figsize=(7, 5))
    groups = [pg_sum_dy[left_mask], pg_sum_dy[center_mask], pg_sum_dy[right_mask]]
    plt.boxplot(groups, tick_labels=["Left", "Center", "Right"], patch_artist=True)
    plt.title("Step 4.5: Reaching ΣdY by Cup Position Bin")
    plt.ylabel("ΣdY (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "step4_position_bin_vs_sum_dY.png", dpi=150)
    plt.close()

    print("All figures saved successfully!")

if __name__ == "__main__":
    main()
