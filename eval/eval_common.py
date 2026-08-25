"""Shared ground-truth loading + metrics for gopro_umi policy evaluation.

Ground truth = canonical 10 Hz LeRobot dataset (yaw-free 6D state [X,Y,Z,Roll,Pitch,Grip],
step-to-step incremental action A[t]=S[t+1]-S[t]).  Every model is evaluated on the same
(episode, t) anchors: given the observation at t, predict the next 15 actions / states.
"""
from __future__ import annotations
import json, io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PROJECT = Path.home() / "GoPro_Umi/gopro_umi"
DATASET = PROJECT / "7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
BASE_ZARR = PROJECT / "7_storage/datasets/202608161903/02_zarr/replay_buffer.zarr"
HOLDOUT = list(range(90, 99))          # 0-based episode indices held out from both trainings
TRAIN_EVAL = [0, 11, 22, 33, 44, 55, 66, 77, 88]  # a few training episodes for train/holdout contrast
HORIZON = 15
STATE_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Gripper"]


def load_gt():
    """Return dict episode_index -> dict(states [N+1,6] incl. terminal, actions [N,6], src_idx [N], img_bytes list)."""
    files = sorted((DATASET / "data").glob("chunk-*/file-*.parquet"))
    tabs = [pq.read_table(f) for f in files]
    import pyarrow as pa
    t = pa.concat_tables(tabs).to_pandas()
    m = pq.read_table(DATASET / "meta/step9a_source_to_10hz_mapping.parquet").to_pandas()
    assert len(m) == len(t)
    t = t.sort_values("index").reset_index(drop=True)
    m = m.sort_values("output_index").reset_index(drop=True)
    eps = {}
    for ep, g in t.groupby("episode_index"):
        g = g.sort_values("frame_index")
        idx = g.index.to_numpy()
        S = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        A = np.stack(g["action"].to_numpy()).astype(np.float64)
        # append terminal state (target of last action) so states has N+1 rows
        S_full = np.concatenate([S, (S[-1] + A[-1])[None]], axis=0)
        eps[int(ep)] = dict(
            states=S_full, actions=A,
            src_idx=m.loc[idx, "source_global_index"].to_numpy(),
            src_next_idx=m.loc[idx, "source_next_global_index"].to_numpy(),
            timestamps=g["timestamp"].to_numpy(),
            imgs=[r["bytes"] for r in g["observation.images.top"].to_numpy()],
        )
    return eps


def anchors(ep_data, stride=5):
    """Anchor indices t such that A[t:t+15] fully exists."""
    n = len(ep_data["actions"])
    return list(range(0, n - HORIZON + 1, stride))


def decode_img(b):
    from PIL import Image
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def action_std():
    return np.array(json.load(open(DATASET / "meta/stats.json"))["action"]["std"], dtype=np.float64)


# ---------------------------------------------------------------- metrics
def integrate(s0, actions):
    """S[k+1]=S[k]+A[k] sequential reconstruction (canonical contract)."""
    return s0[None] + np.cumsum(actions, axis=0)


def dtw_distance(a, b):
    """Plain DTW with Euclidean local cost; a [T,D], b [T',D]. Returns normalized (per-step) distance."""
    T, U = len(a), len(b)
    D = np.full((T + 1, U + 1), np.inf); D[0, 0] = 0.0
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    for i in range(1, T + 1):
        for j in range(1, U + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[T, U] / (T + U)


def dct_coeffs(x, k=None):
    """Orthonormal DCT-II along time (axis 0), like FAST. x [T,D] -> [K,D]."""
    from scipy.fft import dct
    c = dct(x, axis=0, norm="ortho")
    return c if k is None else c[:k]


def dct_similarity(pred, gt, k=5):
    """Compressed time-series similarity: cosine similarity between low-frequency DCT coefficient vectors."""
    cp, cg = dct_coeffs(pred, k).ravel(), dct_coeffs(gt, k).ravel()
    den = np.linalg.norm(cp) * np.linalg.norm(cg)
    return float(cp @ cg / den) if den > 0 else 0.0


def levenshtein(a, b):
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class FastTokenizer:
    """physical-intelligence/fast (FAST+ universal action tokenizer): DCT -> quantize -> BPE."""
    def __init__(self, scale=None):
        from transformers import AutoProcessor
        self.tok = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        self.scale = scale   # per-dim normalization -> approx [-1,1] as recommended by FAST paper (q1/q99)

    def norm(self, a):
        return a / self.scale if self.scale is not None else a

    def encode(self, chunk):   # chunk [T,D] physical units
        return list(self.tok(self.norm(chunk)[None])[0])

    def decode(self, tokens, T, D):
        out = self.tok.decode([tokens], time_horizon=T, action_dim=D)[0]
        return out * self.scale if self.scale is not None else out


def chunk_metrics(pred_actions, gt_actions, s0, std, fast=None):
    """All per-chunk metrics. Actions in physical units [15,6]."""
    ps, gs = integrate(s0, pred_actions), integrate(s0, gt_actions)
    err = ps - gs
    m = {}
    m["action_mse"] = float(np.mean((pred_actions - gt_actions) ** 2))
    m["action_nmse"] = float(np.mean(((pred_actions - gt_actions) / std) ** 2))   # per-dim std-normalized
    m["action_mse_dim"] = np.mean((pred_actions - gt_actions) ** 2, axis=0)
    m["traj_pos_rmse_m"] = float(np.sqrt(np.mean(np.sum(err[:, :3] ** 2, axis=1))))
    m["traj_pos_final_err_m"] = float(np.linalg.norm(err[-1, :3]))
    m["traj_rp_rmse_rad"] = float(np.sqrt(np.mean(np.sum(err[:, 3:5] ** 2, axis=1))))
    m["traj_grip_rmse"] = float(np.sqrt(np.mean(err[:, 5] ** 2)))
    m["traj_pos_err_per_step"] = np.linalg.norm(err[:, :3], axis=1)   # [15]
    # normalized 6D trajectory time series (per-dim scale = std of state increments*15 ~ use action std*sqrt(15))
    sc = std * np.sqrt(HORIZON)
    m["dtw_norm"] = dtw_distance((ps - s0) / sc, (gs - s0) / sc)
    m["dct_cos_k5"] = dct_similarity(pred_actions / std, gt_actions / std, k=5)
    m["dct_cos_k3"] = dct_similarity(pred_actions / std, gt_actions / std, k=3)
    # displacement direction agreement (net motion over the chunk)
    dp, dg = ps[-1, :3] - s0[:3], gs[-1, :3] - s0[:3]
    den = np.linalg.norm(dp) * np.linalg.norm(dg)
    m["disp_cos"] = float(dp @ dg / den) if den > 1e-9 else np.nan
    if fast is not None:
        tp, tg = fast.encode(pred_actions), fast.encode(gt_actions)
        m["fast_ntok_pred"], m["fast_ntok_gt"] = len(tp), len(tg)
        m["fast_exact"] = float(tp == tg)
        m["fast_norm_edit"] = levenshtein(tp, tg) / max(len(tp), len(tg), 1)
        # FAST round-trip of GT (compression error floor) vs pred decoded through same codec
        gt_rt = fast.decode(tg, HORIZON, 6)
        m["fast_gt_roundtrip_nmse"] = float(np.mean(((gt_rt - gt_actions) / std) ** 2))
        # pre-BPE FAST representation: quantized DCT coefficients (scale=10, as in FAST); fraction of equal ints
        qp = np.around(dct_coeffs(fast.norm(pred_actions)) * 10.0); qg = np.around(dct_coeffs(fast.norm(gt_actions)) * 10.0)
        m["fast_prebpe_match"] = float(np.mean(qp == qg))
        nz = qg != 0
        m["fast_prebpe_match_nonzero"] = float(np.mean(qp[nz] == qg[nz])) if nz.any() else np.nan
    return m
