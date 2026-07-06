"""IMU-camera temporal + rotational calibration from the ChArUco video.

- time offset: cross-correlation of |angular velocity| seen by the gyro vs
  the camera (from consecutive ChArUco board poses)
- rotation R_imu_cam: Kabsch/Wahba alignment of the two angular velocity
  vector streams
- validation: gravity-in-board-frame consistency using the accelerometer

Usage:
  python -m gopro_vio.imu_sync calibration/board_poses.npz \
      output/GX014222/imu.csv -o calibration
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R


def _lowpass(t, x, cutoff_hz):
    """Zero-phase low-pass of a (possibly gappy) signal: resample onto a
    uniform grid, filtfilt, sample back at the original times."""
    fs = 1.0 / np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], 1.0 / fs)
    b, a = butter(4, cutoff_hz / (fs / 2), btype="low")
    cols = []
    x2 = x if x.ndim == 2 else x[:, None]
    for i in range(x2.shape[1]):
        xu = np.interp(grid, t, x2[:, i])
        cols.append(np.interp(t, grid, filtfilt(b, a, xu)))
    out = np.stack(cols, axis=1)
    return out if x.ndim == 2 else out[:, 0]


def camera_angular_velocity(times, rvecs, max_dt):
    """Body-frame angular velocity of the camera from consecutive poses.

    R_cb(t) maps board->camera.  Exp(w dt) = R_wc(t)^T R_wc(t+dt)
    = R_cb(t) R_cb(t+dt)^T with w expressed in the camera frame at t.
    """
    rot = R.from_rotvec(rvecs)
    w_t, w = [], []
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt <= 0 or dt > max_dt:
            continue
        dR = rot[i] * rot[i + 1].inv()
        w.append(dR.as_rotvec() / dt)
        w_t.append(0.5 * (times[i] + times[i + 1]))
    return np.asarray(w_t), np.asarray(w)


def find_time_offset(t_cam, w_cam, t_imu, w_imu, search=0.5, fs=200.0):
    """Offset such that imu(t) matches camera(t - offset); i.e. positive
    offset means IMU timeline lags the video timeline."""
    lo = max(t_cam[0], t_imu[0]) + search
    hi = min(t_cam[-1], t_imu[-1]) - search
    grid = np.arange(lo, hi, 1.0 / fs)
    a = np.interp(grid, t_cam, np.linalg.norm(w_cam, axis=1))
    b = np.interp(grid, t_imu, np.linalg.norm(w_imu, axis=1))
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(search * fs)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.dot(a[max(0, -l): len(a) - max(0, l)],
                            b[max(0, l): len(b) - max(0, -l)])
                     / (len(a) - abs(l)) for l in lags])
    k = int(np.argmax(corr))
    # parabolic sub-sample refinement
    if 0 < k < len(corr) - 1:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        denom = c0 - 2 * c1 + c2
        frac = 0.5 * (c0 - c2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    offset = (lags[k] + frac) / fs
    peak = corr[k] / (np.std(a) * np.std(b) + 1e-12)
    return float(offset), float(peak)


def solve_rotation(t_cam, w_cam, t_imu, w_imu, offset):
    """R_imu_cam s.t. w_imu(t) ~= R @ w_cam(t - offset).  Kabsch, weighted
    towards high-rate samples (better SNR)."""
    t_c = t_cam + offset  # camera events on the IMU clock
    lo, hi = max(t_c[0], t_imu[0]), min(t_c[-1], t_imu[-1])
    m = (t_c >= lo) & (t_c <= hi)
    wc = w_cam[m]
    wi = np.stack([np.interp(t_c[m], t_imu, w_imu[:, i]) for i in range(3)], 1)
    mag = np.linalg.norm(wc, axis=1)
    keep = mag > np.percentile(mag, 50)  # informative samples only
    A = wc[keep].T @ wi[keep]            # 3x3
    U, S, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R_ic = Vt.T @ np.diag([1, 1, d]) @ U.T
    resid = wi[keep] - wc[keep] @ R_ic.T
    rms = float(np.sqrt((resid ** 2).sum(1).mean()))
    rms_rel = rms / float(np.sqrt((wi[keep] ** 2).sum(1).mean()))
    return R_ic, rms, rms_rel, S


def gravity_check(t_cam, rvecs_cb, t_imu, acc_imu, w_imu, R_ic, offset):
    """Rotate measured specific force into the board frame; if calibration is
    right it should be a constant ~9.8 m/s^2 vector during slow motion."""
    rot_cb = R.from_rotvec(rvecs_cb)
    t_c = t_cam + offset
    acc = np.stack([np.interp(t_c, t_imu, acc_imu[:, i]) for i in range(3)], 1)
    wmag = np.interp(t_c, t_imu, np.linalg.norm(w_imu, axis=1))
    slow = wmag < np.percentile(wmag, 30)
    g_board = np.einsum("nij,nj->ni", rot_cb[slow].inv().as_matrix(),
                        acc[slow] @ R_ic)  # R_ci = R_ic^T applied per-sample
    mean = g_board.mean(0)
    spread = float(np.linalg.norm(g_board - mean, axis=1).std())
    return mean, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("poses")
    ap.add_argument("imu_csv")
    ap.add_argument("-o", "--out", default="calibration")
    ap.add_argument("--fps", type=float, default=59.94)
    args = ap.parse_args()

    z = np.load(args.poses)
    imu = np.loadtxt(args.imu_csv, delimiter=",", skiprows=1)
    t_imu, w_imu, a_imu = imu[:, 0], imu[:, 1:4], imu[:, 4:7]

    t_cam, w_cam = camera_angular_velocity(z["t"], z["rvec"],
                                           max_dt=1.5 / args.fps)
    print(f"[sync] camera angular velocity samples: {len(t_cam)}")

    # differentiated 60 Hz PnP poses are noisy and the 200 Hz gyro has
    # content above the camera Nyquist: low-pass both to a common band
    cutoff = 8.0
    w_cam = _lowpass(t_cam, w_cam, cutoff)
    w_imu = _lowpass(t_imu, w_imu, cutoff)

    offset, peak = find_time_offset(t_cam, w_cam, t_imu, w_imu)
    print(f"[sync] time offset (imu - video): {offset*1000:.2f} ms  "
          f"(corr peak {peak:.3f})")

    R_ic, rms, rms_rel, S = solve_rotation(t_cam, w_cam, t_imu, w_imu, offset)
    print(f"[sync] R_imu_cam residual: {rms:.4f} rad/s "
          f"({rms_rel*100:.1f}% of signal)  svd ratio {S[0]/S[2]:.1f}")
    print("R_imu_cam=\n", np.array_str(R_ic, precision=4, suppress_small=True))
    eul = R.from_matrix(R_ic).as_euler("xyz", degrees=True)
    print(f"        (xyz euler deg: {eul.round(2)})")

    g_mean, g_spread = gravity_check(z["t"], z["rvec"], t_imu, a_imu,
                                     w_imu, R_ic, offset)
    print(f"[check] gravity in board frame: {g_mean.round(3)} "
          f"|g|={np.linalg.norm(g_mean):.3f} m/s^2, spread {g_spread:.3f}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "imu_extrinsics.json").write_text(json.dumps({
        "time_offset_s": offset,
        "time_offset_note": "imu_time = video_time + offset; subtract from "
                            "IMU cts to align to video",
        "corr_peak": peak,
        "R_imu_cam": R_ic.tolist(),
        "gyro_residual_rms_rad_s": rms,
        "gyro_residual_relative": rms_rel,
        "gravity_board_frame": g_mean.tolist(),
        "gravity_norm": float(np.linalg.norm(g_mean)),
        "gravity_spread": g_spread,
        "t_imu_cam_m": [0.0, 0.0, 0.0],
        "t_note": "translation not observable from gyro-only alignment; "
                  "set to zero (GoPro IMU sits ~1-2 cm from lens)",
    }, indent=2))
    print(f"[done] wrote {out/'imu_extrinsics.json'}")


if __name__ == "__main__":
    main()
