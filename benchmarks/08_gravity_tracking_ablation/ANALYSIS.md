# Benchmark 08 — Gravity Tracking Ablation Reports

---

# Step B8A — Gravity Direction Visual Check Report

## 1. Purpose
This diagnostic step isolates the canonical pose-dependent gravity compensation support bias ($b_{\text{support}}(q_{\text{actual}})$) on physical SO-101 arm hardware by **disabling the $K_{\text{ext}}$ tracking error feedback** ($q_{\text{corr}} = 0$).
This allows the user to directly observe the physical direction of motion imparted by the gravity support term alone relative to the canonical start pose.

---

## 2. Test Configuration & Parameters
* **Fixed $q_{\text{nom}}$**: Canonical Physical Start Pose (`[+1.4945, -83.5165, +79.8242, +45.3626, 0.0000]` deg URDF)
* **Command Law**: $q_{\text{cmd}} = q_{\text{nom}} + b_{\text{support}}(q_{\text{actual}})$
* **Tracking Error Correction ($K_{\text{ext}}$)**: **DISABLED ($0.0$)**
* **Test Duration**: $5.0$ seconds ($150$ ticks at $30.0$ Hz)
* **Applied Gravity Support Bias at Start**:
  * `shoulder_pan`: $+0.2640^\circ$
  * `shoulder_lift`: $+0.6058^\circ$
  * `elbow_flex`: $+2.2738^\circ$
  * `wrist_flex`: $+0.6842^\circ$
  * `wrist_roll`: $+0.3520^\circ$

---

## 3. Physical State Telemetry & Joint Displacements

* **$q_{\text{actual}}$ Before Gravity-Only Application (URDF)**:
  `[+1.4945, -83.3846, +81.2747, +45.7582, +0.2637]` deg
* **$q_{\text{actual}}$ After 5s Gravity-Only Hold (URDF)**:
  `[+1.5824, -82.6813, +83.2088, +46.2857, +0.2637]` deg

### Joint Delta Breakdown ($\Delta q = q_{\text{after}} - q_{\text{before}}$):
* `shoulder_pan`: $+0.0879^\circ$
* `shoulder_lift`: $+0.7033^\circ$
* `elbow_flex`: $+1.9341^\circ$
* `wrist_flex`: $+0.5275^\circ$
* `wrist_roll`: $+0.0000^\circ$

---

## 4. Hardware Safety & Immutability Audit
* `CAMERA ACCESSED`: **NO**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**
* `MOTOR READ FAILURES`: **0**
* `MOTOR WRITE FAILURES`: **0**
* `BENCHMARK-SPECIFIC PID WRITES`: **0**
* `TORQUE WRITES`: **0**
* `GRIPPER WRITES`: **0**
* `CALIBRATION WRITES`: **0**
* `EXISTING FILES MODIFIED/DELETED`: **0**

---

## 5. User Visual Observation
* **Observed Movement**: The robot arm moved **DOWN** away from canonical start pose when $+b_{\text{support}}$ was added.

---

# Step B8B — Reversed Gravity Sign + Accuracy Check Report

## 1. Purpose
This step validates the hypothesis that the gravity compensation bias sign was inverted relative to the physical arm coordinate frame. It evaluates:
1. Physical recovery motion when the gravity support term is subtracted: $q_{\text{cmd}} = q_{\text{nom}} - \alpha(t) \cdot b_{\text{support}}(q_{\text{actual}})$ with $K_{\text{ext}} = 0$.
2. Joint-level tracking accuracy against the canonical fixed start pose ($q_{\text{nom}}$).

---

## 2. Test Configuration & Invariants
* **Fixed $q_{\text{nom}}$**: `[+1.4945, -83.5165, +79.8242, +45.3626, 0.0000]` deg URDF
* **Tracking Error Correction ($K_{\text{ext}}$)**: **DISABLED ($0.0$)**
* **Command Law**:
  * $0.0\text{ s} \sim 1.0\text{ s}$ ($30$ ticks): $q_{\text{cmd}} = q_{\text{nom}} - \alpha(t) \cdot b_{\text{support}}(q_{\text{actual}})$ ($\alpha: 0 \to 1$)
  * $1.0\text{ s} \sim 5.0\text{ s}$ ($120$ ticks): $q_{\text{cmd}} = q_{\text{nom}} - b_{\text{support}}(q_{\text{actual}})$
* **Applied Support Bias**: `[+0.2640, +0.6020, +2.2641, +0.6703, +0.3520]` deg
* **Equation Invariant**: $\max |(q_{\text{cmd}} - q_{\text{nom}}) - (-b_{\text{support}})| = 7.11 \times 10^{-15\circ}$ (**PASS**).

---

## 3. Physical State Telemetry & Accuracy

* **$q_{\text{actual}}$ Before Reversed Gravity (URDF)**:
  `[+1.4945, -83.3846, +81.9780, +45.9341, +0.2637]` deg
* **$q_{\text{actual}}$ Final (URDF)**:
  `[+1.1429, -83.4725, +79.3407, +45.2308, -0.0879]` deg

### Last 2-Second Steady-State Accuracy ($t = 3.0\text{ s} \sim 5.0\text{ s}$, Final 60 Ticks):

| Joint | Signed Mean (deg) | Mean Abs (deg) | P50 Abs (deg) | P95 Abs (deg) | P99 Abs (deg) | Max Abs (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $-0.3516$ | $0.3516$ | $0.3516$ | $0.3516$ | $0.3516$ | $0.3516$ |
| **shoulder_lift** | $+0.0440$ | $0.0440$ | $0.0440$ | $0.0440$ | $0.0440$ | $0.0440$ |
| **elbow_flex** | $-0.4835$ | **$0.4835$** | $0.4835$ | $0.4835$ | $0.4835$ | $0.4835$ |
| **wrist_flex** | $-0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ |
| **wrist_roll** | $-0.0879$ | $0.0879$ | $0.0879$ | $0.0879$ | $0.0879$ | $0.0879$ |

---

## 4. Elbow Detail & Error Reduction

| Metric | Pitch Joint Error ($\text{deg}$) |
| :--- | :---: |
| **Elbow $q_{\text{nom}}$** | $79.8242^\circ$ |
| **Elbow $q_{\text{actual}}$ Before Test** | $81.9780^\circ$ ($\text{Error} = +2.1538^\circ$) |
| **Elbow $q_{\text{actual}}$ Final (Reversed)** | $79.3407^\circ$ ($\text{Error} = -0.4835^\circ$) |
| **Elbow Steady-State Mean Abs Error** | **$0.4835^\circ$** |
| **Same-Run Error Reduction** | **$+1.6703^\circ$** (Error reduced from $2.1538^\circ \to 0.4835^\circ$) |
| **Historical Comparison vs B8A ($+3.3846^\circ$)** | **$+2.9011^\circ$ improvement** |

---

## 5. Stability, Drift & Safety Audit
* `Target Crossings per Joint`: `[0, 0, 0, 0, 0]`
* `Error Sign Changes per Joint`: `[0, 0, 0, 0, 0]`
* `Steady Net Drift per Joint`: `[0.0, 0.0, 0.0, 0.0, 0.0]` deg
* `Oscillations Observed`: **NO**
* `Motor Read/Write Failures`: **0**
* `Benchmark-Specific PID Writes`: **0**
* `Torque Writes`: **0**
* `Gripper Writes`: **0**

---

## 6. Verdict
```text
REVERSED_SIGN_EFFECT = IMPROVED (Elbow steady error reduced to 0.4835 deg without any tracking feedback)
WORST STEADY-STATE JOINT = elbow_flex (0.4835 deg)
EXISTING_FILE_IMMUTABILITY = PASS (0 canonical files modified)
```

---

# Step B8D — Post-Patch Closed-Loop Tracking Hardware Validation Report

## 1. Purpose
This benchmark independently audits and validates the canonical production gravity patch on physical SO-101 hardware using the complete production controller stack:
* Canonical `BPDeltaController` ($K_{\text{ext}} = 0.5$, $q_{\text{corr}}\text{ clamp} = \pm 2.0^\circ$)
* Patched `GravityCompensator` (opposing reference dynamic support bias)
* Fixed nominal target: $q_{\text{nom}} = \text{physical\_start.json}$ ($[+1.4945^\circ, -83.5165^\circ, +79.8242^\circ, +45.3626^\circ, 0.0^\circ]$ URDF)
* Test Horizon: $90$ warmup ticks ($3.0\text{ s}$) followed by $300$ measured ticks ($10.0\text{ s}$) at $30.0\text{ Hz}$.

---

## 2. Phase 1 — Static Source & Decoupling Audit
* **Audited Dynamic Reference**: `[0.0, -0.615, -2.286, -0.703, 0.0]` deg
* **Audited Static Bias**: `[+0.264, 0.0, 0.0, 0.0, -0.352]` deg
* **Motor Bus PID**: $P=64, I=0, D=32$
* **$K_{\text{ext}}$ Decoupling**: Invariant verified ($d(b_{\text{support}}) / d(K_{\text{ext}}) = 0$).
* **Sign Guard**: Verified active in `GravityCompensator.__init__` (fails if reference compensation commands do not oppose gravity torque).
* **Unit Tests**: $70 / 70$ tests passed in unit/integration suite.

---

## 3. Phase 2 — Hardware Tracking Accuracy Decomposition

### 1. Full 10-Second Nominal Tracking Error ($q_{\text{actual}} - q_{\text{nom}}$, in deg):

| Joint | Signed Mean (deg) | Mean Abs (deg) | P50 Abs (deg) | P95 Abs (deg) | P99 Abs (deg) | Max Abs (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $+0.1518$ | $0.2731$ | $0.2637$ | $0.5275$ | $0.6154$ | $0.6154$ |
| **shoulder_lift** | $-0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ | $0.1319$ |
| **elbow_flex** | $-0.7473$ | **$0.7473$** | $0.7473$ | $0.7473$ | $0.7473$ | $0.7473$ |
| **wrist_flex** | $-0.3077$ | $0.3077$ | $0.3077$ | $0.3077$ | $0.3077$ | $0.3077$ |
| **wrist_roll** | $-0.0870$ | $0.0870$ | $0.0879$ | $0.0879$ | $0.0879$ | $0.0879$ |

---

### 2. Last 5-Second Steady-State Window (Final 150 Ticks):

| Joint | Signed Mean (deg) | Mean Abs (deg) | Max Abs (deg) | Net Drift (deg) |
| :--- | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $+0.1489$ | $0.2708$ | $0.6154$ | $-0.4396$ |
| **shoulder_lift** | $-0.1319$ | $0.1319$ | $0.1319$ | $+0.0000$ |
| **elbow_flex** | $-0.7473$ | **$0.7473$** | $0.7473$ | $+0.0000$ |
| **wrist_flex** | $-0.3077$ | $0.3077$ | $0.3077$ | $+0.0000$ |
| **wrist_roll** | $-0.0867$ | $0.0867$ | $0.0879$ | $+0.0000$ |

---

### 3. Old Production (Benchmark 07) vs Patched Production (B8D):

| Joint | Old Production Abs Error (deg) | New Patched Production Abs Error (deg) | Error Reduction (deg) |
| :--- | :---: | :---: | :---: |
| **shoulder_pan** | $0.1758$ | $0.2708$ | $-0.0950$ (Yaw static bias) |
| **shoulder_lift** | $0.5714$ | $0.1319$ | **$+0.4395$** (77% error drop) |
| **elbow_flex** | $2.2418$ | **$0.7473$** | **$+1.4945$** (67% error drop) |
| **wrist_flex** | $0.7473$ | $0.3077$ | **$+0.4396$** (59% error drop) |
| **wrist_roll** | $0.2637$ | $0.0867$ | **$+0.1770$** (67% error drop) |

---

## 4. Stability, Timing & Hardware Safety Audit
* `Elbow Error Reduction`: Dropped from $+2.2418^\circ \to -0.7473^\circ$ ($+1.4945^\circ$ net improvement).
* `Pitch Arm Drift`: $0.0000^\circ$ net drift across the final 5-second steady window on `shoulder_lift`, `elbow_flex`, `wrist_flex`.
* `Pitch Oscillations`: **0** sign changes across measured run on all pitch joints.
* `30 Hz Mean Loop Compute`: $2.7971\text{ ms}$ (P95 = $3.0006\text{ ms} \ll 33.333\text{ ms}$ budget).
* `Deadline Miss Count`: **0 / 300 (0.00%)**.
* `Motor Read/Write Failures`: **0**.
* `Safe/Hard Limit Violations`: **0**.
* `Benchmark-Specific PID Writes`: **0**.

---

## 5. Verdict
```text
PATCH_SOURCE_AUDIT = PASS
PATCH_DIRECTION_VALIDATION = PASS (All pitch joints demonstrate substantial error reduction and correct physical opposing direction)
TRACKING_ACCURACY = MEASURED (Elbow steady error = 0.7473 deg)
30HZ PERFORMANCE = PASS (P95 = 3.0006 ms)
HARDWARE SAFETY = PASS (0 violations, 0 failures)
```

---

# Step B8E — 5-cm Forward Motion Validation Entry

* **Run ID**: `run_20260814_163926`
* **Result Directory**: `/home/kimminje/Desktop/project/gopro_umi/5_benchmarks/08_gravity_tracking_ablation/results/B8E_forward_5cm/run_20260814_163926`
* **Run Status**: `COMPLETED`
* **Forward Frame / Vector**: `base_link` / `[1.0, 0.0, 0.0]` (`50.000 mm`)
* **Final TCP Position Error**: `3.7955 mm` (Forward progress = `51.74 mm`, Lateral = `3.37 mm`)
* **Elbow Steady Error (Final Hold)**: `0.3754 deg` (vs B8D start `0.7473 deg`, vs Old `2.2418 deg`)
* **Worst Joint Steady Error (Final Hold)**: `elbow_flex` (`0.3754 deg`)
* **Hardware Safety / Failures**: 0 failures, 0 limit violations
* **Verdicts**: Forward Motion = `PASS`, Dynamic Gravity = `STABLE`, 30Hz Performance = `PASS`
