# Benchmark 07 — Closed-Loop Tracking Validation Report

## 1. Purpose
This benchmark evaluates real-time closed-loop joint position tracking and steady-state gravity sag on the physical SO-101 follower arm during a 30-second fixed-target hold ($q_{\text{nom}} = \text{canonical physical start pose}$) under the production low-level controller:
* `BPDeltaController` ($K_{\text{ext}} = 0.5$, $q_{\text{corr}}\text{ clamp} = \pm 2.0^\circ$)
* `GravityCompensator` (URDF point-mass dynamic model with reference scaling)
* Invariant Nominal Target: $q_{\text{nom}} = \text{fixed canonical start pose}$ from `physical_start.json`.

---

## 2. Canonical Source Reused
* **Motor Bus IO**: `MotorIO` in [`4_deploy/control/motor_io.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/motor_io.py)
* **Joint Domain Mapping**: `JointMapping` in [`4_deploy/control/joint_mapping.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/joint_mapping.py)
* **Low-Level Controller**: `BPDeltaController` in [`4_deploy/control/bp_delta_controller.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/bp_delta_controller.py)
* **Gravity Compensation**: `GravityCompensator` in [`4_deploy/control/gravity_compensation.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/gravity_compensation.py)
* **Canonical Start Record**: [`4_deploy/config/physical_start.json`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/config/physical_start.json)

---

## 3. Controller Configuration & Telemetry
* **Motor Bus PID**: $P=64, I=0, D=32$
* **$K_{\text{ext}}$ Proportional Gain**: $0.5$
* **$q_{\text{corr}}$ Clamp Bound**: $\pm 2.0^\circ$
* **Static Support Bias**: $[+0.264^\circ, 0.0^\circ, 0.0^\circ, 0.0^\circ, +0.352^\circ]$
* **Gravity Reference Bias**: $[0.0^\circ, +0.615^\circ, +2.286^\circ, +0.703^\circ, 0.0^\circ]$
* **Fixed $q_{\text{nom}}$**: $[+1.4945^\circ, -83.5165^\circ, +79.8242^\circ, +45.3626^\circ, 0.0^\circ]$ (URDF)

---

## 4. Phase A — Start Pose Recovery & Settling
* **Initial Resting RAW**: `[-6.7692, -79.0769, 92.7912, 48.9231, 1.0989]` deg
* **Target Start RAW**: `[-7.1209, -79.4725, 81.8462, 46.4176, 1.3626]` deg
* **Controlled Ramp**: 50 steps over 3.0 seconds (**PASS**).
* **Settled RAW Pose**: `[-6.8571, -79.0769, 83.3846, 46.8132, 1.0989]` deg
* **Start Error to Target**: $[0.2637^\circ, 0.3956^\circ, 1.5385^\circ, 0.3956^\circ, 0.2637^\circ]$ (Max = $1.5385^\circ$).

---

## 5. Phase B — Tracking Error Decomposition (900 Ticks at 30 Hz)

### 1. Nominal Tracking Error ($q_{\text{actual}} - q_{\text{nom}}$) — Total Physical Sag (deg)

| Joint | Signed Mean | Mean Abs | P50 Abs | P95 Abs | P99 Abs | Max Abs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $+0.1758^\circ$ | $0.1758^\circ$ | $0.1758^\circ$ | $0.1758^\circ$ | $0.1758^\circ$ | $0.1758^\circ$ |
| **shoulder_lift** | $+0.5714^\circ$ | $0.5714^\circ$ | $0.5714^\circ$ | $0.5714^\circ$ | $0.5714^\circ$ | $0.5714^\circ$ |
| **elbow_flex** | $+2.2418^\circ$ | $2.2418^\circ$ | $2.2418^\circ$ | $2.2418^\circ$ | $2.2418^\circ$ | $2.2418^\circ$ |
| **wrist_flex** | $+0.7473^\circ$ | $0.7473^\circ$ | $0.7473^\circ$ | $0.7473^\circ$ | $0.7473^\circ$ | $0.7473^\circ$ |
| **wrist_roll** | $+0.2637^\circ$ | $0.2637^\circ$ | $0.2637^\circ$ | $0.2637^\circ$ | $0.2637^\circ$ | $0.2637^\circ$ |

---

### 2. Command Tracking Error ($q_{\text{actual}} - q_{\text{cmd}}$) — Motor Compliance (deg)

| Joint | Signed Mean | Mean Abs | P95 Abs | P99 Abs | Max Abs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $-0.0003^\circ$ | $0.0003^\circ$ | $0.0003^\circ$ | $0.0003^\circ$ | $0.0003^\circ$ |
| **shoulder_lift** | $+0.2572^\circ$ | $0.2572^\circ$ | $0.2572^\circ$ | $0.2572^\circ$ | $0.2572^\circ$ |
| **elbow_flex** | $+1.1001^\circ$ | $1.1001^\circ$ | $1.1001^\circ$ | $1.1001^\circ$ | $1.1001^\circ$ |
| **wrist_flex** | $+0.4524^\circ$ | $0.4524^\circ$ | $0.4524^\circ$ | $0.4524^\circ$ | $0.4524^\circ$ |
| **wrist_roll** | $+0.0436^\circ$ | $0.0436^\circ$ | $0.0436^\circ$ | $0.0436^\circ$ | $0.0436^\circ$ |

---

### 3. Controller Correction Applied ($q_{\text{cmd}} - q_{\text{nom}}$) — Controller Effort (deg)

| Joint | Signed Mean | Mean Abs | P95 Abs | P99 Abs | Max Abs | Clamp Hits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **shoulder_pan** | $+0.1761^\circ$ | $0.1761^\circ$ | $0.1761^\circ$ | $0.1761^\circ$ | $0.1761^\circ$ | 0 |
| **shoulder_lift** | $+0.3143^\circ$ | $0.3143^\circ$ | $0.3143^\circ$ | $0.3143^\circ$ | $0.3143^\circ$ | 0 |
| **elbow_flex** | $+1.1417^\circ$ | $1.1417^\circ$ | $1.1417^\circ$ | $1.1417^\circ$ | $1.1417^\circ$ | 0 |
| **wrist_flex** | $+0.2949^\circ$ | $0.2949^\circ$ | $0.2949^\circ$ | $0.2949^\circ$ | $0.2949^\circ$ | 0 |
| **wrist_roll** | $+0.2201^\circ$ | $0.2201^\circ$ | $0.2201^\circ$ | $0.2201^\circ$ | $0.2201^\circ$ | 0 |

---

## 6. Last 5-Second Steady-State Sag (Final 150 Ticks)

| Joint | Signed Sag | Mean Abs Error | Max Abs Error |
| :--- | :---: | :---: | :---: |
| **shoulder_pan** | $+0.1758^\circ$ | $0.1758^\circ$ | $0.1758^\circ$ |
| **shoulder_lift** | $+0.5714^\circ$ | $0.5714^\circ$ | $0.5714^\circ$ |
| **elbow_flex** | $+2.2418^\circ$ | **$2.2418^\circ$** | **$2.2418^\circ$** |
| **wrist_flex** | $+0.7473^\circ$ | $0.7473^\circ$ | $0.7473^\circ$ |
| **wrist_roll** | $+0.2637^\circ$ | $0.2637^\circ$ | $0.2637^\circ$ |

* **Worst Tracking Joint**: `elbow_flex` (Steady-state error = **$2.2418^\circ$**)
* **Net Drift Across 30s**: $[0.00^\circ, 0.00^\circ, 0.00^\circ, 0.00^\circ, 0.00^\circ]$ (Completely stable steady-state hold).

---

## 7. 30 Hz Real-Time Loop Timing Breakdown
* **Mean Compute Time**: $2.7917$ ms
* **P50 Compute Time**: $2.7389$ ms
* **P95 Compute Time**: **$2.9138$ ms**
* **P99 Compute Time**: $4.9481$ ms
* **Max Compute Time**: $6.3022$ ms
* **30 Hz Deadline Budget**: $33.333$ ms
* **Deadline Miss Count**: **0 / 900 (0.00%)**

---

## 8. Motor Write & Hardware Safety Audit
* `Start-Pose Ramp Writes`: **50**
* `Warmup Goal_Position Writes`: **90**
* `Measured Goal_Position Writes`: **900**
* `Initialization PID Writes`: **15**
* `Benchmark-Specific PID Writes`: **0**
* `Torque Physical Writes`: **0**
* `Gripper Physical Writes`: **0**
* `Safe/Hard Limit Violations (Phase B)`: **0**
* `Unexpected Command Aborts`: **0**

---

## 9. Limitations Note
> [!NOTE]
> **Static Target Scope:**
> Benchmark 07 evaluates position holding around a stationary nominal pose ($q_{\text{nom}} = \text{start pose}$). Dynamic tracking performance along fast time-varying Cartesian trajectories is separately characterized in deployment end-to-end trials.

---

## 10. Verdict
```text
CLOSED_LOOP_TRACKING_MEASUREMENT = COMPLETE
CANONICAL TRACKING THRESHOLD: NOT DEFINED IN CONFIG (Steady-state Sag: Worst Joint = elbow_flex at +2.2418 deg)
30 HZ DEADLINE = PASS (P95 = 2.9138 ms)
MOTOR COMMUNICATION = PASS (0 failures)
HARD-LIMIT SAFETY = PASS (0 violations)
NUMERICAL VALIDITY = PASS (0 NaN / Inf)
```
