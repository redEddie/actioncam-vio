# Benchmark 06 — Motor Bus Latency Validation Report

## 1. Purpose
This benchmark evaluates real-time serial bus communication latency on physical SO-101 arm hardware:
* `Present_Position` Sync Read Latency
* `Goal_Position` Sync Write Latency
* Full Read $\rightarrow$ Minimal Zero-Delta Compute $\rightarrow$ Write Cycle Latency compared against the $30\text{ Hz}$ control budget ($33.333\text{ ms}$).

---

## 2. Canonical Source Inspected
* **Motor Bus IO**: `MotorIO` in [`4_deploy/control/motor_io.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/motor_io.py)
* **Joint Domain Mapping**: `JointMapping` in [`4_deploy/control/joint_mapping.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/joint_mapping.py)
* **Kinematics & Safe Limits**: `DLSInverseKinematicsV7` in [`4_deploy/ik/ik_solver_v7.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/ik_solver_v7.py)
* **Canonical Start Pose**: [`4_deploy/config/physical_start.json`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/config/physical_start.json)

---

## 3. Phase A — Start Pose Recovery & Settling
1. **Pre-Move State**:
   * Initial Physical RAW Pose: `[6.7692, -96.6593, 93.3187, 80.8352, -2.2418]` deg
   * Initial URDF Pose: `[15.3846, -100.7033, 91.2967, 79.7802, -3.6044]` deg
   * Target Canonical Start RAW: `[-7.1209, -79.4725, 81.8462, 46.4176, 1.3626]` deg
   * Target Canonical Start URDF: `[1.4945, -83.5165, 79.8242, 45.3626, 0.0000]` deg
2. **Recovery Direction Audit**:
   * `shoulder_lift` moved from $-100.70^\circ$ toward $-83.52^\circ$ ($\Delta = +17.19^\circ$, strictly inward toward safe envelope).
   * `elbow_flex` moved from $+91.30^\circ$ toward $+79.82^\circ$ ($\Delta = -11.47^\circ$, strictly inward toward safe envelope).
   * Monotonic bounded ramp verified across 50 discrete steps over 3.0 seconds (**PASS**).
3. **Settled State**:
   * Settled RAW: `[-7.0330, -79.3407, 83.2088, 46.8132, 1.0989]` deg
   * Max joint error to target: `1.3626` deg (**SETTLED = YES**)

---

## 4. Phase B — Latency Metrics Breakdown (900 Measured Cycles at 30 Hz)

| Metric | Count | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Present_Position Read** | 900 | 0.9907 | 0.9734 | 1.0470 | 1.3095 | 3.3821 |
| **Goal_Position Write** | 900 | 0.1189 | 0.1019 | 0.1560 | 0.2274 | 3.5524 |
| **Full Read $\rightarrow$ Write Cycle** | 900 | **1.1132** | **1.0817** | **1.1926** | **2.3688** | **4.5622** |

---

## 5. 30 Hz Control Budget & Deadline Audit
* **30 Hz Period Budget**: $33.333333$ ms
* **Measured Full Cycle P95**: **$1.1926$ ms** ($< 3.6\%$ of the total available time budget)
* **Deadline Misses**: **0 / 900 (0.00%)**
* **Worst Overrun**: $0.0000$ ms
* **Verdict**: **PASS**

---

## 6. Physical Movement & Zero-Delta Audit (Phase B)
* **Command Policy**: Fresh `Present_Position` copied to `Goal_Position` on each tick (intentional displacement = $0^\circ$).
* **Zero-Delta Guard**: Verified exact match on all motors before each sync-write (**PASS**).
* **Physical Drift During 30-Second Measurement**:
  * `shoulder_pan`: Net drift = $0.0000^\circ$, Max deviation = $0.0000^\circ$
  * `shoulder_lift`: Net drift = $0.0000^\circ$, Max deviation = $0.0000^\circ$
  * `elbow_flex`: Net drift = $0.0000^\circ$, Max deviation = $0.0000^\circ$
  * `wrist_flex`: Net drift = $0.0000^\circ$, Max deviation = $0.0000^\circ$
  * `wrist_roll`: Net drift = $0.0000^\circ$, Max deviation = $0.0000^\circ$

---

## 7. Motor Write & Hardware Safety Audit
* `START-POSE Goal_Position writes`: **50** (Controlled ramp)
* `LATENCY ZERO-DELTA Goal_Position writes`: **990** (90 warmup + 900 measured)
* `Gripper physical writes`: **0**
* `PID physical writes`: **15** (5 joints $\times$ 3 registers on initial connect verification)
* `Torque physical writes`: **0**
* `CAMERA ACCESSED`: **NO**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**
* `EXISTING FILES MODIFIED/DELETED`: **0**

---

## 8. Limitations Note
> [!NOTE]
> **Measurement Scope:**
> This benchmark validates serial bus transmission latency for `sync_read` and `sync_write` operations. It does not measure closed-loop trajectory tracking errors under active kinematic motion, which are evaluated in Benchmark 07 and Benchmark 08.

---

## 9. Verdict
```text
MOTOR_BUS_LATENCY = PASS
```
*(Full bus cycle P95 = 1.1926 ms << 33.333 ms budget; zero deadline misses across 900 cycles)*
