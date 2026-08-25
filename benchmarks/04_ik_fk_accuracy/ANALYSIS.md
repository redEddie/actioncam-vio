# Benchmark 04 — IK / FK Accuracy Validation Report

## 1. Purpose
This benchmark strictly evaluates whether the canonical inverse kinematics solver (`DLSInverseKinematicsV7`) achieves accurate, consistent, and continuous 5D kinematics solutions (`XYZ + Roll/Pitch` with free Yaw) within the robot's safe operating limits, verified via full forward kinematics reconstruction ($q \rightarrow \text{FK} \rightarrow \text{IK} \rightarrow \text{FK}$).

---

## 2. Canonical Source Inspected
* **IK Solver**: `DLSInverseKinematicsV7` in [`4_deploy/ik/ik_solver_v7.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/ik_solver_v7.py)
* **Kinematics Engine**: `lerobot.model.kinematics.RobotKinematics` with [`4_deploy/ik/urdf/so_arm_with_gopro_final.urdf`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/urdf/so_arm_with_gopro_final.urdf)
* **Safety Gates**: `require_ik_success`, `require_joint_limits` in [`4_deploy/control/safety.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/safety.py)

---

## 3. Solver Contract & Configuration
* **Arm Joint Order**: `("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")`
* **Input Position Units**: meters ($X, Y, Z$)
* **Input Orientation Units**: $3 \times 3$ Rotation Matrix (representing desired global Roll and Pitch, while Yaw is free)
* **Output Joint Units**: radians (`float64 [5]`)
* **Maximum Iterations**: `100`
* **Position Convergence Tolerance**: `5e-4` m ($0.5$ mm)
* **Orientation Convergence Tolerance**: `1e-3` rad ($0.057^\circ$)
* **Damping Constant**: $\lambda = 0.01$ (adaptive DLS)
* **Physical Hard Limits**:
  * `shoulder_pan`: $[-1.91986, 1.91986]$ rad ($\pm 110.0^\circ$)
  * `shoulder_lift`: $[-1.74533, 1.74533]$ rad ($\pm 100.0^\circ$)
  * `elbow_flex`: $[-1.69000, 1.69000]$ rad ($\pm 96.8^\circ$)
  * `wrist_flex`: $[-1.65806, 1.65806]$ rad ($\pm 95.0^\circ$)
  * `wrist_roll`: $[-2.74385, 2.84121]$ rad ($[-157.2^\circ, +162.8^\circ]$)
* **Safe Operating Limits (90% envelope)**:
  * `shoulder_pan`: $\pm 99.0^\circ$
  * `shoulder_lift`: $\pm 90.0^\circ$
  * `elbow_flex`: $\pm 87.1^\circ$
  * `wrist_flex`: $\pm 85.5^\circ$
  * `wrist_roll`: $[-141.2^\circ, +146.8^\circ]$

---

## 4. Test A — Exact-Seed Reconstruction (500 Samples)
Evaluates solver precision and unit/convention consistency when initialized at the exact reference joint pose ($N=500$ safe-interior configurations).
* **IK Success Rate**: **100.0%** (500 / 500)
* **Position Error**: Mean = `0.0000` mm, Max = `0.0000` mm
* **Orientation Error**: Mean = `0.0000` deg, Max = `0.0000` deg
* **IK Runtime**: Mean = `0.1167` ms, P95 = `0.1231` ms, Max = `0.4565` ms

---

## 5. Test B — Perturbed-Seed Reconstruction (500 Samples, $\pm 2^\circ$ Noise)
Evaluates convergence robustness when seeded with random bounded angular noise ($\pm 2.0^\circ$).
* **IK Success Rate**: **100.0%** (500 / 500)
* **Position Error**: Mean = `0.1678` mm, P95 = `0.4273` mm, Max = `0.4956` mm ($< 0.5$ mm tolerance)
* **Orientation Error**: Mean = `0.0127` deg, P95 = `0.0368` deg, Max = `0.0552` deg ($< 0.057^\circ$ tolerance)
* **IK Runtime**: Mean = `0.4766` ms, P95 = `0.7512` ms, Max = `0.8441` ms

---

## 6. Test C — Smooth Trajectory Continuity (300 Steps at 30 Hz)
Evaluates sequential tracking and branch-switching behavior across a continuous bounded 3D trajectory ($\pm 8$ mm position variation, $\pm 1.5^\circ$ orientation variation) with warm-started previous solution seeds.
* **IK Success Rate**: **100.0%** (300 / 300)
* **Position Error**: Mean = `0.0099` mm, P95 = `0.0182` mm, Max = `0.0446` mm
* **Orientation Error**: Mean = `0.0004` deg, P95 = `0.0006` deg, Max = `0.0006` deg
* **IK Runtime**: Mean = `0.4361` ms, P95 = `0.4534` ms, Max = `1.3783` ms
* **Large Joint Jumps ($> 10^\circ$)**: **0**

---

## 7. Joint Continuity Summary (Test C)

| Joint | Mean $\Delta$deg | P95 $\Delta$deg | P99 $\Delta$deg | Max $\Delta$deg |
| :--- | :---: | :---: | :---: | :---: |
| **shoulder_pan** | 0.0284 | 0.0446 | 0.0454 | 0.0454 |
| **shoulder_lift** | 0.1433 | 0.2250 | 0.2294 | 0.2391 |
| **elbow_flex** | 0.2570 | 0.4411 | 0.4511 | 0.4624 |
| **wrist_flex** | 0.1642 | 0.2957 | 0.3037 | 0.3042 |
| **wrist_roll** | 0.0502 | 0.0783 | 0.0785 | 0.0785 |
| **VECTOR NORM** | **0.3509** | **0.5753** | **0.5887** | **0.6017** |

---

## 8. Failure Audit & Safety Verification
* `Hard Limit Violations`: **0**
* `Safe Limit Violations`: **0**
* `IK Non-finite (NaN / Inf)`: **0**
* `FK Non-finite (NaN / Inf)`: **0**
* `Solver-success but FK tolerance failure`: **0**
* `Catastrophic Branch Jumps`: **0**

---

## 9. Hardware, Remote & Immutability Audit
* `MOTOR ACCESSED`: **NO**
* `PRESENT_POSITION READ`: **NO**
* `MOTOR WRITE`: **0**
* `CAMERA ACCESSED`: **NO**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**
* `EXISTING FILES MODIFIED/DELETED`: **0**

---

## 10. Limitation Note
> [!NOTE]
> **FK Self-Consistency Scope Limitation:**
> Primary validation tests verify mathematical and numerical consistency between `DLSInverseKinematicsV7.solve` and `RobotKinematics.forward_kinematics`. Physical URDF-to-RAW joint domain mapping accuracy is independently validated in Benchmark 09.

---

## 11. Verdict
```text
IK_FK_ACCURACY = PASS
```
*(Test A: 100% PASS, Test B: 100% PASS, Test C: 100% PASS, Zero limit violations, Zero branch jumps)*
