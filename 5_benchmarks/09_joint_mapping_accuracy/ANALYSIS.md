# Benchmark 09 — Joint Mapping Accuracy Validation Report

## 1. Purpose
This benchmark evaluates whether the canonical joint domain transformations between Motor RAW degrees, URDF joint degrees, and URDF joint radians strictly satisfy the canonical contract equations ($RAW = URDF \times sign + encoder\_zero$).
The test runs purely in local memory with **Zero hardware/motor access, Zero network/SSH calls**, using independent mathematical oracles.

---

## 2. Canonical Source Inspected
* **Joint Mapping Implementation**: `JointMapping`, `degrees_to_radians`, `radians_to_degrees` in [`4_deploy/control/joint_mapping.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/joint_mapping.py)
* **Authoritative Calibration Records**: [`4_deploy/config/motor_mapping.json`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/config/motor_mapping.json) and [`4_deploy/config/physical_start.json`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/config/physical_start.json)
* **Kinematics Domain**: `DLSInverseKinematicsV7` in [`4_deploy/ik/ik_solver_v7.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/ik_solver_v7.py)
* **Gravity Domain**: `GravityCompensator` in [`4_deploy/control/gravity_compensation.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/gravity_compensation.py)

---

## 3. Joint Mapping Contract & Constants
* **Joint Count**: 5 arm joints
* **Joint Order**: `("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")`
* **Sign Vector**: `[+1.0, +1.0, +1.0, +1.0, +1.0]`
* **Encoder Zero Degrees (Full Precision)**:
  * `shoulder_pan`: `-8.615384615384615` deg
  * `shoulder_lift`: `+4.043956043956044` deg
  * `elbow_flex`: `+2.021978021978022` deg
  * `wrist_flex`: `+1.054945054945055` deg
  * `wrist_roll`: `+1.3626373626373627` deg

---

## 4. Known Canonical Start Pose Validation
Evaluated against `physical_start.json`:
* **RAW Reference**: `[-7.120879, -79.472527, +81.846154, +46.417582, +1.362637]` deg
* **URDF Reference**: `[+1.494505, -83.516484, +79.824176, +45.362637, 0.000000]` deg
* **RAW $\rightarrow$ URDF Max Error**: `0.00e+00` deg
* **URDF $\rightarrow$ RAW Max Error**: `0.00e+00` deg

---

## 5. 1000-Sample Oracle & Round-Trip Accuracy Summary

| Metric | Sample Count | Mean Abs Err (deg) | P50 (deg) | P95 (deg) | P99 (deg) | Max Abs Err (deg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAW $\rightarrow$ URDF vs Oracle** | 5000 | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **URDF $\rightarrow$ RAW vs Oracle** | 5000 | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **RAW Round-Trip ($R \rightarrow U \rightarrow R$)** | 5000 | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **URDF Round-Trip ($U \rightarrow R \rightarrow U$)** | 5000 | `1.52e-16` | `0.00e+00` | `0.00e+00` | `7.11e-15` | `1.42e-14` |

---

## 6. Per-Joint Max Error Breakdown

| Joint | RAW $\rightarrow$ URDF Max (deg) | URDF $\rightarrow$ RAW Max (deg) | Round-Trip Max (deg) |
| :--- | :---: | :---: | :---: |
| **shoulder_pan** | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **shoulder_lift** | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **elbow_flex** | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **wrist_flex** | `0.00e+00` | `0.00e+00` | `0.00e+00` |
| **wrist_roll** | `0.00e+00` | `0.00e+00` | `0.00e+00` |

---

## 7. Domain & Cross-Source Audit
* **IK/FK Joint Input Domain**: `URDF radians` (**PASS**)
* **Gravity Joint Input Domain**: `URDF radians` (**PASS**)
* **Double Encoder Offset**: **NO** (Offset subtraction occurs exclusively in RAW $\rightarrow$ URDF mapping; Gravity model accepts mapped URDF radians without reapplying offsets) (**PASS**)
* **Joint Order Consistency**: `JointMapping`, `DLSInverseKinematicsV7`, and `GravityCompensator` all share the identical 5-joint ordering (**PASS**)
* **Gripper Exclusion**: Gripper is strictly isolated from 5D arm kinematics/mapping (**PASS**)

---

## 8. Hardware, Remote & Immutability Audit
* `MOTOR ACCESSED`: **NO**
* `PRESENT_POSITION READ`: **NO**
* `MOTOR WRITE`: **0**
* `CAMERA ACCESSED`: **NO**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**
* `EXISTING FILES MODIFIED/DELETED`: **0**

---

## 9. Limitations Note
> [!NOTE]
> **Physical Calibration Scope:**
> Benchmark 09 validates software coordinate transformation consistency against canonical calibration records. It does not measure physical mechanical zero-point calibration on the actual motor hardware.

---

## 10. Verdict
```text
JOINT_MAPPING_ACCURACY = PASS
```
*(All 1000 oracle tests, known start poses, round-trips, and cross-source domain checks passed with machine epsilon precision)*
