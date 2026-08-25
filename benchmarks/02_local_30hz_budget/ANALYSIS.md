# Benchmark 02 — Local 30 Hz Computation Budget Validation Report

## 1. Purpose
This benchmark strictly validates whether the canonical local computation chain executing on the real robot host satisfies the 30 Hz control cycle budget (33.333 ms / tick).
The test reads live `Present_Position` from the physical SO-101 robot on `/dev/ttyACM0` in **strict READ-ONLY mode** (Zero motor writes, Zero Torque/PID modifications, No SSH/Remote inference).

---

## 2. Reused Canonical Implementation
* **Joint Mapping / Domain Conversion**: `JointMapping` in [`4_deploy/control/joint_mapping.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/joint_mapping.py)
* **Observation / FK & Yaw Preparation**: `state_from_motor_sample` in [`4_deploy/inference/observation.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/inference/observation.py)
* **Inverse Kinematics**: `DLSInverseKinematicsV7` in [`4_deploy/ik/ik_solver_v7.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/ik_solver_v7.py)
* **Gravity Compensation**: `GravityCompensator` in [`4_deploy/control/gravity_compensation.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/gravity_compensation.py)
* **BP+Delta Controller**: `BPDeltaController` in [`4_deploy/control/bp_delta_controller.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/bp_delta_controller.py)
* **Safety Gates**: `require_ik_success`, `require_joint_limits` in [`4_deploy/control/safety.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/safety.py)
* **Scheduler / Target Sampling**: `ChunkScheduler` in [`4_deploy/trajectory/chunk_scheduler.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/trajectory/chunk_scheduler.py)
* **Trajectory Interpolation**: `sample_trajectory_at_time` in [`4_deploy/trajectory/interpolate.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/trajectory/interpolate.py)
* **Motor IO & Bus Guard**: `MotorIO` & `_ReadOnlyBusGuard` in [`4_deploy/control/motor_io.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/control/motor_io.py)

---

## 3. Input Source & Timing Boundary
* **Motor State**: Live SO-101 physical follower Present_Position read from `/dev/ttyACM0` (Read time is measured separately and excluded from `FULL_LOCAL_COMPUTE`).
* **Stimulus Trajectory**: Deterministic local validation trajectory (15-step incremental window at 10 Hz) anchored at the live robot pose.
* **Timing Boundary**:
  * **T0**: Instant after `Present_Position` dictionary is available in local memory.
  * **Stages**: RAW→URDF $\rightarrow$ Scheduler query $\rightarrow$ 30 Hz Interpolation $\rightarrow$ Actual FK/Yaw $\rightarrow$ DLS IK $\rightarrow$ Gravity compensation $\rightarrow$ BP+delta calculation $\rightarrow$ Hard safety limit check $\rightarrow$ Preview tick generation.
  * **T1**: Instant after safe `q_cmd` tick preview is generated.
  * **FULL_LOCAL_COMPUTE** = $T1 - T0$.

---

## 4. Stage Timing Breakdown (ms)

| Stage | Count | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAW→URDF** | 900 | 0.0333 | 0.0309 | 0.0519 | 0.0686 | 0.1061 |
| **Scheduler** | 900 | 0.0207 | 0.0204 | 0.0288 | 0.0376 | 0.0652 |
| **Interpolation** | 900 | 0.0003 | 0.0003 | 0.0005 | 0.0007 | 0.0011 |
| **Actual FK/Yaw** | 900 | 0.1986 | 0.1979 | 0.2620 | 0.3778 | 0.5433 |
| **IK (DLS 5D)** | 900 | 0.6115 | 0.5939 | 0.7083 | 1.0377 | 1.8231 |
| **Gravity** | 900 | 0.6020 | 0.5938 | 0.6656 | 1.0166 | 1.7370 |
| **BP+delta** | 900 | 0.5707 | 0.5614 | 0.6505 | 0.9791 | 4.0142 |
| **Safety** | 900 | 0.0391 | 0.0380 | 0.0445 | 0.0688 | 0.1235 |

*Informational: Motor Present_Position Read Latency: Mean = 1.9448 ms, P95 = 2.0526 ms.*

---

## 5. Full Local Compute Distribution & Deadline Analysis

* **Sample Count**: 900 measured ticks (excluding 90 warmup ticks)
* **Mean**: 2.0763 ms
* **P50**: 2.0258 ms
* **P95**: **2.2897 ms** *(Required: < 33.333 ms)*
* **P99**: 3.4062 ms
* **Max**: 6.8192 ms
* **Control Budget**: 33.3333 ms
* **Deadline Miss Count**: 0 / 900
* **Deadline Miss Rate**: 0.00%
* **Worst Overrun**: 0.0000 ms

---

## 6. Correctness Counters & Hardware Safety Audit

### Correctness Counters
* Present_Position read failures: `0`
* RAW→URDF nonfinite: `0`
* FK failures: `0`
* IK attempts: `990`
* IK success: `990`
* IK failures: `0`
* IK nonfinite: `0`
* Physical limit violations: `0`
* Safe limit violations: `0`
* Gravity nonfinite: `0`
* Controller nonfinite: `0`
* Safety rejects: `0`

### Hardware & Network Audit
* `MOTOR ACCESSED`: **YES** (Live Present_Position read)
* `MOTOR READ ONLY`: **YES**
* `Goal_Position physical writes`: **0**
* `PID physical writes`: **0**
* `Torque physical writes`: **0**
* `Other register physical writes`: **0**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**

---

## 7. Operating Cautions & Space Boundary Notes
> [!CAUTION]
> **Space / Joint Boundary 이탈 시 연산 지연 주의사항:**
> - 로봇의 관절 위치 또는 목표 궤적이 DLS IK의 **안전 관절 한계 범위(90% Safe Limits / Space Boundary)**를 벗어나거나 특이점(Singularity) 근처에 위치할 경우, IK 솔버가 조기 수렴하지 못하고 최대 반복 횟수(`max_iter=100`)까지 루프를 돌게 됩니다.
> - 이 경우 틱당 IK 계산 시간만 **35 ms 이상**으로 급증하여 30 Hz 제어 예산(**33.333 ms Deadline**)을 초과(Deadline Miss)할 수 있으므로, 로봇의 작업 공간(Workspace/Boundary) 내에서 시작 및 궤적 추종이 이루어지도록 주의해야 합니다.

---

## 8. Verdict
```text
LOCAL_30HZ_BUDGET = PASS
```
*(P95 = 2.2897 ms << 33.333 ms; Deadline miss count = 0 / 900)*

