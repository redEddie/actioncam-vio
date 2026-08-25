# Benchmark 01 — Remote Pipeline Latency Report

## 1. Purpose
Benchmark 01 validates the complete end-to-end latency of the remote SmolVLA inference pipeline used during live deployment:
* Local observation acquisition (GoPro RGB image snapshot + fresh FK robot state)
* Local observation preprocessing & payload packaging ($256 \times 256$ RGB + 6D Cartesian/Gripper state)
* Persistent SSH transport transmission to remote GPU worker
* Remote SmolVLA policy inference execution
* Network response return transmission
* Local action decoding & postprocessing into a ready $[15, 6]$ Cartesian delta action chunk.

---

## 2. Production Pipeline Under Test
* **Client / Transport**: `RemoteInferenceClient` & `SubprocessSSHTransport` in [`4_deploy/inference/remote_smolvla.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/inference/remote_smolvla.py)
* **Observation Engine**: `build_observation` & `state_from_motor_sample` in [`4_deploy/inference/observation.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/inference/observation.py)
* **Camera Capture**: `OpenCVLatestFrameCamera` in [`4_deploy/inference/camera.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/inference/camera.py)
* **Remote Host Target**: `kimminje@155.230.189.77` (Port `17970`)

---

## 3. Timing Boundaries
* **$T_0$ (Start Boundary)**: Fresh observation snapshot acquisition start / pipeline invocation.
* **$T_1$ (End Boundary)**: Validated local $[15, 6]$ action array available in memory for 10 Hz / 30 Hz trajectory reconstruction.
* **$\text{Total Latency} = T_1 - T_0$**.

---

## 4. Execution Contract & Action Specifications
* **Raw Wire Model Output**: Shape `(1, 15, 6)`
* **Local Postprocessed Action Chunk**: Shape `(15, 6)`
* **Action Horizon**: $15\text{ steps} \times 0.100\text{ s} = 1.5\text{ s}$ horizon at $10.0\text{ Hz}$
* **Action Feature Order**: $[dX, dY, dZ, d\text{Roll}, d\text{Pitch}, d\text{Gripper}]$ (Yaw is omitted from model actions)
* **Connection Mode**: Persistent SSH subprocess worker (zero per-request reconnection overhead)
* **Latency Budget Target**: $< 0.400\text{ s}$ ($< 400\text{ ms}$, corresponding to $< 4.0$ action steps at $10\text{ Hz}$)

---

## 5. Temporary Planning Value

> [!WARNING]
> **TEMPORARY PLANNING VALUE NOTICE:**
> **Status:** `NOT_MEASURED` (Hardware / Remote Network execution disabled for this scaffold step).
>
> * **Temporary Planning Total Latency**: **$0.350\text{ s}$ ($350.0\text{ ms}$)**
> * **Planning Action-Step Equivalent**: **$3.5\text{ steps}$ at $10\text{ Hz}$**
> * **Design Target**: $< 0.400\text{ s}$ ($< 400\text{ ms}$)
> * **Actual Measurement Status**: **NOT_RUN / PENDING**
>
> *This value is a temporary placeholder for pipeline planning and must not be cited as an empirically measured hardware result. When Benchmark 01 is executed live with the `--measure` flag, this section will be replaced with real empirical statistics.*

---

## 6. Current Benchmark Status
* **Benchmark Implementation**: `READY` (Scaffold, dry-run CLI, and persistent transport verified)
* **Remote Measurement**: `NOT_RUN`
* **Camera Access**: `NO`
* **SSH / Remote Access**: `NO`
* **Motor Bus Access**: `NO`
* **Actual Latency Verdict**: `PENDING`
