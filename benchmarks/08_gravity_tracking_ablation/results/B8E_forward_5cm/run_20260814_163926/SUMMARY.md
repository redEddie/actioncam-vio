# Step B8E — 5-cm Forward Motion Validation Summary Report

**Run ID:** `run_20260814_163926`  
**Timestamp:** `20260814_163926`  
**Run Status:** `COMPLETED`  

## 1. Setup & Forward Contract
* **Forward Frame / Axis**: `base_link` / `+X` (Vector: `[1.0, 0.0, 0.0]`)
* **Requested Distance**: `50.0 mm`
* **Controller**: $K_{\text{ext}} = 0.5$, Clamp = $\pm 2.0^\circ$, PID = ${'p': 64, 'i': 0, 'd': 32}$
* **Dynamic Reference Support**: `[0.0, -0.615, -2.286, -0.703, 0.0]` deg
* **Static Support**: `[0.264, 0.0, 0.0, 0.0, -0.352]` deg

## 2. TCP Tracking Results
* **Start TCP**: `[0.25244257789988805, -0.005748999132233442, 0.030660565148016933]` m
* **Target TCP**: `[0.30244257789988804, -0.005748999132233442, 0.030660565148016933]` m
* **Final Actual TCP**: `[0.3041860248904897, -0.005876525616282744, 0.03402953851990638]` m
* **Position Error Norm**: `3.7955 mm`
* **Forward Progress**: `51.7434 mm` (Error: `+1.7434 mm`)
* **Lateral Error Norm**: `3.3714 mm`
* **Final Orientation Error**: Mean = `0.8627^\circ`, Max = `0.8627^\circ`

## 3. Joint Tracking Error Decomposition
### Move Phase (3.0s, 90 ticks):
| Joint | Signed Mean (deg) | Mean Abs (deg) | P95 Abs (deg) | Max Abs (deg) |
| :--- | :---: | :---: | :---: | :---: |
| shoulder_pan | +0.1746 | 0.1746 | 0.3355 | 0.3607 |
| shoulder_lift | -0.9323 | 0.9323 | 1.5946 | 1.6368 |
| elbow_flex | -0.0379 | 0.2781 | 0.6593 | 0.6593 |
| wrist_flex | +0.1781 | 0.3102 | 0.6597 | 0.7945 |
| wrist_roll | -0.0879 | 0.0879 | 0.0879 | 0.0879 |

### Final Hold Phase (5.0s, 150 ticks):
| Joint | Signed Mean (deg) | Mean Abs (deg) | Max Abs (deg) | Net Drift (deg) |
| :--- | :---: | :---: | :---: | :---: |
| shoulder_pan | +0.0176 | 0.0176 | 0.0176 | +0.0000 |
| shoulder_lift | -0.1763 | 0.1763 | 0.1763 | +0.0000 |
| elbow_flex | -0.3754 | 0.3754 | 0.3754 | +0.0000 |
| wrist_flex | -0.2835 | 0.2835 | 0.2835 | +0.0000 |
| wrist_roll | -0.0879 | 0.0879 | 0.0879 | +0.0000 |

## 4. Elbow Performance Comparison
* **Old Production Static Sag (Bench 07)**: `+2.2418 deg`
* **Corrected Production Static Sag (Bench B8D)**: `0.7473 deg`
* **B8E Start Hold Mean Abs Error**: `0.6505 deg`
* **B8E Move Phase Mean Abs Error**: `0.2781 deg` (Max = `0.6593 deg`)
* **B8E Final Hold Mean Abs Error**: `0.3754 deg` (Max = `0.3754 deg`)

## 5. Dynamic Gravity Support Range during Move
| Joint | Start Support (deg) | Min Support (deg) | Max Support (deg) | Final Support (deg) |
| :--- | :---: | :---: | :---: | :---: |
| shoulder_pan | +0.2640 | +0.2640 | +0.2640 | +0.2640 |
| shoulder_lift | -0.6192 | -0.9225 | -0.6192 | -0.9225 |
| elbow_flex | -2.2922 | -2.3014 | -2.1848 | -2.1864 |
| wrist_flex | -0.7131 | -0.7139 | -0.7042 | -0.7107 |
| wrist_roll | -0.3520 | -0.3520 | -0.3520 | -0.3520 |

## 6. Verdicts
* **FORWARD_MOTION_EXECUTION**: `PASS`
* **IK_TRAJECTORY**: `PASS`
* **GRAVITY_DYNAMIC_BEHAVIOR**: `STABLE`
* **TCP_TARGET_ACCURACY**: `MEASURED (3.80 mm)`
* **JOINT_TRACKING_ACCURACY**: `MEASURED (Elbow final hold = 0.3754 deg)`
* **30HZ_PERFORMANCE**: `PASS (P95 = 2.9591 ms)`
* **HARDWARE_SAFETY**: `PASS`
