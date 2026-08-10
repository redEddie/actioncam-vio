import time
import numpy as np
import cv2
from pathlib import Path
import sys
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
sys.path.insert(0, str(PROJECT_ROOT / '4_deploy'))

# Import the existing module to reuse the solver loading logic
import deploy_smolvla_yawfree_delta as deploy

def measure_image_preprocessing():
    # GoPro 1080p dummy image
    img_bgr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    times = []
    # Warmup
    for _ in range(10):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        
    for _ in range(200):
        start = time.perf_counter()
        # 1. BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 2. Resize to model input size
        img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        times.append(time.perf_counter() - start)
        
    print(f"[Image Preprocessing] 1080p -> 256x256 RGB")
    print(f" - Average : {np.mean(times)*1000:.3f} ms")
    print(f" - Max     : {np.max(times)*1000:.3f} ms")
    print(f" - Min     : {np.min(times)*1000:.3f} ms\n")

def measure_ik_solver():
    print("Loading IK Solver...")
    solver = deploy.load_calibrated_solver()
    
    current_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    current_pose = solver.forward_kinematics(current_q)
    target_pos = current_pose[:3, 3]
    target_rot = current_pose[:3, :3]
    
    # Warmup
    for _ in range(10):
        solver.solve(target_pos, target_rot, current_q)
        
    times = []
    # Measure 200 IK solves with small random position changes (Delta ~ 1cm)
    for _ in range(200):
        noise = np.random.normal(0, 0.01, 3) # 1cm noise
        new_target_pos = target_pos + noise
        
        start = time.perf_counter()
        solved_q = solver.solve(new_target_pos, target_rot, current_q)
        times.append(time.perf_counter() - start)
        
        # update current_q to simulate continuous movement
        current_q = solved_q
        
    print(f"[IK Solver (DLS V7)]")
    print(f" - Average : {np.mean(times)*1000:.3f} ms")
    print(f" - Max     : {np.max(times)*1000:.3f} ms")
    print(f" - Min     : {np.min(times)*1000:.3f} ms")

if __name__ == '__main__':
    measure_image_preprocessing()
    measure_ik_solver()
