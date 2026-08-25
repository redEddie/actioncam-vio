import os
import re

candidates = [
    "deploy_smolvla_yawfree.py",
    "deploy_smolvla_yawfree_microstep.py",
    "deploy_smolvla_yawfree_delta.py",
    "test_remote_inference_pipeline.py",
    "step8_31b_true_latency.py"
]

for c in candidates:
    if not os.path.exists(c):
        continue
    with open(c, 'r') as f:
        content = f.read()
    
    print(f"--- {c} ---")
    print(f"CAMERA: {'cap.read' in content or 'cv2.VideoCapture' in content or 'v4l2src' in content}")
    print(f"STATE: {'bus.sync_read' in content or 'Present_Position' in content}")
    print(f"REMOTE INFERENCE: {'inference' in content and ('ssh' in content or 'socket' in content or 'paramiko' in content or 'ssh' in c)}")
    print(f"15x6 / 15-step: {'15' in content and '6' in content}")
    print(f"INCREMENTAL: {'cumulative' in content or '+=' in content or 'incremental' in content.lower()}")
    print(f"REANCHOR: {'reanchor' in content.lower() or 'arrival' in content.lower()}")
    print(f"OVERLAP: {'overlap' in content.lower() or 'ensemble' in content.lower() or 'scheduler' in content.lower()}")
    print(f"10HZ: {'10' in content or '0.1' in content}")
    print(f"30HZ / IK / BP CONTROLLER: {'ik_solver' in content.lower() and '30' in content and 'q_corr' in content.lower()}")
    print(f"LOOP: {'while True' in content or 'for' in content}")
    
    old_contract = re.search(r'chunk_size\s*=\s*50', content)
    if old_contract:
        print(f"LEGACY_CONTRACT_FOUND: {old_contract.group(0)}")
        
    old_contract_60 = re.search(r'60Hz', content)
    if old_contract_60:
        print(f"LEGACY_CONTRACT_FOUND: 60Hz")
