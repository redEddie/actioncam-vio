import os
import subprocess

def test_script():
    cmd = ["python3", "/home/kimminje/Desktop/project/gopro_umi/4_deploy/step8_31_latency_benchmark.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0

if __name__ == "__main__":
    test_script()
    print("Static test passed.")
