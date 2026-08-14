import pickle
import numpy as np

from pathlib import Path

def calculate_width():
    pkl_path = Path(__file__).resolve().parents[2] / "1_capture/actioncam-vio/calib_gripper_tags.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    distances = []
    
    for frame in data:
        tag_dict = frame['tag_dict']
        if 0 in tag_dict and 1 in tag_dict:
            tag_left = np.array(tag_dict[0]['tvec']).flatten()
            tag_right = np.array(tag_dict[1]['tvec']).flatten()
            dist = np.linalg.norm(tag_left - tag_right)
            distances.append(dist)

    if not distances:
        print("No valid frames found.")
        return

    distances = np.sort(distances)
    n = len(distances)
    
    min_dist = np.percentile(distances, 1)
    max_dist = np.percentile(distances, 99)
    
    print(f"Total valid frames: {n}")
    print(f"Camera Perceived Min Width: {min_dist:.5f} m ({min_dist*100:.2f} cm)")
    print(f"Camera Perceived Max Width: {max_dist:.5f} m ({max_dist*100:.2f} cm)")

if __name__ == '__main__':
    calculate_width()
