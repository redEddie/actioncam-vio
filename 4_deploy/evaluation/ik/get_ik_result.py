def main():
    import numpy as np
    import sys
    from pathlib import Path
    
    sys.path.insert(0, '/home/kimminje/Desktop/project/gopro_umi/4_deploy')
    from ik.ik_solver_v7 import DLSInverseKinematicsV7
    ik_solver = DLSInverseKinematicsV7(urdf_path="/home/kimminje/Desktop/project/gopro_umi/4_deploy/ik/urdf/so_arm_with_gopro_final.urdf")
    
    ENCODER_ZERO_DEG = np.array([-8.6154, 4.0440, 2.0220, 1.0549, 1.3626], dtype=np.float64)
    SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    RAW_FRONT_BASELINE = np.array([-0.5000, -70.3950, 58.2320, 60.9529, 1.3626], dtype=np.float64)
    q_seed_urdf = (RAW_FRONT_BASELINE - ENCODER_ZERO_DEG) * SIGN
    q_seed_rad = np.deg2rad(q_seed_urdf)
    
    T = ik_solver.forward_kinematics(q_seed_rad)
    xyz_mm_0 = T[:3, 3] * 1000.0
    rot_0 = T[:3, :3]
    
    target_xyz = xyz_mm_0 + np.array([50.0, 0.0, 0.0])
    q_sol_rad = ik_solver.solve(target_xyz / 1000.0, rot_0, q_seed_rad)
    q_sol_rad[0] = q_seed_rad[0]
    q_urdf_final = np.rad2deg(q_sol_rad)
    
    T_final = ik_solver.forward_kinematics(q_sol_rad)
    xyz_final = T_final[:3, 3] * 1000.0
    dx = xyz_final[0] - xyz_mm_0[0]
    dy = xyz_final[1] - xyz_mm_0[1]
    dz = xyz_final[2] - xyz_mm_0[2]
    
    from scipy.spatial.transform import Rotation as R
    ori_diff = R.from_matrix(T_final[:3, :3].dot(rot_0.T)).as_euler('ZYX', degrees=True)
    
    print("Q_NOM_ENDPOINT URDF:", q_urdf_final.tolist())
    print(f"DX: {dx:.2f}, DY: {dy:.2f}, DZ: {dz:.2f}")
    print("RP_ERR:", ori_diff)


if __name__ == "__main__":
    main()
