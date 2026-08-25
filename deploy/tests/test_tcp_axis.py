def main():
    import numpy as np
    from ik.ik_solver_v7 import DLSInverseKinematicsV7
    ik = DLSInverseKinematicsV7(urdf_path="urdf/so_arm_with_gopro_final.urdf")
    q = np.zeros(ik.num_joints)
    R = ik.get_tcp_rotation(q)
    print("TCP Rotation Matrix at q=0:")
    print(np.round(R, 3))


if __name__ == "__main__":
    main()
