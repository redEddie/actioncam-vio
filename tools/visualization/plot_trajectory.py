import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_trajectory(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    if 'is_lost' in df.columns:
        df = df[~df['is_lost']]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 궤적 그리기
    ax.plot(df['x'], df['y'], df['z'], label='Camera Trajectory', color='blue', linewidth=2)
    
    # 시작점과 마커 원점 표시
    ax.scatter(0, 0, 0, color='red', s=150, marker='*', label='Marker Origin (0,0,0)')
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], color='green', s=100, label='Start Point')
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], color='orange', s=100, label='End Point')

    ax.set_xlabel('X (Right/Left)')
    ax.set_ylabel('Y (Forward/Backward)')
    ax.set_zlabel('Z (Up/Down)')
    ax.set_title('3D SLAM Camera Trajectory in Marker Frame')
    
    # 비율 맞추기 (왜곡 방지)
    max_range = np.array([df['x'].max()-df['x'].min(), df['y'].max()-df['y'].min(), df['z'].max()-df['z'].min()]).max() / 2.0
    mid_x = (df['x'].max()+df['x'].min()) * 0.5
    mid_y = (df['y'].max()+df['y'].min()) * 0.5
    mid_z = (df['z'].max()+df['z'].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.grid(True)
    
    # GUI 백엔드 오류(Agg)를 우회하기 위해 이미지 파일로 저장
    save_path = "trajectory_3d.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D 그래프가 이미지로 저장되었습니다! 폴더에서 '{save_path}' 파일을 열어보세요!")

if __name__ == '__main__':
    import numpy as np
    
    if len(sys.argv) < 2:
        print("💡 사용법: python plot_trajectory.py [csv파일경로]")
        print("예시: python plot_trajectory.py capture/Episode_result/ep_0000/world/camera_trajectory.csv")
    else:
        plot_trajectory(sys.argv[1])
