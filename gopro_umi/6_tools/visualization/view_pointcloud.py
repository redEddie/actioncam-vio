import open3d as o3d
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("💡 사용법: python view_pointcloud.py [ply파일경로]")
        sys.exit(1)

    ply_path = sys.argv[1]
    if not os.path.exists(ply_path):
        print(f"❌ 파일을 찾을 수 없습니다: {ply_path}")
        sys.exit(1)

    print("🚀 포인트 클라우드를 불러오는 중...")
    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        print("❌ 텅 빈 파일이거나 지원하지 않는 포맷입니다.")
        sys.exit(1)

    print(f"✅ 총 {len(pcd.points)}개의 3D 특징점을 성공적으로 불러왔습니다!")
    print("👉 조작법: 마우스 왼쪽 버튼(회전), 휠(확대/축소), 오른쪽 버튼(이동)")
    
    # 포인트 클라우드가 잘 보이도록 단색(검은색 톤)으로 칠하기 (선택사항)
    pcd.paint_uniform_color([0.1, 0.7, 0.1]) # 매트릭스 스타일 녹색 점

    # 3D 뷰어 창 띄우기
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="SLAM Point Cloud Viewer", 
                                      width=1280, 
                                      height=720,
                                      point_show_normal=False)

if __name__ == "__main__":
    main()
