import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    # 경로 세팅
    gopro_dir = Path("/home/kimminje/Desktop/kimminje/gopro")
    vio_dir = gopro_dir / "actioncam-vio"
    episode_dir = vio_dir / "Episode"
    result_dir = vio_dir / "Episode_result"
    
    episode_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = []
    for ext in ('*.mp4', '*.MP4'):
        video_files.extend(episode_dir.glob(ext))
    video_files = sorted(video_files)
    
    if not video_files:
        print("❌ 에러: 영상이 없습니다.")
        return
    
    mapping_file = result_dir / "episode_mapping_log.json"
    mapping_data = {}
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            try:
                mapping_data = json.load(f)
            except json.JSONDecodeError:
                pass

    processed_videos = list(mapping_data.values())
    if mapping_data:
        existing_numbers = [int(k.split('_')[1]) for k in mapping_data.keys() if k.startswith("episode_")]
        next_ep_num = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_ep_num = 1

    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.name
        if video_name in processed_videos:
            continue
            
        episode_name = f"episode_{next_ep_num}"
        print(f"\n========================================================")
        print(f"🚀 [진행 상황: {i}/{len(video_files)}] {episode_name} UMI 변환 시작! (원본: {video_name})")
        print(f"========================================================")
        
        ep_out = result_dir / episode_name
        
        # 0. IMU 데이터 추출
        cmd_extract = [
            sys.executable, "-m", "gopro_vio.extract", str(video_path),
            "-o", str(ep_out)
        ]
        
        # 1. SLAM (마스크 적용 + 마커 초기화 크기 0.10 지정)
        cmd_slam = [
            sys.executable, "-m", "gopro_vio.slam", str(video_path),
            "--imu", f"{ep_out}/imu.csv",
            "-o", f"{ep_out}/slam",
            "--calib", "cameras/hero13black/calibration/intrinsics.json",
            "--extr", "cameras/hero13black/calibration/imu_extrinsics.json",
            "--mask", "cameras/hero13black/calibration/gripper_mask.png",
            "--init_tag_size", "0.10",
            "--width", "960", "--features", "2500", "--fps-div", "2"
        ]
        
        # 2. 마커 감지 (월드 앵커 13번, 그리퍼 0번 1번을 동시에 찾음)
        cmd_aruco = [
            sys.executable, "-m", "gopro_vio.aruco_detect", str(video_path),
            "--calib", "cameras/hero13black/calibration/intrinsics.json",
            "-o", f"{ep_out}/tags.pkl",
            "--step", "1",
            "--ids", "0", "1", "13"
        ]
        
        # 3. 월드 앵커 정렬 (13번 마커 기준)
        cmd_world_align = [
            sys.executable, "-m", "gopro_vio.world_align",
            f"{ep_out}/tags.pkl",
            f"{ep_out}/slam/camera_trajectory.csv",
            "-o", f"{ep_out}/world"
        ]
        
        # 4. 그리퍼 너비 추출 (0, 1번 마커)
        cmd_gripper = [
            sys.executable, "-m", "gopro_vio.gripper_width",
            f"{ep_out}/tags.pkl",
            "-o", f"{ep_out}/gripper"
        ]
        
        try:
            print("⏳ [1/5] IMU 센서 데이터 추출 중...")
            subprocess.run(cmd_extract, cwd=vio_dir, check=True)
            
            print("⏳ [2/5] ORB-SLAM3 가동 중 (마스크 + 10cm 스케일 적용)...")
            subprocess.run(cmd_slam, cwd=vio_dir, check=True)
            
            print("⏳ [3/5] ArUco 마커(0, 1, 13) 감지 중...")
            subprocess.run(cmd_aruco, cwd=vio_dir, check=True)
            
            print("⏳ [4/5] 월드 앵커 좌표 정렬 중 (id 13 기준)...")
            subprocess.run(cmd_world_align, cwd=vio_dir, check=True)
            
            print("⏳ [5/5] 그리퍼 상태(Width) 추출 중 (id 0, 1 기준)...")
            subprocess.run(cmd_gripper, cwd=vio_dir, check=True)
            
            mapping_data[episode_name] = video_name
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, indent=4, ensure_ascii=False)
                
            print(f"✅ {episode_name} 모든 변환 성공!")
            next_ep_num += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [에러 발생] {video_name} 실패! 다음 영상으로 넘어갑니다. (Exit code: {e.returncode})")
            continue

if __name__ == "__main__":
    main()
