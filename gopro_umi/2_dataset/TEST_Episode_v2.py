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
    
    # 맵 영상 파일 이름 (사용자가 map_video.MP4 로 저장했다고 가정)
    map_video_name = "map_video.MP4"
    map_video_path = episode_dir / map_video_name
    
    # 소문자 확장자 예외 처리
    if not map_video_path.exists():
        map_video_path = episode_dir / "map_video.mp4"
        map_video_name = "map_video.mp4"
        
    if not map_video_path.exists():
        print(f"❌ [에러] 맵 영상을 찾을 수 없습니다! '{episode_dir}' 안에 '{map_video_name}' 이름으로 저장해주세요.")
        return

    # 데모 영상들 수집 (맵 영상 제외)
    video_files = []
    for ext in ('*.mp4', '*.MP4'):
        video_files.extend(episode_dir.glob(ext))
    
    demo_files = sorted([f for f in video_files if f.name.lower() != "map_video.mp4"])
    
    if not demo_files:
        print("❌ [에러] 데모 영상(에피소드)이 없습니다.")
        return

    # 캘리브레이션 세팅 (Hero 13)
    calib_json = "cameras/hero13black/calibration/intrinsics.json"
    extr_json = "cameras/hero13black/calibration/imu_extrinsics.json"
    mask_png = "cameras/hero13black/calibration/gripper_mask.png"

    # =========================================================================
    # STEP A: 맵 영상(Map Video) 우선 처리
    # =========================================================================
    map_out_dir = result_dir / "map_result"
    map_out_dir.mkdir(parents=True, exist_ok=True)
    
    # 맵 영상이 성공적으로 만들어졌는지 확인하기 위한 플래그 파일
    map_done_flag = map_out_dir / "map_done.txt"
    
    if not map_done_flag.exists():
        print(f"\n========================================================")
        print(f"🌍 [STEP A] 맵 영상(Map Video) 분석 시작...")
        print(f"========================================================")
        
        try:
            print("⏳ [1/4] IMU 추출...")
            subprocess.run([sys.executable, "-m", "gopro_vio.extract", str(map_video_path), "-o", str(map_out_dir)], cwd=vio_dir, check=True)
            
            print("⏳ [2/4] 맵 생성 (SLAM)...")
            subprocess.run([sys.executable, "-m", "gopro_vio.slam", str(map_video_path),
                "--imu", f"{map_out_dir}/imu.csv", "-o", f"{map_out_dir}/slam",
                "--calib", calib_json, "--extr", extr_json, "--mask", mask_png,
                "--init_tag_size", "0.10", "--width", "960", "--features", "2500", "--fps-div", "2"], cwd=vio_dir, check=True)
            
            print("⏳ [3/4] 13번 월드 앵커 탐지...")
            subprocess.run([sys.executable, "-m", "gopro_vio.aruco_detect", str(map_video_path),
                "--calib", calib_json, "-o", f"{map_out_dir}/tags.pkl", "--step", "2", "--ids", "13"], cwd=vio_dir, check=True)
                
            print("⏳ [4/4] 월드 정렬 (World Align)...")
            subprocess.run([sys.executable, "-m", "gopro_vio.world_align",
                f"{map_out_dir}/tags.pkl", f"{map_out_dir}/slam/camera_trajectory.csv", "-o", f"{map_out_dir}/world"], cwd=vio_dir, check=True)
            
            # 성공 플래그 생성
            with open(map_done_flag, "w") as f:
                f.write("Map generation success!")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 맵 영상 처리 실패! 데모 영상을 처리할 수 없습니다. (종료)")
            return
    else:
        print(f"✅ 맵 영상은 이미 처리되어 있습니다. 데모 분석으로 넘어갑니다.")


    # =========================================================================
    # STEP B: 데모 영상(Demo Videos) 일괄 처리
    # =========================================================================
    mapping_file = result_dir / "episode_mapping_log.json"
    mapping_data = {}
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    processed_videos = list(mapping_data.values())
    
    # 시작할 에피소드 번호 파악
    existing_numbers = [int(k.split('_')[1]) for k in mapping_data.keys() if k.startswith("episode_")]
    next_ep_num = max(existing_numbers) + 1 if existing_numbers else 1

    for i, demo_path in enumerate(demo_files, 1):
        video_name = demo_path.name
        if video_name in processed_videos:
            continue
            
        episode_name = f"episode_{next_ep_num}"
        ep_out = result_dir / episode_name
        
        print(f"\n========================================================")
        print(f"🚀 [STEP B 진행: {i}/{len(demo_files)}] {episode_name} 데모 처리 시작! (원본: {video_name})")
        print(f"========================================================")
        
        try:
            print("⏳ [1/5] IMU 추출...")
            subprocess.run([sys.executable, "-m", "gopro_vio.extract", str(demo_path), "-o", str(ep_out)], cwd=vio_dir, check=True)
            
            print("⏳ [2/5] 데모 SLAM (맵 주입 - Load Map)...")
            subprocess.run([sys.executable, "-m", "gopro_vio.slam", str(demo_path),
                "--imu", f"{ep_out}/imu.csv", "-o", f"{ep_out}/slam",
                "--calib", calib_json, "--extr", extr_json, "--mask", mask_png,
                "--load-map", f"{map_out_dir}/slam/map_atlas.osa", # 핵심 포인트!
                "--init_tag_size", "0.10", "--width", "960", "--features", "2500", "--fps-div", "2"], cwd=vio_dir, check=True)
            
            print("⏳ [3/5] 맵 지도를 상속받아 월드 원점 강제 적용...")
            subprocess.run([sys.executable, "-m", "gopro_vio.world_align",
                "--apply", f"{map_out_dir}/world/tx_slam_tag.json", # 핵심 포인트!
                f"{ep_out}/slam/camera_trajectory.csv", "-o", f"{ep_out}/world"], cwd=vio_dir, check=True)
            
            print("⏳ [4/5] 그리퍼 마커(0, 1) 탐지...")
            subprocess.run([sys.executable, "-m", "gopro_vio.aruco_detect", str(demo_path),
                "--calib", calib_json, "-o", f"{ep_out}/tags.pkl", "--step", "1", "--ids", "0", "1"], cwd=vio_dir, check=True)
                
            print("⏳ [5/5] 그리퍼 상태(Width) 추출...")
            subprocess.run([sys.executable, "-m", "gopro_vio.gripper_width",
                f"{ep_out}/tags.pkl", "-o", f"{ep_out}/gripper"], cwd=vio_dir, check=True)
            
            mapping_data[episode_name] = video_name
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, indent=4, ensure_ascii=False)
                
            print(f"✅ {episode_name} 성공!")
            next_ep_num += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [에러 발생] {video_name} 실패! 다음으로 넘어갑니다.")
            continue

if __name__ == "__main__":
    main()
