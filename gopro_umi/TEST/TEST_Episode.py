import os
import sys
import json
import glob
import subprocess
from pathlib import Path

def main():
    # 기본 경로 세팅
    gopro_dir = Path(__file__).resolve().parent.parent
    vio_dir = gopro_dir / "1_data_pipeline" / "actioncam-vio"
    episode_dir = vio_dir / "Episode"
    result_dir = vio_dir / "Episode_result"
    
    # 폴더가 없으면 생성
    episode_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Episode 폴더 내의 모든 mp4 파일 찾기 (대소문자 구분 없이)
    video_files = []
    for ext in ('*.mp4', '*.MP4'):
        video_files.extend(episode_dir.glob(ext))
    
    # 파일 이름순 정렬
    video_files = sorted(video_files)
    
    if not video_files:
        print(f"❌ [에러] '{episode_dir}' 폴더 안에 mp4 영상이 하나도 없습니다!")
        print("영상을 먼저 폴더에 넣어주세요.")
        return
    
    print(f"🎬 총 {len(video_files)}개의 영상을 찾았습니다. 일괄 처리를 시작합니다...\n")
    
    # 장부(매핑 파일) 세팅
    mapping_file = result_dir / "episode_mapping_log.json"
    mapping_data = {}
    
    # 기존 장부가 있으면 불러와서 이어서 작성 (중복 방지)
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            try:
                mapping_data = json.load(f)
            except json.JSONDecodeError:
                pass

    # 이미 처리된 원본 영상들 목록
    processed_videos = list(mapping_data.values())
    
    # 다음 할당할 에피소드 번호 찾기
    if mapping_data:
        # "episode_3" 에서 숫자만 추출하여 최댓값 찾기
        existing_numbers = [int(k.split('_')[1]) for k in mapping_data.keys() if k.startswith("episode_")]
        next_ep_num = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_ep_num = 1

    # 순차적으로 처리 시작
    for i, video_path in enumerate(video_files, 1):
        video_name = video_path.name
        
        # 이미 장부에 있는 영상이면 건너뛰기
        if video_name in processed_videos:
            print(f"⏩ [스킵] '{video_name}' 파일은 이미 처리된 기록이 있습니다.")
            continue
            
        episode_name = f"episode_{next_ep_num}"
        
        print(f"\n========================================================")
        print(f"🚀 [진행 상황: {i}/{len(video_files)}] {episode_name} 작업 시작! (원본: {video_name})")
        print(f"========================================================")
        
        # 1. IMU 데이터 추출 명령어
        cmd_extract = [
            sys.executable, "-m", "gopro_vio.extract", str(video_path),
            "-o", f"Episode_result/{episode_name}"
        ]
        
        # 2. SLAM 추출 명령어
        cmd_slam = [
            sys.executable, "-m", "gopro_vio.slam", str(video_path),
            "--imu", f"Episode_result/{episode_name}/imu.csv",
            "-o", f"Episode_result/{episode_name}/slam",
            "--calib", "cameras/hero13black/calibration/intrinsics.json",
            "--extr", "cameras/hero13black/calibration/imu_extrinsics.json",
            "--width", "960", "--features", "2500", "--fps-div", "2"
        ]
        
        try:
            print(f"⏳ [1/2] IMU 센서 데이터 추출 중...")
            subprocess.run(cmd_extract, cwd=vio_dir, check=True)
            
            print(f"⏳ [2/2] ORB-SLAM3 가동 중... (시간이 꽤 걸립니다 ☕)")
            subprocess.run(cmd_slam, cwd=vio_dir, check=True)
            
            # 성공적으로 끝나면 장부에 기록
            mapping_data[episode_name] = video_name
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, indent=4, ensure_ascii=False)
                
            print(f"✅ {episode_name} 처리 및 장부 기록 완료!")
            next_ep_num += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [에러 발생] {video_name} 처리 중 문제가 생겼습니다! (Exit code: {e.returncode})")
            print("다음 영상으로 넘어갑니다...\n")
            continue
            
    print("\n🎉🎉 모든 에피소드 처리가 완료되었습니다!!! 장부(json)를 확인해보세요!")

if __name__ == "__main__":
    main()
