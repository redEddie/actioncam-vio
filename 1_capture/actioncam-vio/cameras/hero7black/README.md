# GoPro HERO7 Black

<img src="images/gopro_hero7_black.jpg" width="420">

| 샘플 (공원) | 캘리브레이션 보드 |
|---|---|
| ![](images/sample_park.jpg) | ![](images/sample_board.jpg) |

## 측정 스펙 (이 데이터셋에서 실측)

| 항목 | 값 |
|---|---|
| 영상 모드 | 2704×2028 (4:3) @ 59.94 fps, H.264 |
| **FOV (실측)** | **H 120.5° / V 93.7° / D 146.6°** |
| 텔레메트리 | MP4 `gpmd` 트랙 (GPMF) — 순수 Python 파서로 추출 |
| IMU | ACCL/GYRO **197.7 Hz** |
| 가속도계 품질 | \|g\| = 9.49 m/s² (**-3.3% 스케일 오차**) |
| IMU-영상 시간 오프셋 | **-53 ms** (imu_sync로 자동 추정, 반복 ±2 ms) |
| 안정화 | HyperSmooth **OFF 필수** (udta GPMF `EISE=N`으로 확인 가능) |
| 주의 설정 | 자동 저조도(`VLTE`)도 끄는 것 권장 |

## 캘리브레이션 (KB4 fisheye)

- fx=1215.0, fy=1217.3, cx=1343.3, cy=1030.4 / k1~k4 = 0.0476, 0.0191, -0.0109, 0.0023
- RMS **0.826 px** (60뷰), 전 프레임 확장 평균 0.65 px — 가장자리 발산 없음
- **FOV: H 120.5° / V 93.7° / D 146.6°** (GoPro Wide 4:3 명목치와 일치)
- 커버리지: 10×8 셀 중 78셀에 100+ 샘플

![캘리브레이션 리포트](calibration/calib_report.png)

## 결과

| 영상 | 환경 | 결과 |
|---|---|---|
| GX014219 | 공원 산책로 | 전 구간 궤적 (99.7%, 174 m) |
| GX014220 | 연못 공원 | 클린 완주 + 맵 저장 (91.5%, 307 m) |
| GX014221 | 보도 | 부분 궤적 (최대 맵 53%) |
| GX014217 | 전시홀(무지 파티션) | **실패** — 아틀라스 분절 (연구노트 §3-1) |

![공원 궤적](results/trajectory_GX014219_park.png)
![연못 공원 궤적](results/trajectory_GX014220_pond.png)

**근거리 정밀도** (ChArUco 기준 대비, 보드 15~63 cm 거리 48초):
ATE **RMSE 1.88 cm** (rigid) / 0.34 cm (similarity, 스케일 계수 1.126)

![근거리 ATE](results/board_eval_GX014222.png)

### UMI 그리퍼 검증 (2026-07-07)

그리퍼 장착 + 화면 하단 33% 마스킹 상태로 여닫음/pick&place 데모 **99.5/100%
추적**(단일 세그먼트), id13 태그 월드 앵커 잔차 0.23 cm. 한계: 던지기급
고속 모션은 50%로 조각남(블러/198 Hz IMU — 매칭된 맵으로도 회복 안 됨).
상세는 [UMI 검증 문서](../../docs/umi_gripper_pipeline.md).

![맵 영상 월드 정렬](results/umi/world_GX014229_map.png)
![pick&place 월드 궤적](results/umi/world_GX014233_pickplace.png)

## 알려진 한계

- 가속도계 -3.3% 스케일 오차 → IMU 스케일 계수 1.126으로 나타남
- 무지 벽/파티션이 시야를 채우는 실내(전시홀)에서 맵 분절 — FOV 120°의 한계
- 컨테이너의 구형 FFmpeg가 2.7K 원본을 못 읽음 → 사전 트랜스코딩 필수(파이프라인이 자동 처리)

## HDMI 캡처 경로 (AVerMedia GC553Pro)

HERO7은 웹캠 모드가 없어 라이브 스트림은 HDMI→USB 캡처 장치 경유가 유일.
현행 장치는 **AVerMedia Live Gamer ULTRA S (GC553Pro)**:
`/dev/v4l/by-id/usb-AVerMedia_Live_Gamer_ULTRA_S_GC553Pro_...-video-index0`.

| 항목 | 값 |
|---|---|
| **종단 지연 (2026-07-13, 측정도구 v3, 615샘플)** | **median 111 ms (p10 94 / p90 131, 지터 37 ms)** → 롤아웃 보정 상수 **μ≈88 ms, σ≈14 ms** ([노이즈 모델](../../docs/observation_latency_model.md)) |
| 권장 포맷 | **1080p60 YUYV(무압축)** — MJPG는 1080p240까지 지원. 구형 동글의 블렌딩/손상 문제 없음 |
| **기하 (4:3 모드)** | **필러박스 4:3** — GoPro가 4:3 영상 양옆에 검은 보더를 넣어 16:9로 출력. 콘텐츠는 x=240..1679 (1440×1080, 정확히 4:3), 녹화 기하의 균일 ×0.5325 축소. **크롭 `frame[:, 240:1680]` 한 줄이면 녹화와 동일 기하** (녹화 K를 0.5325배로 스케일해 사용). 과거 "SuperView식 스트레치" 결론(MBF 시절 분석)은 폐기 |
| 체감 지연 주의 | ffplay 등 뷰어의 표시 버퍼링이 0.3~0.5 s를 더해 보임 — 실제 캡처 지연과 구분할 것 |

<details><summary>구형: MacroSilicon(MBF) 동글 기록 (교체됨)</summary>

- 종단 지연 720p60: median 186 ms, p10 131 / p90 227 (측정도구 v1 — 수십 ms 과대 추정 포함)
- USB2 대역폭 위장: 1080p30은 인접 프레임 블렌딩(이중상), 1080p60은 프레임 손상 → 720p60만 유효
- 캡처를 닫으면 HDMI 핫플러그로 GoPro 상태 리셋 → `scripts/hdmi_record.sh`가 단일 오픈 유지로 우회 (GC553Pro에서는 미확인 — 재현 시 같은 스크립트 사용)

</details>

## 사용법

```bash
python -m gopro_vio.extract data/hero7black/GX014222.MP4 -o output/GX014222
python -m gopro_vio.charuco data/hero7black/GX014222.MP4 -o cameras/hero7black/calibration \
    --squares 10 8 --square-size 0.023
python -m gopro_vio.imu_sync cameras/hero7black/calibration/board_poses.npz \
    output/GX014222/imu.csv -o cameras/hero7black/calibration
python -m gopro_vio.slam data/hero7black/GX014219.MP4 --imu output/GX014219/imu.csv \
    -o output/GX014219/slam --calib cameras/hero7black/calibration/intrinsics.json \
    --extr cameras/hero7black/calibration/imu_extrinsics.json
```
