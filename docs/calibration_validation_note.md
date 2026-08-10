# HERO13 UMI 좌표계·캘리브레이션 검증 기록

검증일: 2026-08-03

## 결론

- `trajectory_tcp_robot.csv`는 `robot_base -> tcp_link` TCP pose이며, IK/학습의
  유일한 pose 입력으로 승인한다.
- camera→TCP와 tag→robot 고정 변환은 변경하지 않는다.
- camera→TCP 실측값 `[0.0, 0.09, 0.16] m` 및 `diag(-1, +1, -1)` 회전은
  장착 상태에서 사용자 확인을 완료한 승인 기준값이다. URDF `root -> tcp_link`
  14 cm 오프셋을 여기에 추가 적용하지 않는다.

## 독립 산출물 검증

맵 1개와 에피소드 56개, 총 28,479행을 CSV에서 재합성했다.

| 검사항목 | 결과 |
| --- | ---: |
| camera→TCP 위치 최대 오차 | `1.61e-16 m` |
| tag→robot 위치 최대 오차 | `1.67e-16 m` |
| 두 자세 합성의 최대 오차 | `9.39e-16 rad` |
| provenance / calibration manifest 불일치 | 0 / 57 |
| ORB-SLAM KB4 설정 불일치 | 0 / 57 |
| tag 정렬 residual median / max | `0.192 / 0.995 cm` |

오차는 부동소수점 반올림 범위이며, 좌표축 반전·중복 오프셋·timestamp/행 불일치는
발견되지 않았다.

## Hero13 내참값

- KB4 fisheye, 3840×3360에서 캘리브레이션 후 960×840 SLAM 입력으로 일관되게
  스케일됐다.
- calibration RMS: `0.397 px`; 전체 재투영 오차 mean/median/p95:
  `0.386 / 0.334 / 0.852 px`.

## 알려진 잔여 리스크 (현재 작업 차단 안 함)

`imu_extrinsics.json`의 IMU→camera 병진은 gyro-only 정렬에서 관측되지 않아
`[0, 0, 0] m`으로 둔다. 실제 GoPro 렌즈와 IMU 사이의 약 1–2 cm 병진은 아직
모델에 없다. 이는 이미 tag-frame으로 정렬된 카메라 pose에서 TCP로 가는 고정
변환에는 재적용되지 않으며, 빠른 동작에서만 VIO의 미세한 동적 정확도에 영향을
줄 수 있다. 현 단계에서는 기록만 유지하고 재보정은 수행하지 않는다.
