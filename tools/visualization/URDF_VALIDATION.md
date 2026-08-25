# URDF / 제어입력 검증 노트 (2026-08-21)

대상: `4_deploy/ik/urdf/so_arm_with_gopro_final.urdf` (so101_new_calib, 29 links)

| 검증 | 결과 |
|---|---|
| TCP 체인 | wrist_roll → gripper_link → (−0.01 m) → root(그리퍼 어셈블리) → (−0.14 m) → tcp_link(손끝). FK/IK 모두 tcp_link 조준 |
| FK 교차검증 | IK 라이브러리 vs MuJoCo tcp_link, 랜덤 관절각 100개: **max 0.0000 mm** |
| IK↔FK 왕복 | V6 예측 웨이포인트: 평균 0.13 mm, p95 0.41 mm |
| 작업공간 | 로봇 최대 도달 r=0.529 m vs **데이터셋 TCP 최대 r=1.108 m** — 사람 시연에 로봇 도달 불가 구간 존재(IK 실패 ~17%). VISTA식 물리 검증 필터 도입 후보 |
| 주의 1 | 그리퍼 어셈블리 12개 링크는 visual 메시만 있음 → MuJoCo 렌더에 안 보임(기구학은 정상). 충돌검사 하려면 collision 메시 추가 필요 |
| 주의 2 | 중립 자세에서 인접 링크 collision 메시 4쌍 접촉 경고(기구학 무해) |

시각화: `mujoco_motorless_replay.py` — GT/예측 청크를 deploy IK 경로로 풀어 mp4 재생
(빨강 구=FK TCP, 주황=TCP 궤적, 파랑=명령 웨이포인트). MUJOCO_GL=egl 필요.

지연 실측(관련): 원격 추론 p50 256 ms(벤치마크 01), 카메라 glass-to-host 88±14 ms
(HERO7+GC553Pro, gopro_slam docs/observation_latency_model.md) → 런타임 관측 정렬 상수.
HERO13 캡처 경로로 바꾸면 latency_test.py로 재측정할 것.
