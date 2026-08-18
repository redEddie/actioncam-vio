import numpy as np

from incremental_action_contract import ACTION_DIM, CHUNK_SIZE, integrate_actions


def convert_delta_to_absolute(
    current_state: np.ndarray, physical_delta_chunk: np.ndarray
) -> np.ndarray:
    """
    UNNORMALIZED physical delta 15개를 15개 future absolute state로 복원합니다.

    이 함수에는 normalized model output을 직접 넣으면 안 됩니다. 저장된
    MEAN_STD postprocessor로 먼저 unnormalize한 뒤 호출해야 합니다. Gripper
    safety clip은 누적 복원된 physical absolute state에만 적용됩니다.
    
    Args:
        current_state: 현재 로봇의 6D 절대 위치 (shape: [6])
                       [x, y, z, roll, pitch, gripper]
        physical_delta_chunk: unnormalize된 100-ms incremental action (shape: [15, 6])
    
    Returns:
        predicted_absolute_chunk: +0.1 ... +1.5 s future state (shape: [15, 6])
    """
    current = np.asarray(current_state, dtype=np.float64)
    actions = np.asarray(physical_delta_chunk, dtype=np.float64)
    if current.shape != (ACTION_DIM,):
        raise ValueError(f"current_state must have shape [{ACTION_DIM}], got {current.shape}")
    if actions.shape != (CHUNK_SIZE, ACTION_DIM):
        raise ValueError(
            f"physical_delta_chunk must have shape [{CHUNK_SIZE},{ACTION_DIM}], got {actions.shape}"
        )
    if not np.isfinite(current).all() or not np.isfinite(actions).all():
        raise ValueError("current state or physical action chunk contains NaN/Inf")

    # integrate_actions returns S0..S15. S0 is the observation anchor and is
    # deliberately excluded from the 15 predicted future trajectory points.
    predicted_absolute_chunk = integrate_actions(current, actions)[1:]

    # This is an absolute physical gripper-state clip. Normalized dGripper is
    # never clipped to [0,1].
    predicted_absolute_chunk[:, 5] = np.clip(predicted_absolute_chunk[:, 5], 0.0, 1.0)
    return predicted_absolute_chunk

# =====================================================================
# [실제 로컬 Inference 스크립트 적용 예시]
#
# 1. 모델에서 델타 청크를 뽑아냅니다.
# normalized_delta_chunk = policy.predict_action_chunk(model_observation)
#
# 2. 저장된 MEAN_STD postprocessor로 physical delta를 복원합니다.
# physical_delta_chunk = postprocessor(normalized_delta_chunk)
#
# 3. physical delta만 누적하여 15개 absolute future point를 만듭니다.
# absolute_chunk = convert_delta_to_absolute(current_state, physical_delta_chunk)
#
# 4. 이제 기존에 쓰시던 앙상블(Temporal Ensembling) 로직에
# absolute_chunk를 그대로 던져주시면 완벽하게 돌아갑니다!
# =====================================================================
