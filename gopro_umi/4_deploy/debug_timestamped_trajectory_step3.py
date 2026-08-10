import time
import numpy as np
from dataclasses import dataclass

@dataclass
class TimestampedTrajectory:
    target_times: np.ndarray
    actions: np.ndarray

def build_timestamped_trajectory(
    absolute_chunk: np.ndarray,
    anchor_time: float,
    trajectory_hz: float = 60.0
) -> TimestampedTrajectory:
    num_steps = absolute_chunk.shape[0]
    dt = 1.0 / trajectory_hz
    
    # action[0] is state[t+1], so offset starts from dt
    offsets = np.arange(1, num_steps + 1) * dt
    target_times = anchor_time + offsets
    
    return TimestampedTrajectory(
        target_times=target_times,
        actions=absolute_chunk.copy()  # Ensure it's not accidentally modified
    )

def main():
    # 1. Create a dummy absolute chunk simulating the output of STEP 2 [60, 6]
    original_chunk = np.random.randn(60, 6).astype(np.float32)
    
    # 2. Get monotonic anchor time
    anchor_time = time.monotonic()
    
    # 3. Build timestamped trajectory
    trajectory = build_timestamped_trajectory(original_chunk, anchor_time)
    
    # 4. Perform Verifications
    print(f"ABSOLUTE CHUNK SHAPE: {original_chunk.shape}")
    print(f"TIMESTAMP ARRAY SHAPE: {trajectory.target_times.shape}")
    print(f"ANCHOR TIME: {anchor_time:.6f}")
    
    print("\nRELATIVE TARGET TIMES")
    print(f"index 0: {trajectory.target_times[0] - anchor_time:.7f}")
    print(f"index 1: {trajectory.target_times[1] - anchor_time:.7f}")
    print(f"index 2: {trajectory.target_times[2] - anchor_time:.7f}")
    print(f"index 29: {trajectory.target_times[29] - anchor_time:.7f}")
    print(f"index 59: {trajectory.target_times[59] - anchor_time:.7f}")
    
    dts = np.diff(trajectory.target_times)
    print("\nINTER-SAMPLE DT")
    print(f"min: {dts.min():.9f}")
    print(f"max: {dts.max():.9f}")
    print(f"mean: {dts.mean():.9f}")
    print(f"std: {dts.std():.9f}")
    
    is_monotonic = np.all(dts > 0)
    print(f"\nSTRICTLY MONOTONIC? {'YES' if is_monotonic else 'NO'}")
    
    max_action_diff = np.max(np.abs(original_chunk - trajectory.actions))
    print(f"ACTION VALUES MODIFIED? {'YES' if max_action_diff > 0 else 'NO'}")
    print(f"max_abs_difference: {max_action_diff}")
    
    first_offset = trajectory.target_times[0] - anchor_time
    total_horizon = trajectory.target_times[-1] - anchor_time
    print(f"\nTRAJECTORY HORIZON: {total_horizon:.6f}")
    print(f"FIRST TARGET OFFSET: {first_offset:.7f}")

if __name__ == '__main__':
    main()
