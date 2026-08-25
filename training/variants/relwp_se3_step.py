"""SE(3) relative-waypoint action processor step (UMI convention).
Training batch: action = absolute future poses as 10D [pos3, rot6d(first two rows), grip] (B,15,10),
observation.state_abs = current pose 10D (B,10). Rewrites action := 10D of inv(T_t) @ T_{t+k}.
Inference: no action in transition -> no-op.
"""
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, TransitionKey

STATE_ABS_KEY = "observation.state_abs"

def rot6d_to_mat(d6):  # (...,6) first-two-rows convention -> (...,3,3)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)

def ten_to_T(x):  # (...,10) -> (...,4,4), grip separate
    T = torch.zeros(*x.shape[:-1], 4, 4, dtype=x.dtype, device=x.device)
    T[..., :3, :3] = rot6d_to_mat(x[..., 3:9]); T[..., :3, 3] = x[..., :3]; T[..., 3, 3] = 1.0
    return T, x[..., 9:10]

def T_to_ten(T, grip):
    return torch.cat([T[..., :3, 3], T[..., :2, :3].reshape(*T.shape[:-2], 6), grip], -1)

@ProcessorStepRegistry.register(name="se3_relative_waypoint_action_v1")
class SE3RelativeWaypointActionStep(ProcessorStep):
    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        if action is None or STATE_ABS_KEY not in obs:
            return transition
        s_abs = obs[STATE_ABS_KEY].float()
        while s_abs.ndim > 2:                      # collapse stray time/batch dims -> (B,10)
            s_abs = s_abs.squeeze(1)
        if s_abs.ndim == 1:
            s_abs = s_abs.unsqueeze(0)
        T_abs, grip = ten_to_T(action.float())     # (B,T,4,4)
        T_cur, _ = ten_to_T(s_abs.unsqueeze(1))    # (B,1,4,4)
        rel = torch.linalg.solve(T_cur.expand(T_abs.shape), T_abs)   # inv(T_cur) @ T_abs
        new = dict(transition); new[TransitionKey.ACTION] = T_to_ten(rel, grip).to(action.dtype)
        return new
    def transform_features(self, features):
        return features
    def get_config(self):
        return {}

def install():
    import lerobot.scripts.lerobot_train as T
    orig = T.make_pre_post_processors
    def patched(*a, **k):
        pre, post = orig(*a, **k)
        names = [type(s).__name__ for s in pre.steps]
        if "SE3RelativeWaypointActionStep" not in names:
            idx = names.index("AddBatchDimensionProcessorStep") + 1 if "AddBatchDimensionProcessorStep" in names else 0
            pre.steps.insert(idx, SE3RelativeWaypointActionStep())
            print(f"[se3relwp] inserted at {idx}", flush=True)
        return pre, post
    T.make_pre_post_processors = patched
