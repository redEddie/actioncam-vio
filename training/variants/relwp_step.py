"""Relative-waypoint action processor step for lerobot SmolVLA training/inference.

Training batch (after add_batch_dim): action = absolute future states S[t+1..t+15] (B,15,6) as stored in the
`relwp` dataset variant, observation.state_abs = S[t] (B,6). This step rewrites action := action - state_abs,
i.e. waypoints relative to the current pose (UMI / XR-1 / RDT2 / VISTA convention). At inference there is no
action in the transition -> no-op. Normalization stats in the variant dataset already describe relative waypoints.
"""
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry, TransitionKey

STATE_ABS_KEY = "observation.state_abs"


@ProcessorStepRegistry.register(name="relative_waypoint_action_v1")
class RelativeWaypointActionStep(ProcessorStep):
    def __call__(self, transition):
        action = transition.get(TransitionKey.ACTION)
        obs = transition.get(TransitionKey.OBSERVATION) or {}
        if action is None or STATE_ABS_KEY not in obs:
            return transition
        s_abs = obs[STATE_ABS_KEY]
        if s_abs.ndim == action.ndim - 1:
            s_abs = s_abs.unsqueeze(-2)
        new = dict(transition)
        new[TransitionKey.ACTION] = action - s_abs.to(action.dtype)
        return new

    def transform_features(self, features):
        return features

    def get_config(self):
        return {}


def install(insert_after="AddBatchDimensionProcessorStep"):
    """Monkeypatch lerobot_train.make_pre_post_processors to insert this step after add_batch_dim."""
    import lerobot.scripts.lerobot_train as T
    orig = T.make_pre_post_processors

    def patched(*args, **kwargs):
        pre, post = orig(*args, **kwargs)
        names = [type(s).__name__ for s in pre.steps]
        if "RelativeWaypointActionStep" not in names:
            idx = names.index(insert_after) + 1 if insert_after in names else 0
            pre.steps.insert(idx, RelativeWaypointActionStep())
            print(f"[relwp] inserted RelativeWaypointActionStep at {idx}: {[type(s).__name__ for s in pre.steps]}", flush=True)
        return pre, post

    T.make_pre_post_processors = patched
