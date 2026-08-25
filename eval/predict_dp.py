#!/usr/bin/env python3
"""Offline chunk prediction with the original-UMI Diffusion Policy on canonical 10 Hz anchors.
Outputs npz with pred/gt actions+states in yaw-free 6D space, aligned to the LeRobot GT."""
import argparse, sys, json, time
import numpy as np, torch, dill, hydra, zarr
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/home/jeonchanwook/universal_manipulation_interface')
sys.path.insert(0, '/home/jeonchanwook/work/eval')
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action
from eval_common import load_gt, anchors, HOLDOUT, TRAIN_EVAL, HORIZON, BASE_ZARR

def yawfree6(pos, rotvec, grip):
    e = R.from_rotvec(rotvec).as_euler("ZYX")   # [yaw,pitch,roll]
    return np.concatenate([pos, e[..., 2:3], e[..., 1:2], grip], axis=-1)

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stride', type=int, default=5)
ap.add_argument('--episodes', default='holdout+train')
ap.add_argument('--device', default='cuda:0')
args = ap.parse_args()

payload = torch.load(open(args.ckpt, 'rb'), pickle_module=dill, map_location='cpu', weights_only=False)
cfg = payload['cfg']
cls = hydra.utils.get_class(cfg._target_)
workspace: BaseWorkspace = cls(cfg)
workspace.load_payload(payload, exclude_keys=None, include_keys=None)
policy = workspace.ema_model if cfg.training.use_ema else workspace.model
policy.num_inference_steps = 16
policy.eval().to(args.device)
shape_meta = cfg.task.shape_meta
down = int(cfg.task.obs_down_sample_steps)
img_h = int(shape_meta.obs.camera0_rgb.horizon)
print('ckpt', args.ckpt, 'epoch', payload.get('pickles',{}).get('epoch'), 'global_step', payload.get('pickles',{}).get('global_step'), 'down', down, 'obs_pose_repr', cfg.task.pose_repr.obs_pose_repr, 'action_pose_repr', cfg.task.pose_repr.action_pose_repr)

z = zarr.open(str(BASE_ZARR), mode='r')
rgb = z['data/camera0_rgb']; pos = z['data/robot0_eef_pos'][:]; rot = z['data/robot0_eef_rot_axis_angle'][:]; grip = z['data/robot0_gripper_width'][:]
ends = z['meta/episode_ends'][:]; starts = np.concatenate([[0], ends[:-1]])
gt = load_gt()
ep_list = {'holdout': HOLDOUT, 'train': TRAIN_EVAL, 'holdout+train': HOLDOUT + TRAIN_EVAL}[args.episodes]

recs = []
t0 = time.time()
for ep in ep_list:
    e = gt[ep]; S = e['states']; A = e['actions']; src = e['src_idx']
    es = int(starts[ep])
    start_pose = np.concatenate([pos[es], rot[es]])
    # sanity: GT lerobot state == yawfree(base zarr at src idx)
    chk = yawfree6(pos[src[0]], rot[src[0]], grip[src[0]])
    assert np.allclose(chk, S[0], atol=1e-4), (chk, S[0])
    for t in anchors(e, args.stride):
        f = int(src[t])
        frames = [max(f - k * down, es) for k in range(img_h - 1, -1, -1)]
        env_obs = {
            'camera0_rgb': np.stack([rgb[i] for i in frames]),
            'robot0_eef_pos': pos[frames], 'robot0_eef_rot_axis_angle': rot[frames], 'robot0_gripper_width': grip[frames],
        }
        obs_np = get_real_umi_obs_dict(env_obs, shape_meta, obs_pose_repr=cfg.task.pose_repr.obs_pose_repr, episode_start_pose=[start_pose])
        obs = dict_apply(obs_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(args.device))
        with torch.no_grad():
            act = policy.predict_action(obs)['action_pred'][0].cpu().numpy().astype(np.float64)
        act_abs = get_real_umi_action(act, env_obs, action_pose_repr=cfg.task.pose_repr.action_pose_repr)  # [16,7] pos,rotvec,grip
        S_pred16 = yawfree6(act_abs[:, :3], act_abs[:, 3:6], act_abs[:, 6:7])
        # action[0] is the current-frame pose (should be ~identity); future = 1..15
        s0 = S[t]
        S_pred = S_pred16[1:1 + HORIZON]
        # unwrap roll/pitch relative to s0 to avoid ±pi jumps
        for d in (3, 4):
            S_pred[:, d] = s0[d] + np.angle(np.exp(1j * (S_pred[:, d] - s0[d])))
        A_pred = np.diff(np.concatenate([s0[None], S_pred], 0), axis=0)
        recs.append(dict(ep=ep, t=t, s0=s0, gt_actions=A[t:t + HORIZON], gt_states=S[t + 1:t + 1 + HORIZON], pred_actions=A_pred, pred_states=S_pred, pred_cur_err=float(np.linalg.norm(S_pred16[0, :3] - s0[:3]))))
    print(f'ep {ep}: {len([r for r in recs if r["ep"]==ep])} chunks, elapsed {time.time()-t0:.0f}s', flush=True)

np.savez(args.out, model='umi_dp', ckpt=args.ckpt, epoch=payload.get('pickles',{}).get('epoch'), global_step=payload.get('pickles',{}).get('global_step'),
         ep=np.array([r['ep'] for r in recs]), t=np.array([r['t'] for r in recs]),
         s0=np.stack([r['s0'] for r in recs]), gt_actions=np.stack([r['gt_actions'] for r in recs]), gt_states=np.stack([r['gt_states'] for r in recs]),
         pred_actions=np.stack([r['pred_actions'] for r in recs]), pred_states=np.stack([r['pred_states'] for r in recs]),
         pred_cur_err=np.array([r['pred_cur_err'] for r in recs]))
print('saved', args.out, len(recs))
