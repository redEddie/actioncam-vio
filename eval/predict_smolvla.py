#!/usr/bin/env python3
"""Offline chunk prediction with a SmolVLA checkpoint (gopro_umi 10Hz/chunk15 incremental contract)
on the canonical anchors. Mirrors deploy/inference/remote_smolvla.py inline worker exactly."""
import argparse, sys, os, time, json
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np, torch
sys.path.insert(0, '/home/jeonchanwook/work/eval')
from eval_common import load_gt, anchors, decode_img, HOLDOUT, TRAIN_EVAL, HORIZON, integrate
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--stride', type=int, default=5)
ap.add_argument('--episodes', default='holdout+train')
ap.add_argument('--task', default='pick and place the target object')
ap.add_argument('--state-mode', default='abs', choices=['abs','relvel','dropxyz'], help='must match the dataset variant the checkpoint was trained on')
ap.add_argument('--prev-image', action='store_true', help='feed previous 10Hz frame as observation.images.camera2')
ap.add_argument('--action-mode', default='incr', choices=['incr','relwp','se3relwp'], help='incr: step-to-step deltas (canonical); relwp: waypoints relative to current state')
args = ap.parse_args()
if args.action_mode == 'relwp':
    sys.path.insert(0, '/home/jeonchanwook/work/smolvla/variants'); import relwp_step  # registers step for processor loading
if args.action_mode == 'se3relwp':
    sys.path.insert(0, '/home/jeonchanwook/work/smolvla/variants'); import relwp_se3_step
    import zarr as _zarr
    from scipy.spatial.transform import Rotation as _R
    _z = _zarr.open(str(__import__('eval_common').BASE_ZARR), mode='r')
    _pos = _z['data/robot0_eef_pos'][:]; _rot = _z['data/robot0_eef_rot_axis_angle'][:]; _grip = _z['data/robot0_gripper_width'][:]
    def _T_of(i):
        import numpy as _np
        T=_np.eye(4); T[:3,:3]=_R.from_rotvec(_rot[int(i)]).as_matrix(); T[:3,3]=_pos[int(i)]; return T
    def _ten(T,g):
        import numpy as _np
        return _np.concatenate([T[:3,3], T[:2,:3].reshape(6), [float(g)]]).astype(_np.float32)
    def _rot6d_mat(d6):
        import numpy as _np
        a1,a2=d6[:3],d6[3:]; b1=a1/max(_np.linalg.norm(a1),1e-8); a2p=a2-(b1@a2)*b1; b2=a2p/max(_np.linalg.norm(a2p),1e-8)
        return _np.stack([b1,b2,_np.cross(b1,b2)])

policy_config = PreTrainedConfig.from_pretrained(args.ckpt, local_files_only=True)
policy = SmolVLAPolicy.from_pretrained(args.ckpt, config=policy_config, local_files_only=True).to('cuda')
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
policy.eval()
print('loaded', args.ckpt, 'chunk', policy.config.chunk_size, 'n_action_steps', policy.config.n_action_steps, flush=True)

gt = load_gt()
ep_list = {'holdout': HOLDOUT, 'train': TRAIN_EVAL, 'holdout+train': HOLDOUT + TRAIN_EVAL}[args.episodes]
recs = []; t0 = time.time()
for ep in ep_list:
    e = gt[ep]; S = e['states']; A = e['actions']
    for t in anchors(e, args.stride):
        img = decode_img(e['imgs'][t])
        if args.action_mode == 'se3relwp':
            f = int(e['src_idx'][t]); fp = int(e['src_idx'][t-1]) if t > 0 else f
            T_t, T_p = _T_of(f), _T_of(fp)
            st10 = _ten(np.linalg.inv(T_t) @ T_p, _grip[fp])
            obs = {
                "observation.images.camera1": torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0,
                "observation.state": torch.from_numpy(st10).unsqueeze(0).cuda(),
                "task": [args.task], "robot_type": ["so_follower"],
            }
            with torch.inference_mode():
                act = post(policy.predict_action_chunk(pre(obs)))
            act = act.detach().cpu().numpy(); act = act[0] if act.ndim == 3 else act
            out10 = act[:HORIZON, :10].astype(np.float64)
            from scipy.spatial.transform import Rotation as _Rr
            pred_states = np.zeros((HORIZON, 6))
            for k in range(HORIZON):
                Trel = np.eye(4); Trel[:3,:3] = _rot6d_mat(out10[k,3:9]); Trel[:3,3] = out10[k,:3]
                Tk = T_t @ Trel
                eul = _Rr.from_matrix(Tk[:3,:3]).as_euler('ZYX')   # [yaw,pitch,roll]
                pred_states[k] = [Tk[0,3], Tk[1,3], Tk[2,3], eul[2], eul[1], out10[k,9]]
            s0 = S[t]
            for d in (3,4):
                pred_states[:,d] = s0[d] + np.angle(np.exp(1j*(pred_states[:,d]-s0[d])))
            A_pred = np.diff(np.concatenate([s0[None], pred_states],0), axis=0)
            recs.append(dict(ep=ep, t=t, s0=s0, gt_actions=A[t:t+HORIZON], gt_states=S[t+1:t+1+HORIZON], pred_actions=A_pred, pred_states=pred_states))
            continue
        st = S[t].astype(np.float32).copy()
        if args.state_mode == 'relvel':
            st[:3] = (S[t] - S[t-1]).astype(np.float32)[:3] if t > 0 else 0.0
        elif args.state_mode == 'dropxyz':
            st[:3] = 0.0
        obs = {
            "observation.images.camera1": torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0,
            "observation.state": torch.from_numpy(st).unsqueeze(0).cuda(),
            "task": [args.task], "robot_type": ["so_follower"],
        }
        if args.prev_image:
            pimg = decode_img(e['imgs'][t-1 if t > 0 else t])
            obs["observation.images.camera2"] = torch.from_numpy(pimg).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        with torch.inference_mode():
            act = post(policy.predict_action_chunk(pre(obs)))
        act = act.detach().cpu().numpy()
        act = act[0] if act.ndim == 3 else act
        out = act[:HORIZON].astype(np.float64)
        assert out.shape == (HORIZON, 6), act.shape
        s0 = S[t]
        if args.action_mode == 'relwp':
            pred_states = s0[None] + out                       # waypoints relative to current state
            A_pred = np.diff(np.concatenate([s0[None], pred_states], 0), axis=0)
        else:
            A_pred = out; pred_states = integrate(s0, A_pred)
        recs.append(dict(ep=ep, t=t, s0=s0, gt_actions=A[t:t + HORIZON], gt_states=S[t + 1:t + 1 + HORIZON], pred_actions=A_pred, pred_states=pred_states))
    print(f'ep {ep}: {len([r for r in recs if r["ep"]==ep])} chunks, elapsed {time.time()-t0:.0f}s', flush=True)

np.savez(args.out, model='smolvla', ckpt=args.ckpt, state_mode=args.state_mode, prev_image=args.prev_image, action_mode=args.action_mode,
         ep=np.array([r['ep'] for r in recs]), t=np.array([r['t'] for r in recs]),
         s0=np.stack([r['s0'] for r in recs]), gt_actions=np.stack([r['gt_actions'] for r in recs]), gt_states=np.stack([r['gt_states'] for r in recs]),
         pred_actions=np.stack([r['pred_actions'] for r in recs]), pred_states=np.stack([r['pred_states'] for r in recs]))
print('saved', args.out, len(recs))
