#!/usr/bin/env python3
"""Convert gopro_umi replay_buffer.zarr (DirectoryStore, UMI-like schema) into the
original-UMI `dataset.zarr.zip` format consumed by diffusion_policy.dataset.umi_dataset.UmiDataset.

Adds robot0_demo_start_pose / robot0_demo_end_pose (episode first/last pose, repeated per frame),
which UmiDataset needs for the *_wrt_start observation keys.
"""
import argparse, sys, numpy as np, zarr
sys.path.insert(0, '/home/jeonchanwook/universal_manipulation_interface')
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--jpegxl', action='store_true', help='compress rgb with JpegXl like original UMI')
args = ap.parse_args()

src = zarr.open(args.src, mode='r')
ends = src['meta/episode_ends'][:]
n = int(ends[-1])
print('frames', n, 'episodes', len(ends))
pos = src['data/robot0_eef_pos'][:].astype(np.float32)
rot = src['data/robot0_eef_rot_axis_angle'][:].astype(np.float32)
grip = src['data/robot0_gripper_width'][:].astype(np.float32)
assert pos.shape[0] == n
pose = np.concatenate([pos, rot], axis=-1)
start_pose = np.empty_like(pose); end_pose = np.empty_like(pose)
s = 0
for e in ends:
    start_pose[s:e] = pose[s]; end_pose[s:e] = pose[e-1]; s = e

out = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
rgb = src['data/camera0_rgb']
print('rgb', rgb.shape, rgb.dtype)
# write lowdim + rgb episode by episode
s = 0
for i, e in enumerate(ends):
    ep = {
        'camera0_rgb': rgb[s:e],
        'robot0_eef_pos': pos[s:e],
        'robot0_eef_rot_axis_angle': rot[s:e],
        'robot0_gripper_width': grip[s:e],
        'robot0_demo_start_pose': start_pose[s:e],
        'robot0_demo_end_pose': end_pose[s:e],
    }
    comp = None
    chunks = {'camera0_rgb': (1, *rgb.shape[1:])}
    if args.jpegxl:
        comp = {'camera0_rgb': JpegXl(level=99, numthreads=1)}
    out.add_episode(ep, chunks=chunks, compressors=comp)
    s = e
    if i % 10 == 0:
        print('episode', i, 'done', flush=True)
print('n_episodes', out.n_episodes, 'n_steps', out.n_steps)
with zarr.ZipStore(args.out, mode='w') as zs:
    out.save_to_store(store=zs)
print('saved', args.out)
