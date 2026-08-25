#!/usr/bin/env python3
"""Build a LeRobot-v3 dataset with UMI-style SE(3) relative representation from the canonical 10Hz dataset.

Uses the canonical dataset's source mapping to pull FULL rotation (axis-angle) from the base zarr
(yaw was only dropped downstream), then:
  observation.state      = [ inv(T_t)*T_{t-1} -> pos(3)+rot6d(6), grip(1) ]  (10D; identity at episode start)
  observation.state_abs  = [ T_t -> pos(3)+rot6d(6), grip(1) ]               (10D; anchor for relwp step)
  action (per row)       = [ T_{t+1} -> pos(3)+rot6d(6), grip(1) ]           (10D absolute next pose;
                            training processor converts chunk to inv(T_t)*T_{t+k} rel waypoints)
Action stats = SE(3) relative waypoints inv(T_t)*T_{t+k}, k=1..15.
"""
import argparse, json, shutil, numpy as np, pyarrow as pa, pyarrow.parquet as pq, zarr
from pathlib import Path
from scipy.spatial.transform import Rotation as R
SRC = Path.home()/"GoPro_Umi/gopro_umi/7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
BASE = Path.home()/"GoPro_Umi/gopro_umi/7_storage/datasets/202608161903/02_zarr/replay_buffer.zarr"

def pose_to_mat(pos, rotvec):
    T = np.tile(np.eye(4), (len(pos),1,1)); T[:,:3,:3]=R.from_rotvec(rotvec).as_matrix(); T[:,:3,3]=pos; return T
def mat_to_10d(T, grip):
    return np.concatenate([T[:,:3,3], T[:,:2,:3].reshape(len(T),6), grip], -1).astype(np.float32)  # rot6d = first two rows (UMI convention)
def rel(Ta, Tb):  # inv(Ta) @ Tb
    return np.linalg.inv(Ta) @ Tb

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
ap.add_argument('--latency-aug', action='store_true', help='images sampled at t-L, L~N(88ms,14ms) per frame (observation_latency_model.md); state/actions unchanged')
a = ap.parse_args()
out = Path(a.out)
if out.exists(): shutil.rmtree(out)
shutil.copytree(SRC, out)
files = sorted((out/"data").glob("chunk-*/file-*.parquet"))
tables=[pq.read_table(f) for f in files]; counts=[t.num_rows for t in tables]; schema=tables[0].schema
df = pa.concat_tables(tables).to_pandas(); assert (np.diff(df['index'].to_numpy())==1).all()
m = pq.read_table(out/"meta/step9a_source_to_10hz_mapping.parquet").to_pandas().sort_values('output_index')
src_idx = m['source_global_index'].to_numpy(); src_next = m['source_next_global_index'].to_numpy()
z = zarr.open(str(BASE),'r'); pos=z['data/robot0_eef_pos'][:]; rot=z['data/robot0_eef_rot_axis_angle'][:]; grip=z['data/robot0_gripper_width'][:]
ep = df['episode_index'].to_numpy(); fi = df['frame_index'].to_numpy(); first = np.r_[True, ep[1:]!=ep[:-1]] | (fi==0)
T_t   = pose_to_mat(pos[src_idx], rot[src_idx]);      G_t   = grip[src_idx]
T_n   = pose_to_mat(pos[src_next], rot[src_next]);    G_n   = grip[src_next]
prev  = np.where(first, np.arange(len(df)), np.arange(len(df))-1)
T_p   = T_t[prev]; G_p = G_t[prev]
state    = mat_to_10d(rel(T_t, T_p), G_p)                      # prev pose in current frame (identity+G at start)
state_abs= mat_to_10d(T_t, G_t)
act_abs  = mat_to_10d(T_n, G_n)                                 # absolute next pose (row action)
df['observation.state']=list(state); df['observation.state_abs']=list(state_abs); df['action']=list(act_abs)
if a.latency_aug:
    import io as _io
    from PIL import Image as _Image
    rgbz = z['data/camera0_rgb']
    rng = np.random.default_rng(0)
    ep_start_src = {}
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        ep_start_src[e] = int(src_idx[idx[0]] - 0)  # first source frame of episode at 10Hz anchor 0
    lag_frames = np.clip(np.round(rng.normal(0.088, 0.014, size=len(df)) * 59.94).astype(int), 0, 12)
    new_imgs = []
    for i in range(len(df)):
        j = int(src_idx[i]) - int(lag_frames[i])
        lo = ep_start_src[ep[i]]
        j = max(j, lo)
        arr = rgbz[j]
        buf = _io.BytesIO(); _Image.fromarray(arr).save(buf, format='PNG')
        new_imgs.append({'bytes': buf.getvalue(), 'path': None})
        if i % 2000 == 0: print('latency-aug img', i, flush=True)
    df['observation.images.top'] = new_imgs
# rel-waypoint stats over k=1..15 in SE(3)
rels=[]
for e in np.unique(ep):
    idx=np.where(ep==e)[0]; Te=np.concatenate([T_t[idx], T_n[idx][-1:]],0); Ge=np.concatenate([G_t[idx], G_n[idx][-1:]],0)
    for k in range(1,16): rels.append(mat_to_10d(rel(Te[:-k], Te[k:]), Ge[k:]))
REL=np.concatenate(rels,0)
def stats(arr):
    q=np.quantile(arr,[0.01,0.1,0.5,0.9,0.99],axis=0)
    return dict(min=arr.min(0).tolist(),max=arr.max(0).tolist(),mean=arr.mean(0).tolist(),std=arr.std(0).tolist(),count=[int(len(arr))],q01=q[0].tolist(),q10=q[1].tolist(),q50=q[2].tolist(),q90=q[3].tolist(),q99=q[4].tolist())
info=json.load(open(out/"meta/info.json")); st=json.load(open(out/"meta/stats.json"))
ax10=['x','y','z','r11','r12','r13','r21','r22','r23','grip']
for key,arr in [('observation.state',state),('observation.state_abs',state_abs),('action',act_abs)]:
    info['features'][key]={'dtype':'float32','shape':[10],'names':{'axes':ax10}}
st['observation.state']=stats(state); st['observation.state_abs']=stats(state_abs); st['action']=stats(REL)
fields=[f if f.name not in ('observation.state','action') else pa.field(f.name, pa.list_(pa.float32(),10)) for f in schema]
fields.append(pa.field('observation.state_abs', pa.list_(pa.float32(),10)))
md=dict(schema.metadata or {})
if b'huggingface' in md:
    hf=json.loads(md[b'huggingface']); feats=hf.get('info',{}).get('features',{})
    for k in ['observation.state','action']:
        if k in feats and 'length' in feats[k]: feats[k]['length']=10
    feats['observation.state_abs']=json.loads(json.dumps(feats.get('observation.state',{})))
    md[b'huggingface']=json.dumps(hf).encode()
new_t=pa.Table.from_pandas(df, schema=pa.schema(fields, metadata=md), preserve_index=False)
off=0
for f,n in zip(files,counts): pq.write_table(new_t.slice(off,n), f); off+=n
json.dump(info, open(out/"meta/info.json",'w'), indent=4); json.dump(st, open(out/"meta/stats.json",'w'), indent=4)
epf=sorted((out/"meta/episodes").glob("chunk-*/file-*.parquet")); et=pq.read_table(epf[0]).to_pandas()
for k in ['min','max','mean','std','count','q01','q10','q50','q90','q99']:
    dt=np.float32 if k!='count' else np.int64
    et[f'stats/observation.state/{k}']=[np.array(stats(state[ep==e])[k],dtype=dt) for e in et['episode_index']]
    et[f'stats/observation.state_abs/{k}']=[np.array(stats(state_abs[ep==e])[k],dtype=dt) for e in et['episode_index']]
    et[f'stats/action/{k}']=[np.array(stats(REL)[k],dtype=dt)]*len(et)
pq.write_table(pa.Table.from_pandas(et,preserve_index=False), epf[0])
print('wrote',out,'state10/action10 SE(3); REL std',np.round(REL.std(0),4))
