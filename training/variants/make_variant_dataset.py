#!/usr/bin/env python3
"""Build LeRobot-v3 dataset variants of the canonical gopro_umi 10 Hz dataset for SmolVLA ablations.
  --state relvel : observation.state = [dX,dY,dZ (S[t]-S[t-1], 0 at episode start), Roll, Pitch, Grip]  (no absolute XYZ)
  --state dropxyz: observation.state = [0,0,0, Roll, Pitch, Grip]
  --prev-image   : add observation.images.prev = previous frame in episode (first frame duplicated)
Recomputes meta/stats.json, meta/episodes per-episode stats, meta/info.json features.
"""
import argparse, json, shutil, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
SRC = Path.home()/"GoPro_Umi/gopro_umi/7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--state', choices=['abs','relvel','dropxyz'], default='abs'); ap.add_argument('--prev-image', action='store_true')
ap.add_argument('--action', choices=['incr','relwp'], default='incr', help='relwp: store action=S[t+1] (absolute next state) + observation.state_abs; action stats = relative waypoints S[t+k]-S[t]')
a = ap.parse_args()
out = Path(a.out)
if out.exists(): shutil.rmtree(out)
shutil.copytree(SRC, out)
files = sorted((out/"data").glob("chunk-*/file-*.parquet"))
tables = [pq.read_table(f) for f in files]; t = tables[0]
counts = [tb.num_rows for tb in tables]
df = pa.concat_tables(tables).to_pandas()
assert (np.diff(df['index'].to_numpy()) == 1).all(), 'rows must be contiguous in file order'

S = np.stack(df['observation.state'].to_numpy()).astype(np.float32)
ep = df['episode_index'].to_numpy(); fi = df['frame_index'].to_numpy()
def stats(arr):
    q = np.quantile(arr, [0.01,0.1,0.5,0.9,0.99], axis=0)
    return dict(min=arr.min(0).tolist(), max=arr.max(0).tolist(), mean=arr.mean(0).tolist(), std=arr.std(0).tolist(), count=[int(len(arr))], q01=q[0].tolist(), q10=q[1].tolist(), q50=q[2].tolist(), q90=q[3].tolist(), q99=q[4].tolist())
info = json.load(open(out/"meta/info.json")); st = json.load(open(out/"meta/stats.json"))
names = info['features']['observation.state']['names']['axes']
if a.state != 'abs':
    new = S.copy()
    if a.state == 'relvel':
        prev = np.roll(S, 1, axis=0); first = np.r_[True, ep[1:] != ep[:-1]] | (fi == 0)
        d = S - prev; d[first] = 0.0
        new[:, :3] = d[:, :3]; names = ['dX_prev','dY_prev','dZ_prev','Roll','Pitch','Gripper']
    else:
        new[:, :3] = 0.0; names = ['zero','zero','zero','Roll','Pitch','Gripper']
    df['observation.state'] = list(new.astype(np.float32)); S = new
    info['features']['observation.state']['names']['axes'] = names
    st['observation.state'] = stats(S)
if a.prev_image:
    col = df['observation.images.top'].to_numpy()
    prev_col = []
    for i in range(len(df)):
        j = i-1 if (i > 0 and ep[i-1] == ep[i]) else i
        prev_col.append(col[j])
    df['observation.images.prev'] = prev_col
    info['features']['observation.images.prev'] = dict(info['features']['observation.images.top'])
    st['observation.images.prev'] = st['observation.images.top']
S_abs = np.stack(t.to_pandas()['observation.state'].to_numpy()).astype(np.float32) if False else None
if a.action == 'relwp':
    S0 = np.stack(pa.concat_tables(tables).to_pandas()['observation.state'].to_numpy()).astype(np.float32)  # original absolute state
    A = np.stack(df['action'].to_numpy()).astype(np.float32)                 # original incremental A[t]=S[t+1]-S[t]
    S_next = S0 + A                                                            # absolute next state (valid also at episode end)
    df['action'] = list(S_next.astype(np.float32))
    df['observation.state_abs'] = list(S0.astype(np.float32))
    info['features']['action']['names']['axes'] = ['X_next','Y_next','Z_next','Roll_next','Pitch_next','Gripper_next']
    info['features']['observation.state_abs'] = {'dtype':'float32','shape':[6],'names':{'axes':['X','Y','Z','Roll','Pitch','Gripper']}}
    # relative-waypoint stats over all anchors: S[t+k]-S[t], k=1..15 within episode
    rel = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]; Se = np.concatenate([S0[idx], S_next[idx][-1:]], 0)  # N+1 states
        for k in range(1, 16):
            rel.append(Se[k:] - Se[:-k])
    rel = np.concatenate(rel, 0)
    st['action'] = stats(rel); st['observation.state_abs'] = stats(S0)
    S_abs = S0; REL = rel
# write parquet preserving schema for existing cols
schema = t.schema
fields = list(schema)
if a.action == 'relwp':
    fields.append(pa.field('observation.state_abs', schema.field('observation.state').type))
if a.prev_image:
    fields.append(pa.field('observation.images.prev', schema.field('observation.images.top').type))
new_t = pa.Table.from_pandas(df, schema=pa.schema(fields, metadata=schema.metadata), preserve_index=False)
# update huggingface metadata features (for datasets lib) if present
md = dict(schema.metadata or {})
if b'huggingface' in md:
    hf = json.loads(md[b'huggingface']); 
    if a.prev_image and 'features' in hf.get('info', {}): hf['info']['features']['observation.images.prev'] = hf['info']['features']['observation.images.top']
    if a.action == 'relwp' and 'features' in hf.get('info', {}): hf['info']['features']['observation.state_abs'] = hf['info']['features']['observation.state']
    md[b'huggingface'] = json.dumps(hf).encode(); new_t = new_t.replace_schema_metadata(md)
off = 0
for f, n in zip(files, counts):
    pq.write_table(new_t.slice(off, n), f); off += n
json.dump(info, open(out/"meta/info.json", 'w'), indent=4); json.dump(st, open(out/"meta/stats.json", 'w'), indent=4)
# per-episode stats
epf = sorted((out/"meta/episodes").glob("chunk-*/file-*.parquet")); assert len(epf) == 1
et = pq.read_table(epf[0]).to_pandas()
for k in ['min','max','mean','std','count','q01','q10','q50','q90','q99']:
    vals = []
    for e in et['episode_index']:
        s = stats(S[ep == e])[k]; vals.append(np.array(s, dtype=np.float32 if k != 'count' else np.int64))
    et[f'stats/observation.state/{k}'] = vals
    if a.prev_image:
        et[f'stats/observation.images.prev/{k}'] = et[f'stats/observation.images.top/{k}']
    if a.action == 'relwp':
        et[f'stats/action/{k}'] = [np.array(stats(REL)[k], dtype=np.float32 if k != 'count' else np.int64)] * len(et)  # global rel-waypoint stats
        et[f'stats/observation.state_abs/{k}'] = [np.array(stats(S_abs[ep == e])[k], dtype=np.float32 if k != 'count' else np.int64) for e in et['episode_index']]
pq.write_table(pa.Table.from_pandas(et, preserve_index=False), epf[0])
print('wrote', out, 'state', a.state, 'prev_image', a.prev_image, 'action', a.action, 'names', names)
