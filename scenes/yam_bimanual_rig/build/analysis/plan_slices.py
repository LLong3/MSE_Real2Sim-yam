import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'metric'))
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rigcam import metric
import common
m = metric(); S = np.array(m['transform_reference_to_metric']['S']); t = np.array(m['transform_reference_to_metric']['t'])
V, MV, C = common.points()
P = np.asarray(V).reshape(-1,3); Cc = np.asarray(C).reshape(-1,3); mv = MV.reshape(-1)
ok = np.isfinite(P).all(1) & mv
P = P[ok]*S + t; Cc = Cc[ok]
if Cc.max() > 1.5: Cc = Cc/255.
sel = (P[:,0]>-5)&(P[:,0]<0)&(P[:,1]>0)&(P[:,1]<4)
P, Cc = P[sel], Cc[sel]
rng = np.random.default_rng(0); idx = rng.choice(len(P), min(len(P), 1500000), replace=False); P, Cc = P[idx], Cc[idx]
bands = [(-0.05,0.08,'floor 0-8cm'),(0.08,0.6,'0.08-0.6'),(0.6,0.78,'desk 0.6-0.78'),(0.78,1.2,'0.78-1.2'),(1.2,1.8,'1.2-1.8'),(1.8,3.0,'1.8-3.0')]
fig, axs = plt.subplots(2,3, figsize=(24,14))
x0,x1 = m['desk']['box_x_m']; y0,y1 = m['desk']['box_y_m']
for ax,(a,b,name) in zip(axs.flat,bands):
    s = (P[:,2]>=a)&(P[:,2]<b)
    ax.scatter(P[s,0],P[s,1],c=Cc[s],s=0.3)
    ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'g-',lw=0.8)
    ax.axhline(y1+0.035,color='r',lw=0.6); ax.axvline(m['planes']['back_wall']['wall_end_x_m'],color='m',lw=0.6)
    ax.set_title(name); ax.set_aspect('equal'); ax.set_xlim(-5,0); ax.set_ylim(0,4); ax.grid(True,lw=0.3)
    ax.set_xticks(np.arange(-5,0.01,0.25)); ax.set_yticks(np.arange(0,4.01,0.25)); ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
np.savez_compressed(Path(sys.argv[1]).with_suffix('.npz'), P=P.astype(np.float32), C=Cc.astype(np.float16))
