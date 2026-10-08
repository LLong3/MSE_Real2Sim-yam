import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rigcam import *
m = metric(); K, T, _ = scene_camera(m)
def bp(uv, n, d): return np.round(backproject_plane([uv], K, T, n, d)[0], 3)
print('frosted bottom on glass plane y=2.155:', [bp(uv,(0,1,0),-2.155) for uv in [(40,325),(150,326),(235,328)]])
print('post bottom on floor', bp((160,445),(0,0,1),0), bp((180,440),(0,0,1),0))
print('monitor desk left edge on z=0.72:', [bp(uv,(0,0,1),-0.72) for uv in [(810,353),(850,410),(900,480),(960,560)]])
print('rig desk right edge on z=0.72:', [bp(uv,(0,0,1),-0.72) for uv in [(778,357),(850,470),(920,590)]])
print('panel bottom (810,343) on z?:', bp((810,343),(0,1,0),-2.09), 'panel top-left (953,193):', bp((953,193),(0,1,0),-2.09))
print('panel bottom-right (850,340) on y=2.09', bp((850,340),(0,1,0),-2.09))
print('white box (900,345) y=2.09', bp((900,345),(0,1,0),-2.09), bp((955,372),(0,0,1),-0.72))
print('bracket top (915,318) y=2.05', bp((915,318),(0,1,0),-2.05), 'bottom (830,390) z=.72', bp((830,390),(0,0,1),-0.72))
print('dark bracket left (180,360) on y=2.10', bp((180,360),(0,1,0),-2.10), bp((200,420),(0,1,0),-2.10), bp((150,355),(0,1,0),-2.10))
# wall/desk crease pixel rows
for x in [-3.0,-2.5,-2.0,-1.6]:
    uv,_ = project([[x,2.0839,0.72],[x,2.1189,0.72],[x,2.1189,0.0]],K,T); print('x',x,'desk back edge v',round(uv[0,1],1),'wall@desk-height v',round(uv[1,1],1),'u',round(uv[0,0],1))
