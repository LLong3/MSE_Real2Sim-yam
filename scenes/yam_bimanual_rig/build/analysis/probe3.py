import numpy as np
d = np.load('analysis/plan_slices.npz'); P = d['P'].astype(float); C = d['C'].astype(float)
lum = C.mean(1)
def rep(name, s):
    Q = P[s]; print(name, len(Q), 'x', np.round(np.percentile(Q[:,0],[5,50,95]),3), 'y', np.round(np.percentile(Q[:,1],[5,50,95]),3), 'z', np.round(np.percentile(Q[:,2],[5,50,95]),3), 'rgb', np.round(np.median(C[s],0),3))
rep('monitor desk top', (P[:,0]>-1.35)&(P[:,0]<-0.6)&(P[:,1]>1.45)&(P[:,1]<1.95)&(P[:,2]>0.6)&(P[:,2]<0.85))
rep('rig desk top', (P[:,0]>-2.9)&(P[:,0]<-1.7)&(P[:,1]>1.45)&(P[:,1]<1.95)&(P[:,2]>0.6)&(P[:,2]<0.85))
for x0,x1 in [(-1.5,-1.4),(-1.4,-1.3)]:
    rep(f'monitor desk front edge band x{x0}', (P[:,0]>x0)&(P[:,0]<x1)&(P[:,1]>1.1)&(P[:,1]<1.45)&(P[:,2]>0.68)&(P[:,2]<0.76))
rep('dark bars left', (P[:,0]>-3.7)&(P[:,0]<-3.0)&(P[:,1]>1.6)&(P[:,1]<2.3)&(P[:,2]>0.35)&(P[:,2]<1.3)&(lum<0.25))
rep('dark bars right', (P[:,0]>-1.55)&(P[:,0]<-1.2)&(P[:,1]>1.6)&(P[:,1]<2.2)&(P[:,2]>0.78)&(P[:,2]<1.3)&(lum<0.25))
rep('panel', (P[:,0]>-1.45)&(P[:,0]<-0.5)&(P[:,1]>1.9)&(P[:,1]<2.25)&(P[:,2]>0.8)&(P[:,2]<1.4)&(lum<0.25))
rep('glass blue', (P[:,0]>-4.0)&(P[:,0]<-3.05)&(P[:,1]>2.0)&(P[:,1]<2.3)&(P[:,2]>0.9)&(P[:,2]<2.0))
rep('post white', (P[:,0]>-4.1)&(P[:,0]<-3.6)&(P[:,1]>2.0)&(P[:,1]<2.3)&(P[:,2]>0.1)&(P[:,2]<2.0)&(lum>0.6))
rep('wall', (P[:,0]>-2.9)&(P[:,0]<-1.7)&(P[:,1]>2.0)&(P[:,1]<2.3)&(P[:,2]>0.9)&(P[:,2]<2.0))
rep('pole', (P[:,0]>-2.4)&(P[:,0]<-2.15)&(P[:,1]>1.25)&(P[:,1]<1.5)&(P[:,2]>1.0)&(P[:,2]<1.5))
rep('pole top', (P[:,0]>-2.45)&(P[:,0]<-2.1)&(P[:,1]>1.2)&(P[:,1]<1.6)&(P[:,2]>1.5)&(P[:,2]<2.2))
rep('carpet left near', (P[:,0]>-4.0)&(P[:,0]<-3.1)&(P[:,1]>1.2)&(P[:,1]<2.1)&(P[:,2]<0.05))
rep('floor under desk', (P[:,0]>-3.0)&(P[:,0]<-1.6)&(P[:,1]>1.4)&(P[:,1]<2.0)&(P[:,2]<0.05))
# dark bars left: cluster in plan
s=(P[:,0]>-3.7)&(P[:,0]<-2.95)&(P[:,1]>1.2)&(P[:,1]<2.3)&(P[:,2]>0.75)&(P[:,2]<1.3)&(lum<0.3)
H,xe,ye=np.histogram2d(P[s,0],P[s,1],bins=[np.arange(-3.7,-2.94,0.02),np.arange(1.2,2.31,0.02)])
idx=np.argwhere(H>np.percentile(H[H>0],90)) if (H>0).any() else []
print('dark-left high-density cells:', [(round(xe[i],2),round(ye[j],2),int(H[i,j])) for i,j in idx][:30])
