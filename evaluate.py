"""CPU evaluation and publication figures from saved arrays; no model training."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).parent
def nearest(a,b):
 a=a.reshape(-1,224);b=b.reshape(-1,224)
 return np.concatenate([np.count_nonzero(x[:,None]!=b[None],axis=2).min(1) for x in np.array_split(a,max(1,len(a)//16))])

def reachable(a,vocab):
 # Landing states have an empty foot cell with solid support below.
 solid=np.isin(a,[i for i,c in enumerate(vocab) if c not in '-oE'])
 def free(r,c): return 0<=r<14 and 0<=c<16 and not solid[r,c] and vocab[a[r,c]]!='E'
 def stand(r,c):return free(r,c) and r<13 and solid[r+1,c]
 seen={(r,0) for r in range(13) if stand(r,0)};queue=list(seen)
 while queue:
  r,c=queue.pop()
  if c==15:return True
  for dx in [-4,-3,-2,-1,1,2,3,4]:
   cc=c+dx
   if not 0<=cc<16:continue
   for rr in range(max(0,r-4),min(13,r+5)):
    if (rr,cc) in seen or not stand(rr,cc):continue
    for height in [1,2,3,4]:
     path=[(round(r+(rr-r)*t-4*height*t*(1-t)),round(c+dx*t)) for t in np.linspace(0,1,4*abs(dx)+1)]
     if all(free(y,x) for y,x in path):seen.add((rr,cc));queue.append((rr,cc));break
 return False

def checks():
 b=np.zeros((1,14,16),dtype=int);a=np.repeat(b,3,axis=0);a[1].flat[:11]=1;a[2].flat[:12]=1
 assert nearest(a,b).tolist()==[0,11,12]
 floor=np.zeros((14,16),int);floor[-1]=1
 assert reachable(floor,['-','X']);floor[:,:]=1;assert not reachable(floor,['-','X'])
 print('Metric and reachability checks passed')

def main():
 checks();res=ROOT/'results';fig=ROOT/'figures';fig.mkdir(exist_ok=True)
 manifest=json.loads((res/'manifest.json').read_text());vocab=manifest['vocab'];records=[]
 for f in sorted(res.glob('*_k*_s*.json')):
  r=json.loads(f.read_text());z=np.load(f.with_suffix('.npz'));g=z['generated'];d=nearest(g,z['train'])
  assert np.array_equal(d,z['distances']);assert len(g)==r['samples'];assert g.shape[1:]==(14,16);assert g.min()>=0 and g.max()<len(vocab)
  reach=np.array([reachable(a,vocab) for a in g]);mask=d>11
  r.update(reachability=float(reach.mean()),noncopy_count=int(mask.sum()),noncopy_reachability=float(reach[mask].mean()) if mask.any() else None)
  records.append(r)
 (res/'summary.json').write_text(json.dumps(records,indent=2))
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
 fig1,axes=plt.subplots(1,3,figsize=(9,2.3))
 for arch,color in [('transformer','#2469a1'),('diffusion','#d66b27')]:
  rows=[r for r in records if r['architecture']==arch];ks=sorted({r['k'] for r in rows})
  for ax,key in zip(axes,['copy_rate','mean_distance','reachability']):
   groups=[[r[key] for r in rows if r['k']==k] for k in ks]
   scale=1 if key=='mean_distance' else 100
   ax.errorbar(ks,[np.mean(g)*scale for g in groups],yerr=[np.std(g,ddof=1)*scale if len(g)>1 else 0 for g in groups],marker='o',capsize=3,label=arch.title(),color=color)
  if arch=='transformer':
   groups=[[r['heldout_copy_rate'] for r in rows if r['k']==k] for k in ks]
   axes[0].plot(ks,[100*np.mean(g) for g in groups],'--',color='gray',label='Held-out levels')
 for ax,label in zip(axes,['Near-copy rate (%)','Mean nearest distance (tiles)','Reachability proxy (%)']):
  ax.set_xlabel('Training levels');ax.set_ylabel(label);ax.set_xticks([3,6,9,12,15]);ax.grid(alpha=.2)
 axes[0].legend(fontsize=7);fig1.tight_layout();fig1.savefig(fig/'metrics.pdf');plt.close(fig1)
 fig2,axs=plt.subplots(2,4,figsize=(8,2.9));cmap=ListedColormap(plt.cm.tab20(np.linspace(0,1,len(vocab))))
 for row,arch in enumerate(['transformer','diffusion']):
  z=np.load(res/f'{arch}_k3_s0.npz');d=z['distances'];idxs=[int(d.argmin()),int(np.argsort(d)[len(d)//2])]
  for pair,idx in enumerate(idxs):
   sample=z['generated'][idx];train=z['train'];j=np.count_nonzero(train!=sample,axis=(1,2)).argmin()
   for col,a,title in [(pair*2,sample,f'{arch.title()}: d={d[idx]}'),(pair*2+1,train[j],'Nearest training slice')]:
    axs[row,col].imshow(a,cmap=cmap,vmin=0,vmax=len(vocab)-1,interpolation='nearest');axs[row,col].set_title(title,fontsize=8);axs[row,col].axis('off')
 fig2.tight_layout();fig2.savefig(fig/'examples.pdf');plt.close(fig2)
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
