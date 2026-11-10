import os,time,json,random,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

def distance(a,b):
 return np.concatenate([np.min(np.count_nonzero(x[:,None,:]!=b[None,:,:],axis=2),axis=1) for x in np.array_split(a,max(1,len(a)//16))])

class UNet(nn.Module):
 def __init__(self,v):
  super().__init__();self.a=nn.Conv2d(v+1,32,3,padding=1);self.b=nn.Conv2d(32,64,3,padding=1);self.c=nn.Conv2d(96,32,3,padding=1);self.out=nn.Conv2d(32,v,1)
 def forward(self,x,t):
  h=F.silu(self.a(torch.cat([x,t[:,None,None,None].expand(-1,1,14,16)],1)))
  lo=F.silu(self.b(F.avg_pool2d(h,2)))
  return self.out(F.silu(self.c(torch.cat([h,F.interpolate(lo,size=(14,16),mode='nearest')],1))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',type=int,default=1);args=p.parse_args()
 torch.set_num_threads(4);assert torch.cuda.is_available()
 root=Path(__file__).parent;out=root/'results';out.mkdir(exist_ok=True)
 files=sorted((root/'data').glob('*.txt'));assert len(files)>=18
 levels=[f.read_text().splitlines() for f in files];assert all(len(l)==14 for l in levels)
 vocab=sorted(set(''.join(''.join(l) for l in levels)));v=len(vocab);enc={c:i for i,c in enumerate(vocab)}
 slices=[np.array([[enc[c] for c in row] for row in l],dtype=np.int64) for l in levels]
 slices=[np.stack([a[:,j:j+16] for j in range(a.shape[1]-15)]) for a in slices]
 split=np.random.default_rng(2026).permutation(len(files));held_ids=split[:5];train_ids=split[5:];assert len(train_ids)>=15
 held=np.concatenate([slices[i] for i in held_ids]);pool=[slices[i] for i in train_ids]
 (out/'manifest.json').write_text(json.dumps(dict(files=[f.name for f in files],heldout=[files[i].name for i in held_ids],vocab=vocab,gpu=torch.cuda.get_device_name(),torch=torch.__version__,vglc_revision='0e86a8f31f20ecad4eaa5741ff061af88767f7fb'),indent=2))
 schedule=[(a,k,s) for s in range(5) for k in [3,6,9,12,15] for a in ['transformer','diffusion']]
 for arch,k,seed in schedule:
  start=time.monotonic();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  name=f'{arch}_k{k}_s{seed}'
  if (out/(name+'.json')).exists():continue
  order=np.random.default_rng(seed).permutation(len(pool));train=np.concatenate([pool[i] for i in order[:k]])
  tx=torch.tensor(train,device='cuda');losses=[]
  if arch=='transformer':
   model=GPT2LMHeadModel.from_pretrained(str(root/'distilgpt2')).cuda();model.resize_token_embeddings(v+1)
   bos=v;opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.)
  else:
   model=UNet(v).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-3)
  beta=torch.linspace(.0001,.02,100,device='cuda');alpha=1-beta;abar=torch.cumprod(alpha,0)
  model.train();steps=0
  # Equal updates within architecture across k; bound each run independently.
  target=300 if arch=='transformer' else 1500
  while steps<target and time.monotonic()-start<290:
   b=tx[torch.randint(len(tx),(16,),device='cuda')];opt.zero_grad(set_to_none=True)
   if arch=='transformer':
    seq=b.transpose(1,2).reshape(16,224);seq=torch.cat([torch.full((16,1),bos,device='cuda'),seq],1)
    loss=model(input_ids=seq,labels=seq).loss
   else:
    x=F.one_hot(b,v).permute(0,3,1,2).float()*2-1;t=torch.randint(100,(16,),device='cuda');noise=torch.randn_like(x);a=abar[t,None,None,None]
    loss=F.mse_loss(model(a.sqrt()*x+(1-a).sqrt()*noise,t.float()/99),noise)
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss));steps+=1
  model.eval();generated=[]
  with torch.no_grad():
   for _ in range(20):
    if arch=='transformer':
     prefix=torch.full((50,1),bos,device='cuda')
     seq=model.generate(prefix,attention_mask=torch.ones_like(prefix),max_new_tokens=224,do_sample=True,temperature=1.,top_k=0,pad_token_id=bos,eos_token_id=None,bad_words_ids=[[bos]],use_cache=True)
     x=seq[:,1:].reshape(50,16,14).transpose(1,2)
    else:
     x=torch.randn(50,v,14,16,device='cuda')
     for t in reversed(range(100)):
      eps=model(x,torch.full((50,),t/99,device='cuda'));mean=(x-beta[t]/(1-abar[t]).sqrt()*eps)/alpha[t].sqrt()
      variance=beta[t]*(1-abar[t-1])/(1-abar[t]) if t else 0
      x=mean+(variance.sqrt()*torch.randn_like(x) if t else 0)
     x=x.argmax(1)
    generated.append(x.cpu().numpy())
  gen=np.concatenate(generated);d=distance(gen.reshape(-1,224),train.reshape(-1,224));hd=distance(held.reshape(-1,224),train.reshape(-1,224))
  name=f'{arch}_k{k}_s{seed}';np.savez_compressed(out/(name+'.npz'),generated=gen,train=train,distances=d,heldout_distances=hd,loss=losses)
  if k==3 and seed==0:torch.save(model.state_dict(),out/(name+'.pt'))
  result=dict(architecture=arch,k=k,seed=seed,updates=steps,target_updates=target,parameters=sum(p.numel() for p in model.parameters()),samples=len(gen),copy_rate=float((d<=11).mean()),exact_rate=float((d==0).mean()),heldout_copy_rate=float((hd<=11).mean()),mean_distance=float(d.mean()),final_loss=losses[-1],seconds=time.monotonic()-start,training_levels=[files[train_ids[i]].name for i in order[:k]],status='complete' if steps==target else 'time_limited')
  (out/(name+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
  del model,opt,tx;torch.cuda.empty_cache()

if __name__=='__main__':main()
