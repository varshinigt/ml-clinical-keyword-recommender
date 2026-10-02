import sys; sys.path.insert(0,'.')
import numpy as np, scipy.sparse as sp
from cf_model import *

# ---- 1. numerical gradient check on a tiny problem
rng=np.random.default_rng(0)
m,nb=6,10
B=(rng.random((m,nb))*(rng.random((m,nb))<0.5)).astype(np.float64)
W=rng.random((m,m)); np.fill_diagonal(W,0)
M=CFModel(W,lam=0.1); M.W=W; M.Theta=rng.normal(size=(m,m))*0.3; M.b=rng.normal(size=(m,1))*0.1
def J(Th,b):
    Z=(W*Th)@B+b
    return 0.5*((np.maximum(Z,0)-B)**2).sum()/nb+0.5*0.1*(Th**2).sum()
gT,gb=M.grads(B.astype(np.float64))
eps=1e-6; num=np.zeros_like(gT)
for i in range(m):
    for j in range(m):
        T1=M.Theta.copy();T1[i,j]+=eps;T2=M.Theta.copy();T2[i,j]-=eps
        num[i,j]=(J(T1,M.b)-J(T2,M.b))/(2*eps)
nb_=np.zeros_like(gb)
for i in range(m):
    b1=M.b.copy();b1[i]+=eps;b2=M.b.copy();b2[i]-=eps
    nb_[i]=(J(M.Theta,b1)-J(M.Theta,b2))/(2*eps)
print("grad check Theta max err:",np.abs(gT-num).max(), " b max err:",np.abs(gb-nb_).max())

# ---- 2. synthetic topic data: terms co-occur within topics
rng=np.random.default_rng(1)
m,n,T=200,6000,10
topic_terms=[rng.choice(m,25,replace=False) for _ in range(T)]
X=np.zeros((m,n),dtype=np.float32)
for j in range(n):
    for t in rng.choice(T,2,replace=False):
        sel=rng.choice(topic_terms[t],12,replace=False)
        X[sel,j]=rng.random(12)*3+0.5
X=sp.csc_matrix(X)
cols=rng.permutation(n); a,b=int(.8*n),int(.9*n)
R,Xd,Xt=X[:,np.sort(cols[:a])],X[:,np.sort(cols[a:b])],X[:,np.sort(cols[b:])]
W=cosine_weights(R,tau=-1,k=4)
model=CFModel(W,lam=1e-3,alpha=0.1)
hist=model.fit(R,epochs=5,batch=256,X_dev=Xd,log=lambda r:print({k:round(v,4) for k,v in r.items()}))
print("TEST",{k:round(v,4) for k,v in evaluate(model,Xt).items()})
