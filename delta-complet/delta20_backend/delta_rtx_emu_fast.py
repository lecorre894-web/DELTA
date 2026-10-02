import numpy as np,time,json
SM,WARP,CORES,CLK=76,32,128,2.505e9
def k_add(gid,a,b,o,N):
    m=gid<N;g=gid[m];o[g]=a[g]+b[g];return 3
def k_matmul(gid,A,B,C,N):
    m=gid<N*N;g=gid[m];r,c=g//N,g%N;C[r,c]=np.einsum("ij,ji->i",A[r],B[:,c]);return 2*N
def launch(kernel,grid,block,*args,fast):
    t=time.perf_counter();wpb=block//WARP;W=grid*wpb
    if fast:n=kernel(np.arange(grid*block),*args)
    else:
        for w in range(W):n=kernel(np.arange(w*WARP,(w+1)*WARP),*args)
    load=np.bincount((np.arange(W)//wpb)%SM,minlength=SM)*n
    return time.perf_counter()-t,W,load.max()*WARP/(CORES*CLK)
print("=== DELTA RTX EMU : warp par warp vs FONCTIONNEL par lots (memes kernels) ===")
R={};ok=True;rng=np.random.default_rng(7)
for nom,N in (("VECTOR_ADD",1<<20),("MATMUL",256)):
    res={}
    for mode in (False,True):
        if nom=="VECTOR_ADD":
            a=rng.random(N,dtype=np.float32);b=rng.random(N,dtype=np.float32);o=np.zeros(N,np.float32);w,W,tr=launch(k_add,(N+255)//256,256,a,b,o,N,fast=mode);err=float(np.abs(o-(a+b)).max())
        else:
            A=rng.random((N,N),dtype=np.float32);B=rng.random((N,N),dtype=np.float32);C=np.zeros((N,N),np.float32);w,W,tr=launch(k_matmul,(N*N+255)//256,256,A,B,C,N,fast=mode);ref=A@B;err=float(np.abs(C-ref).max()/np.abs(ref).max())
        v=err<1e-4;ok&=v;res["lots" if mode else "warp"]={"wall_s":w,"err":err,"warps":W,"rtx_modele_s":tr}
        print("%-10s %-5s warps=%6d %s | Xeon %.4f s | vraie RTX %.2e s [MODELE] | ecart x%.0f"%(nom,"LOTS" if mode else "WARP",W,"EXACT" if v else "FAUX",w,tr,w/tr))
    g=res["warp"]["wall_s"]/res["lots"]["wall_s"];print("%-10s GAIN FONCTIONNEL = x%.1f"%(nom,g));R[nom]=dict(res,gain=g)
json.dump(R,open("delta_rtx_emu_fast_result.json","w"),indent=1);print("RTX_EMU_FAST_VALIDATION=%s"%("OK" if ok else "FAIL"))
