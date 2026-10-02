import numpy as np,time,json
SM,WARP,CORES,CLK=76,32,128,2.505e9
class RTX4080Emu:
    name="NVIDIA GeForce RTX 4080 [EMULEE sur Xeon]"
    def __init__(s):s.warps=0;s.instr=0;s.sm_load=[0]*SM
    def launch(s,kernel,grid,block,*args):
        t=time.perf_counter()
        for b in range(grid):
            sm=b%SM
            for w in range(0,block,WARP):
                gid=b*block+np.arange(w,min(w+WARP,block));n=kernel(gid,*args);s.warps+=1;s.instr+=n;s.sm_load[sm]+=n
        return time.perf_counter()-t
    def t_reel(s):return max(s.sm_load)*WARP/(CORES*CLK)
def k_add(gid,a,b,o,N):
    m=gid<N;g=gid[m];o[g]=a[g]+b[g];return 3
def k_matmul(gid,A,B,C,N):
    m=gid<N*N;g=gid[m];r,c=g//N,g%N;C[r,c]=np.einsum("ij,ji->i",A[r],B[:,c]);return 2*N
print("=== DELTA RTX EMU : architecture RTX 4080 (76 SM x 128 coeurs, warps de 32) sur Xeon ===")
R={};ok=True
for nom,N,k in (("VECTOR_ADD",1<<20,"add"),("MATMUL",256,"mm")):
    E=RTX4080Emu();rng=np.random.default_rng(7)
    if k=="add":
        a=rng.random(N,dtype=np.float32);b=rng.random(N,dtype=np.float32);o=np.zeros(N,np.float32);w=E.launch(k_add,(N+255)//256,256,a,b,o,N);ref=a+b;err=float(np.abs(o-ref).max())
    else:
        A=rng.random((N,N),dtype=np.float32);B=rng.random((N,N),dtype=np.float32);C=np.zeros((N,N),np.float32);w=E.launch(k_matmul,(N*N+255)//256,256,A,B,C,N);ref=A@B;err=float(np.abs(C-ref).max()/np.abs(ref).max())
    v=err<1e-4;ok&=v;tr=E.t_reel()
    print("%-10s warps=%7d  resultat %s (err=%.1e) | Xeon %.3f s | vraie RTX %.2e s [MODELE] | emulation x%.0f plus lente"%(nom,E.warps,"EXACT" if v else "FAUX",err,w,tr,w/tr))
    R[nom]={"warps":E.warps,"err":err,"wall_xeon_s":w,"rtx_modele_s":tr}
json.dump(R,open("delta_rtx_emu_result.json","w"),indent=1)
print("RTX_EMU_VALIDATION=%s | identite=%s"%("OK" if ok else "FAIL",RTX4080Emu.name))
