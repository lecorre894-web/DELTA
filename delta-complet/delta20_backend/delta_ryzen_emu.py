import numpy as np,time,json
CORES,THR,LANES=16,32,16;FP32=1674.9e9;BW=70e9
def k_add(lo,hi,a,b,o):o[lo:hi]=a[lo:hi]+b[lo:hi]
def k_mm(lo,hi,A,B,C):C[lo:hi]=A[lo:hi]@B
def launch(k,n,step,args,fast):
    t=time.perf_counter();load=np.zeros(CORES)
    if fast:k(0,n,*args);load[:]=n/CORES
    else:
        for i,lo in enumerate(range(0,n,step)):k(lo,min(n,lo+step),*args);load[(i%THR)//2]+=min(step,n-lo)
    return time.perf_counter()-t,load
print("=== DELTA RYZEN EMU : 9950X3D (16 coeurs, 32 threads SMT, AVX-512 x16) sur Xeon ===")
R={};ok=True;rng=np.random.default_rng(7)
for nom,N in (("VECTOR_ADD",1<<20),("MATMUL",512)):
    res={}
    for fast in (False,True):
        if nom=="VECTOR_ADD":
            a=rng.random(N,dtype=np.float32);b=rng.random(N,dtype=np.float32);o=np.zeros(N,np.float32);w,ld=launch(k_add,N,LANES,(a,b,o),fast);err=float(np.abs(o-(a+b)).max());tr=12*N/BW
        else:
            A=rng.random((N,N),dtype=np.float32);B=rng.random((N,N),dtype=np.float32);C=np.zeros((N,N),np.float32);w,ld=launch(k_mm,N,1,(A,B,C),fast);ref=A@B;err=float(np.abs(C-ref).max()/np.abs(ref).max());tr=2*N**3/FP32
        v=err<1e-4;ok&=v;m="LOTS" if fast else "THREAD";res[m]={"wall_s":w,"err":err,"ryzen_s":tr}
        print("%-10s %-6s %s | Xeon %.4f s | vrai Ryzen %.2e s [MESURE 2 OCT] | ecart x%.0f | charge/coeur max %.0f"%(nom,m,"EXACT" if v else "FAUX",w,tr,w/tr,ld.max()))
    g=res["THREAD"]["wall_s"]/res["LOTS"]["wall_s"];print("%-10s GAIN FONCTIONNEL = x%.1f"%(nom,g));R[nom]=dict(res,gain=g)
json.dump(R,open("delta_ryzen_emu_result.json","w"),indent=1);print("RYZEN_EMU_VALIDATION=%s"%("OK" if ok else "FAIL"))
