import os,sys,json,time,platform,math
import torch
DEV="cuda" if torch.cuda.is_available() else "cpu";NAME=torch.cuda.get_device_name(0) if DEV=="cuda" else (platform.processor() or "CPU")
def sync():
    if DEV=="cuda":torch.cuda.synchronize()
def gemm(n,dt,amp=False,rep=5):
    a=torch.randn(n,n,device=DEV,dtype=torch.float32 if amp else dt);b=torch.randn(n,n,device=DEV,dtype=torch.float32 if amp else dt)
    def f():
        if amp:
            with torch.autocast(device_type=DEV,dtype=torch.float16 if DEV=="cuda" else torch.bfloat16):return a@b
        return a@b
    f();sync();t=time.perf_counter()
    for _ in range(rep):c=f()
    sync();w=(time.perf_counter()-t)/rep;return 2*n**3/w/1e9
def hpl(n):
    A=torch.rand(n,n,device=DEV,dtype=torch.float64)-0.5;b=torch.rand(n,device=DEV,dtype=torch.float64)-0.5
    torch.linalg.solve(A,b);sync();t=time.perf_counter();x=torch.linalg.solve(A,b);sync();w=time.perf_counter()-t
    r=(torch.linalg.norm(A@x-b,float("inf"))/(torch.finfo(torch.float64).eps*(torch.linalg.norm(A,float("inf"))*torch.linalg.norm(x,float("inf"))+torch.linalg.norm(b,float("inf")))*n)).item()
    return (2/3*n**3+2*n**2)/w/1e9,r
def physique(nb=200,steps=1000):
    try:import pybullet as p
    except ImportError:return None
    p.connect(p.DIRECT);p.setGravity(0,0,-9.81);p.createMultiBody(0,p.createCollisionShape(p.GEOM_PLANE))
    sh=p.createCollisionShape(p.GEOM_SPHERE,radius=0.1);ids=[p.createMultiBody(1,sh,basePosition=[(i%20)*0.25,(i//20)*0.25,1+0.01*i]) for i in range(nb)]
    t=time.perf_counter()
    for _ in range(steps):p.stepSimulation()
    w=time.perf_counter()-t;z=sum(p.getBasePositionAndOrientation(i)[0][2] for i in ids)/nb;p.disconnect()
    return steps/w,nb,z
if __name__=="__main__":
    N=int(os.environ.get("GPU_N","4096" if DEV=="cuda" else "1024"));HN=int(os.environ.get("GPU_HPL_N","8192" if DEV=="cuda" else "2000"))
    print("=== DELTA GPU VIRTUEL : routeur %s (%s) | torch %s ==="%(DEV.upper(),NAME,torch.__version__))
    r={"date":time.strftime("%Y-%m-%d %H:%M"),"device":DEV,"name":NAME}
    r["gemm_fp64"]=gemm(N,torch.float64);r["gemm_fp32"]=gemm(N,torch.float32);r["gemm_amp"]=gemm(N,None,amp=True)
    print("GEMM N=%d  FP64=%9.1f GFLOPS  FP32=%9.1f  AMP(%s, successeur natif d'Apex)=%9.1f"%(N,r["gemm_fp64"],r["gemm_fp32"],"FP16" if DEV=="cuda" else "BF16",r["gemm_amp"]))
    r["amp_actif"]=r["gemm_amp"]>1.1*r["gemm_fp32"];r["precision_route"]="AMP" if r["amp_actif"] else "FP32"
    print("ROUTEUR precision=%s : AMP %s (%.1f vs FP32 %.1f GFLOPS) | capacite=%s"%(r["precision_route"],"ACTIVE" if r["amp_actif"] else "COUPEE car plus lente",r["gemm_amp"],r["gemm_fp32"],torch.backends.cpu.get_cpu_capability() if DEV=="cpu" else "CUDA"))
    g,res=hpl(HN);r["hpl_fp64"]=g;r["hpl_residu"]=res
    print("HPL FP64 N=%d  %.1f GFLOPS  RESIDU=%.4f %s"%(HN,g,res,"PASSED" if res<16 else "FAILED"))
    ph=physique()
    if ph:r["physique_steps_s"],nb,z=ph;print("PHYSIQUE PyBullet (CPU) %d spheres : %.0f pas/s | hauteur moyenne finale=%.3f m (sol + rayon 0.1 attendu)"%(nb,ph[0],z))
    open("delta_gpu_results.jsonl","a").write(json.dumps(r)+"\n")
    print("REFERENCE CPU Codespace HPL ~45-49 GFLOPS -> facteur x%.1f"%(g/47))
    print("GPU_VALIDATION=%s"%("OK" if res<16 and (not ph or abs(ph[2]-0.1)<0.05) else "FAIL"))
