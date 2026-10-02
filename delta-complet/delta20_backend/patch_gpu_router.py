s=open("delta_gpu.py").read();a='    g,res=hpl(HN)'
assert s.count(a)==1 and "amp_actif" not in s,"ANCRE ou deja patche"
add='    r["amp_actif"]=r["gemm_amp"]>1.1*r["gemm_fp32"];r["precision_route"]="AMP" if r["amp_actif"] else "FP32"\n    print("ROUTEUR precision=%s : AMP %s (%.1f vs FP32 %.1f GFLOPS) | capacite=%s"%(r["precision_route"],"ACTIVE" if r["amp_actif"] else "COUPEE car plus lente",r["gemm_amp"],r["gemm_fp32"],torch.backends.cpu.get_cpu_capability() if DEV=="cpu" else "CUDA"))\n'
open("delta_gpu.py","w").write(s.replace(a,add+a));print("PATCH_ROUTEUR=OK")
