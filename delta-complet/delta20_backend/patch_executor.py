import glob
n=0
for f in sorted(glob.glob("*.py")):
    if f in("patch_executor.py","probe_executor.py"):continue
    s=open(f).read()
    if "SamplerV2(" not in s or "_ExecSampler" in s:continue
    L=s.split("\n")
    for i,l in enumerate(L):
        if "import" in l and "SamplerV2" in l and "#" not in l:L[i]=l+";from qiskit_ibm_runtime.executor_sampler import Sampler as _ExecSampler"
    s="\n".join(L)
    if "_ExecSampler" not in s:print("SANS_IMPORT (a traiter a la main)",f);continue
    k=s.count("SamplerV2(");s=s.replace("SamplerV2(","_ExecSampler(")
    compile(s,f,"exec");open(f,"w").write(s);n+=1;print("MIGRE %-28s %d appel(s)"%(f,k))
print("FICHIERS_MIGRES=%d"%n)
