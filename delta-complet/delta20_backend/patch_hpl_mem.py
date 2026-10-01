f="delta_hpl.py";s=open(f).read()
a='NS=[int(x) for x in os.environ.get("HPL_NS","1000,2000,4000,8000,12000").split(",")]'
b='''def avail_mb():
    if os.environ.get("HPL_MEM_MB"):return int(os.environ["HPL_MEM_MB"])
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):return int(l.split()[1])//1024
    return 2048
AV=avail_mb();BUDGET=0.35*AV*2**20;NMAX=int((BUDGET/(8*3))**0.5)//500*500
NS=[int(x) for x in os.environ["HPL_NS"].split(",")] if os.environ.get("HPL_NS") else [n for n in (1000,2000,4000,6000,8000,12000,16000) if n<=NMAX] or [max(500,NMAX)]
print("MEM_AVAILABLE=%d MB BUDGET_35%%=%d MB N_MAX_SUR=%d (DELTA sans memoire dediee : bench adaptatif)"%(AV,BUDGET/2**20,NMAX))'''
c='n=4096;A=rng.random((n,n))'
d='n=min(4096,max(512,NMAX));A=rng.random((n,n))'
e='k=100000;'
g='k=int(min(100000,max(1000,BUDGET/(8*32*32*3))));'
for x,y in ((a,b),(c,d),(e,g)):
    assert s.count(x)==1,"ANCRE "+x[:30];s=s.replace(x,y)
open(f,"w").write(s);print("PATCH HPL MEMOIRE OK")
