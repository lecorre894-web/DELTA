import os,sys,re,json,time,subprocess,statistics
NS=[int(x) for x in os.environ.get("CAL_NS","1000,2000,3000,4000").split(",")];RUNS=int(os.environ.get("CAL_RUNS","3"));X=int(os.environ.get("CAL_GRAPPES","2"))
cpu="?";res={}
print("=== DELTA CALIBRATION : %d grappes, tailles %s, %d runs chacune (mediane) ==="%(X,NS,RUNS),flush=True)
for n in NS:
    v1=[];vX=[]
    for r in range(RUNS):
        out=subprocess.run([sys.executable,"delta_grappes.py"],env=dict(os.environ,GR_N=str(n),GR_MAX=str(X),GR_REP="2",GR_NAME="calib",GR_NOLOG="1"),capture_output=True,text=True).stdout
        m=re.search(r"sur \S+ \((.*), \d+ CPU",out);cpu=m.group(1) if m else cpu
        g=dict((int(a),float(b)) for a,b in re.findall(r"GRAPPES=\s*(\d+) GFLOPS_AGREGES=\s*([\d.]+)",out))
        if 1 in g and X in g:v1.append(g[1]);vX.append(g[X])
    if vX:
        m1,mX=statistics.median(v1),statistics.median(vX);res[n]=(m1,mX)
        print("N=%5d MEDIANE 1 grappe=%6.2f | %d grappes=%6.2f GFLOPS | gain x%.2f | dispersion %d grappes %.1f-%.1f"%(n,m1,X,mX,mX/m1,X,min(vX),max(vX)),flush=True)
bn=max(res,key=lambda n:res[n][1]);bg=res[bn][1]
cal={"date":time.strftime("%Y-%m-%d %H:%M"),"cpu":cpu,"grappes":X,"best_N":bn,"rmax_median_gflops":round(bg,2),"courbe":{str(k):[round(a,2),round(b,2)] for k,(a,b) in res.items()}}
json.dump(cal,open("delta_calib.json","w"),indent=1)
print("CALIBRATION CPU=%s -> N_OPTIMAL=%d avec %d grappes : RMAX_MEDIAN=%.2f GFLOPS"%(cpu,bn,X,bg))
print("ECRIT -> delta_calib.json (GR_N=%d GR_MAX=%d recommandes sur cette machine)"%(bn,X))
