"""DELTA CONTROL UI : decouverte du depot. Lit les fichiers, n'execute JAMAIS de code.
Regle : une classification n'est jamais inventee. Absente -> UNKNOWN, sauf regle explicite (source indiquee)."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
SKIP=("/.git/","node_modules","/delta_ui/","__pycache__","_x512_build",".venv")
TYPES=[("QPU",("qpu","ibm","quantum","kingston")),("GPU",("gpu","rtx","cuda")),("DSPC",("dspc",)),
       ("CPU",("cpu","xeon","ryzen","x512","avx","risc","socle")),("CACHE",("cache","index","marque","signature")),
       ("BENCH",("bench",)),("DELTA",("delta","hybride"))]
def type_de(nom):
    n=nom.lower()
    for t,cles in TYPES:
        if any(k in n for k in cles):return t
    return "OTHER"
def num(v):return isinstance(v,(int,float)) and not isinstance(v,bool)
def premier(d,cles):
    for k in cles:
        if k in d and d[k] is not None:return d[k]
    return None
UTILISES=set()
def aplatir(d,prof=0,pre=""):
    """mesures parfois rangees dans des sous-blocs (metrics, timing, result...) : on descend de 2 niveaux, sans rien inventer"""
    out={}
    if not isinstance(d,dict) or prof>2:return out
    for k,v in d.items():
        k2=str(k).lower()
        if isinstance(v,dict):
            for kk,vv in aplatir(v,prof+1,pre+k2+".").items():out.setdefault(kk,vv)
        else:out.setdefault(k2,v);out.setdefault(pre+k2,v)
    return out
T_US=["time_us","temps_us","duration_us","elapsed_us","latency_us","wall_us","runtime_us","us","microseconds"]
T_MS=["time_ms","temps_ms","duration_ms","elapsed_ms","latency_ms","wall_ms","runtime_ms","total_ms","ms","milliseconds"]
T_S=["time_s","temps_s","duration_s","elapsed_s","elapsed","seconds","duration","runtime","wall_time","total_s","latency_s","execution_time","med"]
CHAMPS=(("gflops",["gflops","gflop_s","gflops_s"]),("qubits",["qbits","qubits","physicalqubits","physical_qubits","num_qubits","n_qubits"]),
        ("shots",["shots","nshots","n_shots"]),("threads",["threads","nthreads"]),("cores",["cpu_cores","cores","coeurs","ncores"]),
        ("fidelite",["fidelite_min","fidelity","fidelite"]),("iterations_s",["iterations_s","it_s","iter_per_s"]),
        ("ram",["ram","ram_gb"]),("vram",["vram","vram_gb"]),("sm",["sm","sms"]),("warps",["warps"]))
TEXTES=("backend","job_id","checksum","signature","sig")
IGNORE={"date","timestamp","ts","mtime","started","ended","created","seed","version","pid"}
def metriques(d):
    """extrait les metriques presentes ; rien n'est converti d'une unite physique a une autre (QPU jamais en GFLOPS).
    Les champs numeriques non reconnus sont gardes tels quels dans 'autres' (nom brut), pour ne rien cacher."""
    f=aplatir(d);m={};pris=set()
    def p(cles):
        for k in cles:
            if k in f and num(f[k]):pris.add(k);return f[k]
        return None
    us,ms,s=p(T_US),p(T_MS),p(T_S)
    if us is not None:m["us"]=us;m["ms"]=us/1e3
    elif ms is not None:m["ms"]=ms;m["us"]=ms*1e3
    elif s is not None and s<1e5:m["us"]=s*1e6;m["ms"]=s*1e3
    for cle,noms in CHAMPS:
        v=p(noms)
        if v is not None:m[cle]=v
    if num(f.get("debit")) and isinstance(f.get("unite"),str):
        pris|={"debit","unite"}
        if "GFLOPS" in f["unite"] and "gflops" not in m:m["gflops"]=f["debit"]
        else:m["debit"]=f["debit"];m["unite"]=f["unite"]
    j=p(["jobs","job_count","n_jobs"])
    if j is not None:m["jobs"]=j
    elif isinstance(f.get("job_ids"),list):m["jobs"]=len(f["job_ids"]);pris.add("job_ids")
    for cle in TEXTES:
        if isinstance(f.get(cle),str):m[cle]=f[cle];pris.add(cle)
    autres={}
    for k,v in d.items() if isinstance(d,dict) else []:
        k2=str(k).lower()
        if k2 in pris or k2 in IGNORE or isinstance(v,bool):continue
        if num(v):autres[str(k)]=v
        elif isinstance(v,dict):
            for kk,vv in v.items():
                if num(vv) and not isinstance(vv,bool) and str(kk).lower() not in pris|IGNORE:autres[str(k)+"."+str(kk)]=vv
        if len(autres)>=10:break
    if autres:m["autres"]=autres
    return m
def composant(nom,d,source,regle=None):
    cl=d.get("classification")
    c={"id":(source+":"+nom).lower(),"name":nom,"type":type_de(nom+" "+source),"source":source,
       "classification":cl if isinstance(cl,str) and cl else "UNKNOWN","classification_source":"json" if isinstance(cl,str) and cl else None,
       "status":str(premier(d,["status","validation","etat"]) or "N/A"),"metrics":metriques(d)}
    if c["classification"]=="UNKNOWN" and regle:c["classification"],c["classification_source"]=regle
    return c
def depuis_json(nom_fichier,d):
    out=[]
    if isinstance(d,dict) and isinstance(d.get("results"),list):           # format results[] (benchmark monde)
        for i,it in enumerate(d["results"]):
            if isinstance(it,dict):out.append(composant(str(it.get("label") or it.get("name") or "resultat %d"%i),it,nom_fichier))
    elif isinstance(d,dict) and isinstance(d.get("backend"),str) and d.get("job_id"):   # tir QPU grave
        regle=("PHYSICAL_QPU","regle: backend %s + job_id IBM"%d["backend"]) if d["backend"].startswith("ibm_") else None
        out.append(composant("QPU "+d["backend"],d,nom_fichier,regle))
    elif isinstance(d,dict):                                               # format cle -> bloc de mesures (bench final)
        for k,v in d.items():
            if not isinstance(v,dict) or str(k).startswith("_"):continue
            fils=[(kk,vv) for kk,vv in v.items() if isinstance(vv,dict) and metriques(vv)]
            direct=any(num(x) for x in v.values())
            if fils and (len(fils)>=2 or not direct):out+=[composant(str(kk),vv,nom_fichier) for kk,vv in fils]   # groupe (ex. kernels -> VECTOR_ADD, MATMUL)
            elif metriques(v):out.append(composant(str(k),v,nom_fichier))
    return out
def inventaire():
    comps,results,unknown,mods=[],[],[],[]
    for p in sorted(ROOT.rglob("*.json")):
        sp=str(p)
        if any(x in sp for x in SKIP) or p.stat().st_size>5_000_000:continue
        try:d=json.loads(p.read_text(errors="ignore"))
        except Exception:continue
        cs=depuis_json(p.name,d);comps+=cs
        results.append({"file":str(p.relative_to(ROOT)),"mtime":p.stat().st_mtime,"components":len(cs),
                        "validation":isinstance(d,dict) and any(k in d for k in ("validation","status"))})
    for p in sorted(ROOT.glob("*.py")):
        if p.name.startswith("delta_") or p.parent==ROOT:
            t=type_de(p.stem);m={"name":p.stem,"file":p.name,"type":t}
            (mods if t!="OTHER" else unknown).append(m)
    return {"components":comps,"results":sorted(results,key=lambda r:-r["mtime"]),"modules":mods,"unknown":unknown,"scanned":time.time()}
if __name__=="__main__":print(json.dumps(inventaire(),indent=1)[:4000])
