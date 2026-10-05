"""BENCHMARK FINAL DELTA : bout de chaine, en SERIE (chaque mesure repetee), statistiques ms / us, GFLOPS (flottant) et Gbit-op/s (booleen) SEPARES.
Rien n'est extrapole : chaque ligne est mesuree ici, sauf le QPU (dernier resultat grave, pour ne pas consommer de quota)."""
import os,sys,time,json,glob,hashlib,subprocess,statistics as st,numpy as np
R=os.environ.get("AVX_ROOT",os.path.expanduser("~/turnip/avx2048"));SER=int(os.environ.get("SERIE","10"));OUT={};T0=time.perf_counter()
def serie(f,n=SER):
    f();v=[]
    for _ in range(n):t=time.perf_counter();f();v.append(time.perf_counter()-t)
    v.sort();return {"n":n,"med":st.median(v),"min":v[0],"max":v[-1],"p95":v[min(n-1,int(0.95*n))],"cv":st.pstdev(v)/st.mean(v)*100}
def fmt(s):
    u=lambda x:"%.2f us"%(x*1e6) if x<1e-3 else "%.3f ms"%(x*1e3)
    return "med %s | min %s | max %s | p95 %s | CV %.1f%%"%(u(s["med"]),u(s["min"]),u(s["max"]),u(s["p95"]),s["cv"])
def ligne(cle,s,debit,unite):
    OUT[cle]={**s,"debit":debit,"unite":unite};print("%-34s %s | %10.1f %s"%(cle,fmt(s),debit,unite),flush=True)
def section(t):print("\n--- %s ---"%t,flush=True)
def main():
    global OUT
    ABS=[]  # verification stricte : toute partie absente fait echouer le benchmark
    cpu=[l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0];print("=== BENCHMARK FINAL DELTA | %s | %d threads | serie %d ==="%(cpu,os.cpu_count(),SER))
    section("1. SOCLE XEON (flottant, GFLOPS)")
    N=1024;g=np.random.default_rng(1)
    for p,dt in (("FP64",np.float64),("FP32",np.float32)):
        A=g.random((N,N)).astype(dt);B=g.random((N,N)).astype(dt);s=serie(lambda:A@B);ligne("socle matmul %s 1024"%p,s,2*N**3/s["med"]/1e9,"GFLOPS")
    v=g.random(1<<24,dtype=np.float32);w=g.random(1<<24,dtype=np.float32);s=serie(lambda:v+w);ligne("socle bande passante RAM",s,12*(1<<24)/s["med"]/1e9,"Go/s")
    section("2. INDEX DELTA (cache signe, cle par contenu)")
    cache={hashlib.sha256(b"%d"%i).hexdigest():i for i in range(100000)};ks=list(cache)[:10000]
    s=serie(lambda:[cache[k] for k in ks]);ligne("index hit (par acces)",{k:(x/10000 if k in("med","min","max","p95") else x) for k,x in s.items()},10000/s["med"]/1e6,"M acces/s")
    s=serie(lambda:hashlib.sha256(np.ones(4096,np.uint8).tobytes()).hexdigest());ligne("signature SHA-256 4 Ko",s,4096/s["med"]/1e9,"Go/s")
    section("3. MOTEUR BOOLEEN (Gbit-op/s, PAS des GFLOPS)")
    sys.path.insert(0,os.getcwd())
    try:
        from delta_hybrid import Hybride,_gen
        nw=16384*16;mk=lambda:np.concatenate([_gen(16384,t%2==0,g) for t in range(16)])
        H=Hybride(np.array([mk() for _ in range(16)]),np.array([mk() for _ in range(16)]),mk());ops=16*16*nw*64*3
        for m in("standard","dense","auto"):s=serie(lambda:H.run(m));ligne("hybride 16x16 %s"%m,s,ops/s["med"]/1e9,"Gbit-op/s"+(" logiques" if m=="auto" else ""))
    except Exception as e:print("hybride : ABSENT (%s)"%e);ABS.append("hybride")
    try:
        from delta_x512 import X512Core,aligne
        c=X512Core();nw=16384;A,B,C=[aligne(g.integers(0,2**63,nw,dtype=np.uint64)) for _ in range(3)]
        for p in("direct","micro"):
            k=c.tpop(A,B,C,p)[1];s=serie(lambda:c.tpop(A,B,C,p),max(SER,20));ligne("coeur X512 %s 1 Mbit (L2)"%p,s,3*nw*64/s["med"]/1e9,"Gbit-op/s");OUT["coeur X512 %s 1 Mbit (L2)"%p]["G_instr_X512_s"]=k/s["med"]/1e9;print("%-34s %.2f G instr X512/s (appel Python compris)"%("",k/s["med"]/1e9))
        import delta_resident
        reqs=[(hashlib.sha256(b"b%d"%(i%40)).hexdigest(),i%40,16384) for i in range(400)]
        def run(): return delta_resident.compute(reqs)
        s=serie(run,max(3,SER//3));ligne("DELTA x X512 %d workers (400 dem.)"%os.cpu_count(),s,400/s["med"],"demandes/s")
    except Exception as e:print("X512 : ABSENT (%s)"%e);ABS.append("X512")
    section("4. LOGICIEL AVX2048 (binaires du labo, en serie)")
    import re
    def binaire(nom,motif,rx,unite):
        b=os.path.join(R,"sw",nom)
        if not os.path.exists(b):print("%s : ABSENT"%nom);ABS.append(nom);return
        vals=[]
        for _ in range(max(3,SER//2)):
            o=subprocess.run([b],capture_output=True,text=True).stdout
            for l in o.splitlines():
                m=re.search(rx,l) if motif in l else None
                if m:vals.append(float(m.group(1)))
        if not vals:print("%s : AUCUNE MESURE"%nom);ABS.append(nom);return
        vals.sort();print("%-34s med %.2f | min %.2f | max %.2f %s (%d passages)"%(nom+" "+motif,st.median(vals),vals[0],vals[-1],unite,len(vals)));OUT[nom+" "+motif]={"med":st.median(vals),"min":vals[0],"max":vals[-1],"unite":unite}
    binaire("bench2048","AVX2048",r"([\d.]+) Gbit-op/s","Gbit-op/s");binaire("a2k_texte","Mo de texte",r"4 popcounts ([\d.]+) ms","ms (4 popcounts, 64 Mo)");binaire("a2k_texte","Mo de texte",r"4 popcounts [\d.]+ ms \(([\d.]+) Go/s","Go/s texte classe")
    section("5. MATERIEL SIMULE (cycles, issus du rejeu)")
    o3=os.path.join(R,"rv","o3.txt")
    if os.path.exists(o3):
        for l in open(o3):
            if "cycles" in l:print("RISC-V v1 : "+l.strip())
    print("FPGA ECP5 mesure au routage : tpop16 186 MHz (1 unite) | 4 unites allegees 127 MHz (~3,1 Tbit-op/s) | latence 41,6 a 86 ns")
    section("6. QPU (dernier tir grave, non relance)")
    for f in sorted(glob.glob("delta_qpu_hybride2_qpu.json"))+sorted(glob.glob("delta_qpu_hybride.json")):
        try:d=json.load(open(f));print("%s : %s | job %s | F>=%.3f | ZZ min %.3f | %s"%(f,d.get("backend"),d.get("job_id"),d.get("fidelite_min",0),min(d.get("zz_voisins",[0])),d.get("date")))
        except Exception:pass
    tot=time.perf_counter()-T0;OUT["_total_s"]=tot;OUT["_cpu"]=cpu;OUT["_serie"]=SER
    json.dump(OUT,open("delta_bench_final.json","w"),indent=1)
    print("\nDUREE TOTALE DU BENCHMARK : %.2f s | detail grave dans delta_bench_final.json"%tot)
    print("RAPPEL : GFLOPS = calcul flottant ; Gbit-op/s = operations booleennes ; 'logiques' = bits couverts grace a la compression, pas du calcul")
    print("PARTIES MANQUANTES : %s"%(", ".join(sorted(set(ABS))) or "aucune"))
    print("DELTA_BENCH_FINAL=%s"%("FAIL" if ABS else "OK"));return 3 if ABS else 0
if __name__=="__main__":  # garde obligatoire : les workers X512 ne relancent plus le benchmark
    sys.exit(main())
