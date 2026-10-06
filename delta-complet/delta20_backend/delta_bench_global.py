#!/usr/bin/env python3
"""DELTA X512 — BENCH GLOBAL : relance chaque version dans sa meilleure config et derive toutes les unites.
Contrat commun : NW=65536 mots x 3 tableaux (A,B,C) = 12 582 912 bits logiques par passe, resultat = popcount(C ^ (A & B))."""
import os,re,subprocess,sys,time,datetime
B=os.path.expanduser("~/DELTA/delta-complet/delta20_backend"); W="/mnt/c/delta_v7"
BITS=3*65536*64; NUMPY_REF=1864.0; FAST_REF=45.579
def run(cmd,cwd,pat):
    try: out=subprocess.run(cmd,cwd=cwd,shell=True,capture_output=True,text=True,timeout=300).stdout
    except Exception as e: return None,str(e)
    m=re.search(pat,out); return (float(m.group(1)) if m else None),out
def numpy_pipeline():
    X=os.path.join(B,"x512_runtime"); py=os.path.join(B,".venv/bin/python")
    code=("import time\nfrom delta_x512 import vecs,aligne,X512Core\nc=X512Core();N=50;t=time.perf_counter()\n"
          "for s in range(N):\n V=[aligne(v) for v in vecs(s,65536)];c.tpop(*V,prog='micro')\nprint('PIPE_US=%.3f'%((time.perf_counter()-t)/N*1e6))")
    return run(f"X512_DIR={X} {py} -c \"{code}\"",B,r"PIPE_US=([\d.]+)")
R=[]
def add(nom,contrat,coeurs,res):
    us=res[0]; R.append((nom,contrat,coeurs,us)); print(f"  {nom:<34} {'%.3f us'%us if us else 'ECHEC'}",flush=True)
print("Mesures en cours (~2 min)...")
add("Pipeline NumPy DELTA (gen+tpop)","complet","1 (Python)",numpy_pipeline())
add("v3 noyau natif (WSL)","complet","1",run("taskset -c 2 ./bench_x512_internal_v3",B,r"V3_MEAN_US=([\d.]+)"))
add("v7 Windows natif","complet","8 CCD1",run("./bench_x512_mt_v7.exe 8 1",W,r"BEST_US=([\d.]+)"))
add("v8 drapeaux par thread","complet","8 CCD1",run("./bench_x512_mt_v8.exe 8 1 0",W,r"BEST_US=([\d.]+)"))
add("v9 masque AB precalcule","A,B fixes","8 CCD1",run("./bench_x512_mt_v9.exe 8 1",W,r"BEST_US=([\d.]+)"))
add("v11 BIAS maitre 15%","A,B fixes","8 CCD1",run("./bench_x512_mt_v11.exe 8 1 15",W,r"BEST_US=([\d.]+)"))
add("v13 fusion gen+tpop","gen+calcul","1",run("taskset -c 2 ./bench_x512_fused_v13",B,r"FUSED_US=([\d.]+)"))
add("v14 fusion multi-coeur","gen+calcul","8 CCD1",run("./bench_x512_mt_v14.exe 8 1",W,r"BEST_US=([\d.]+)"))
add("v15 fusion 16 flux","gen+calcul","16",run("./bench_x512_mt_v15.exe 16 1",W,r"BEST_US=([\d.]+)"))
add("v16 fusion 32 flux","gen+calcul","16",run("./bench_x512_mt_v16.exe 16 1",W,r"BEST_US=([\d.]+)"))
inc=run("taskset -c 2 ./bench_x512_incr_v12",B,r"M=64\s+mots/appel\s+VALIDATION=\w+\s+BEST_US=([\d.]+)")
L=[]; L.append("="*118)
L.append(f"DELTA X512 BENCH GLOBAL {datetime.datetime.now():%Y-%m-%d %H:%M:%S} | Ryzen 9 9950X3D | NW=65536 x 3 = {BITS:,} bits/passe".replace(","," "))
L.append("="*118)
L.append(f"{'VERSION':<34}{'CONTRAT':<12}{'COEURS':<11}{'us':>10}{'ns':>12}{'ms':>11}{'passes/s':>13}{'Gbit/s':>9}{'Gmots/s':>9}{'x NumPy':>9}{'x 45.579':>9}")
for nom,c,co,us in R:
    if not us: L.append(f"{nom:<34}{c:<12}{co:<11}{'ECHEC':>10}"); continue
    L.append(f"{nom:<34}{c:<12}{co:<11}{us:>10.3f}{us*1e3:>12.1f}{us/1e3:>11.6f}{1e6/us:>13,.0f}{BITS/(us*1e3):>9.1f}{3*65536/(us*1e3):>9.2f}{NUMPY_REF/us:>9.0f}{FAST_REF/us:>9.1f}".replace(","," "))
if inc[0]:
    us=inc[0]; L.append("-"*118)
    L.append(f"{'v12 incremental (64 mots modifies)':<34}{'64 mots':<12}{'1':<11}{us:>10.3f}{us*1e3:>12.1f}{us/1e3:>11.6f}{1e6/us:>13,.0f}{'  (contrat different : 64 mots/appel, pas 65536)':<40}".replace(","," "))
L.append("-"*118)
L.append("Gbit/s = bits logiques traites par passe / temps ; pour la fusion, les bits sont generes en registres puis consommes (aucune lecture memoire).")
L.append("x NumPy = vs pipeline DELTA reference 1864 us (generation NumPy + tpop) ; x 45.579 = vs FASTPATH_PREVIOUS du noyau seul.")
txt="\n".join(L); print("\n"+txt)
f=os.path.join(B,"x512_bench_global_final_%s.txt"%datetime.date.today()); open(f,"w").write(txt+"\n"); print("\nJOURNAL="+f)
