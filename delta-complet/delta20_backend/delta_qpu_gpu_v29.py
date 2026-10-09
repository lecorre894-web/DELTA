#!/usr/bin/env python3
"""DELTA QPU EMULE V29 — noyau v26 EXACT + renumerotation dynamique des qubits (RTX 4080).
Lecons v28 : copie seule = ~50 % du temps a ~390 Go/s (acces disperses) ; L=4 coalesce mieux (~440 Go/s) mais les 4 qubits bas
FIXES occupent S pour rien -> plus de passages. NR=5 / B=13 / noyau generique = plus lents -> on revient au noyau v26 tel quel.
v29 : les L bits PHYSIQUES bas restent toujours charges (lignes memoire pleines), mais leur CONTENU change : a la fin de chaque
passage, le noyau reecrit le bloc avec une autre table d'adresses (OFF2) qui place dans les bits bas les qubits logiques dont
le passage suivant aura besoin. Cout : zero acces memoire en plus (meme lecture, meme ecriture, seules les adresses changent).
La permutation finale est suivie cote hote (relabel) : le vecteur reste exact, a lire avec la table phys.
Mesure equitable : v26 d'origine rechronometree dans la MEME execution (meme temperature GPU).
Validation : emulation NumPy vs Qiskit (n=12) ; GPU v29 FP32 vs GPU v21 FP64 (n=16, 20, 24), apres remise en ordre logique."""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_arena_v20 import circuit, qiskit_sv
from delta_qpu_gpu_v21 import compile_ops, fid, GPU
from delta_qpu_gpu_v25 import schedule_dag, fuse, gq
from delta_qpu_gpu_v26 import SRC as SRC26, group, RW
if GPU:
    import cupy as cp
    from delta_qpu_gpu_v21 import GPUSim
    from delta_qpu_gpu_v26 import RegSim as RegSim26
# noyau v26 a l'identique, sauf l'ecriture finale : table OFF2 (nouvelle disposition des bits)
SRC=SRC26.replace("const int *GR, const float *G){","const int *GR, const float *G, const long long *OFF2){",1)
SRC=SRC.replace("for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF[j]]=sh[j];","for(int j=threadIdx.x;j<M;j+=blockDim.x) psi[boff|OFF2[j]]=sh[j];",1)
assert SRC!=SRC26 and SRC.count("OFF2")==2
def schedule_rl(n,prog,B,L=4,fz=True):
    """ordonnanceur DAG v25 + renumerotation : passes=[(Sl logiques tries par bit physique, portes locales, phys_in, phys_out)]"""
    B=min(B,n); Q=[[] for _ in range(n)]
    for i,g in enumerate(prog):
        for q in gq(g): Q[q].append(i)
    ptr=[0]*n; left=len(prog); passes=[]; phys=list(range(n)); inv=list(range(n))
    def ready(i): return all(Q[q][ptr[q]]==i for q in gq(prog[i]))
    def nxt(q): return Q[q][ptr[q]] if ptr[q]<len(Q[q]) else 1<<30
    while left:
        S={inv[p] for p in range(L)}; gs=[]
        while True:
            moved=True
            while moved:
                moved=False
                for q in sorted(S):
                    while ptr[q]<len(Q[q]):
                        i=Q[q][ptr[q]]
                        if set(gq(prog[i]))<=S and ready(i):
                            gs.append(prog[i]); left-=1; moved=True
                            for r in gq(prog[i]): ptr[r]+=1
                        else: break
            if len(S)>=B: break
            best=None
            for q in range(n):
                if ptr[q]<len(Q[q]):
                    i=Q[q][ptr[q]]
                    if ready(i):
                        new=set(gq(prog[i]))-S
                        if new and len(S)+len(new)<=B and (best is None or (len(new),i)<best[0]): best=((len(new),i),new)
            if best is None: break
            S|=best[1]
        if not gs: raise RuntimeError("blocage ordonnanceur")
        for q in sorted((q for q in range(n) if q not in S),key=lambda q:(nxt(q),q))[:B-len(S)]: S.add(q)
        Sl=sorted(S,key=lambda q:phys[q]); P=[phys[q] for q in Sl]; pos={q:k for k,q in enumerate(Sl)}; loc=[]
        for g in (fuse(gs) if fz else gs):
            if g[0]=="u": loc.append(("u",pos[g[1]],g[2]))
            elif g[0]=="cx": loc.append(("cx",pos[g[1]],pos[g[2]]))
            else: loc.append(("u2",pos[g[1]],pos[g[2]],g[3]))
        F=sorted(sorted(Sl,key=lambda q:(nxt(q),phys[q]))[:L],key=lambda q:phys[q])
        order=F+[q for q in Sl if q not in F]; newp={q:P[k] for k,q in enumerate(order)}
        passes.append((Sl,loc,P,[newp[q] for q in Sl]))
        for q in Sl: phys[q]=newp[q]; inv[newp[q]]=q
    return passes,phys
def to_logical(psi,n,phys):
    """remet un vecteur en disposition physique dans l'ordre logique (qubit q au bit q)"""
    a=np.asarray(psi).reshape([2]*n)
    return np.transpose(a,[n-1-phys[n-1-ax] for ax in range(n)]).reshape(-1)
def offs(bits):
    j=np.arange(1<<len(bits),dtype=np.int64); o=np.zeros_like(j)
    for k,p in enumerate(bits): o|=((j>>k)&1)<<p
    return o
def run_numpy(n,passes):
    """emulation NumPy de v29 : chargement par bits physiques P, portes locales, reecriture par bits physiques P2"""
    N=1<<n; psi=np.zeros(N,np.complex128); psi[0]=1
    for Sl,loc,P,P2 in passes:
        B=len(P); C=[p for p in range(n) if p not in P]; bases=np.arange(1<<(n-B),dtype=np.int64); boff=np.zeros_like(bases)
        for k,p in enumerate(C): boff|=((bases>>k)&1)<<p
        sh=psi[boff[:,None]|offs(P)[None,:]]
        for g in loc:
            if g[0]=="u":
                a,M=g[1],g[2]; i=np.arange((1<<B)//2); i0=((i>>a)<<(a+1))|(i&((1<<a)-1)); i1=i0|(1<<a)
                x,y=sh[:,i0].copy(),sh[:,i1].copy(); sh[:,i0]=M[0,0]*x+M[0,1]*y; sh[:,i1]=M[1,0]*x+M[1,1]*y
            else:
                a,b=g[1],g[2]; lo,hi=min(a,b),max(a,b); i=np.arange((1<<B)//4)
                x=((i>>lo)<<(lo+1))|(i&((1<<lo)-1)); x=((x>>hi)<<(hi+1))|(x&((1<<hi)-1))
                if g[0]=="cx": i0=x|(1<<a); i1=i0|(1<<b); t=sh[:,i0].copy(); sh[:,i0]=sh[:,i1]; sh[:,i1]=t
                else:
                    ids=[x,x|(1<<a),x|(1<<b),x|(1<<a)|(1<<b)]; v=np.stack([sh[:,k] for k in ids],0); r=np.tensordot(g[3],v,1)
                    for k in range(4): sh[:,ids[k]]=r[k]
        out=np.empty_like(psi); out[boff[:,None]|offs(P2)[None,:]]=sh; psi=out
    return psi
class RLSim:
    def __init__(s):
        s.k=cp.RawModule(code=SRC,options=("-std=c++14",)).get_function("blk")
        try: s.k.max_dynamic_shared_size_bytes=65536
        except Exception as e: print("note: opt-in 64 Ko indisponible (%s)"%e)
    def upload(s,n,passes):
        out=[]
        for Sl,loc,P,P2 in passes:
            B=len(P); C=[p for p in range(n) if p not in P]; grs=group(loc,B); rows=[]; GRi=[]
            for R,gs in grs:
                pos={q:j for j,q in enumerate(R)}; GRi.append(R+sorted(R)+[len(rows),len(gs)])
                for g in gs:
                    r=np.zeros(RW)
                    if g[0]=="u": M=g[2]; r[:11]=[0,pos[g[1]],0,M[0,0].real,M[0,0].imag,M[0,1].real,M[0,1].imag,M[1,0].real,M[1,0].imag,M[1,1].real,M[1,1].imag]
                    elif g[0]=="cx": r[:3]=[1,pos[g[1]],pos[g[2]]]
                    else: r[:3]=[2,pos[g[1]],pos[g[2]]]; r[3:]=np.stack([g[3].real,g[3].imag],-1).reshape(-1)
                    rows.append(r)
            out.append((B,cp.asarray(offs(P)),cp.asarray(np.array(C or [0],np.int32)),len(grs),cp.asarray(np.array(GRi,np.int32)),cp.asarray(np.array(rows,np.float32)),cp.asarray(offs(P2))))
        return out
    def run(s,n,dev,copy=False):
        psi=cp.zeros(1<<n,dtype=cp.complex64); psi[0]=1
        for B,OFF,C,ngr,GR,G,OFF2 in dev:
            s.k((1<<(n-B),),(1<<(B-4),),(psi,np.int32(n),np.int32(B),OFF,C,np.int32(0 if copy else ngr),GR,G,OFF2),shared_mem=(1<<B)*8)
        return psi
def tmin(f,it=3):
    ms=1e30; r=None
    for _ in range(it):
        r=None; cp.get_default_memory_pool().free_all_blocks()
        e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); r=f(); e1.record(); e1.synchronize(); ms=min(ms,cp.cuda.get_elapsed_time(e0,e1))
    return ms,r
def norm1(psi):
    x=psi.view(cp.float32); K=1<<26; return abs(math.sqrt(sum(float(x[i:i+K].dot(x[i:i+K])) for i in range(0,x.size,K)))-1)
def main():
    L=[]; P=lambda s:(print(s,flush=True),L.append(s))
    P("="*104); P("DELTA QPU EMULE V29 | noyau v26 exact + renumerotation dynamique des qubits (bits bas toujours pleins, contenu choisi) | RTX 4080"); P("="*104)
    ops=circuit(12,12,5); prog=compile_ops(12,ops); ref=qiskit_sv(12,ops)
    for B,Lb in ((8,3),(8,4),(10,4),(10,5)):
        ps,ph=schedule_rl(12,prog,B,Lb); f=fid(ref,to_logical(run_numpy(12,ps),12,ph))
        P("emulation NumPy v29 n=12 B=%d L=%d : %d passages (dag v26 %d)  fidelite vs Qiskit=%.12f %s"%(B,Lb,len(ps),len(schedule_dag(12,prog,B,Lb,True)),f,"PASS" if f>1-1e-9 else "FAIL"))
    P("-"*104)
    for n in (24,28,30):
        pr=compile_ops(n,circuit(n,20,n))
        P("passages n=%d : "%n+"  ".join("L=%d dag %2d / v29 %2d"%(Lb,len(schedule_dag(n,pr,12,Lb,True)),len(schedule_rl(n,pr,12,Lb)[0])) for Lb in (2,3,4,5)))
    if not GPU: P("CuPy absent : arret apres validation logique."); return
    rs=RLSim(); r26=RegSim26(); v21=GPUSim()
    for n in (16,20,24):
        pr=compile_ops(n,circuit(n,20,n)); a=cp.asnumpy(v21.run(n,pr)).astype(np.complex128)
        for Lb in (2,4):
            ps,ph=schedule_rl(n,pr,12,Lb); b=to_logical(cp.asnumpy(rs.run(n,rs.upload(n,ps))).astype(np.complex128),n,ph); f=fid(a,b)
            P("validation GPU v29 FP32 L=%d vs GPU v21 FP64 n=%d : infidelite=%.2e  %s"%(Lb,n,1-f,"PASS" if abs(1-f)<1e-4 else "FAIL"))
        del a,b; cp.get_default_memory_pool().free_all_blocks()
    free=cp.cuda.runtime.memGetInfo()[0]; best={}; ref26={}
    for n in (24,28,29,30):
        if (1<<n)*8>free*0.85: P("n=%-2d : VRAM insuffisante, saute"%n); continue
        pr=compile_ops(n,circuit(n,20,n))
        try:
            ps=schedule_dag(n,pr,12,2,True); dev=r26.upload(n,ps); r26.run(20,r26.upload(20,schedule_dag(20,compile_ops(20,circuit(20,1,1)),12,2,True)))
            ms,psi=tmin(lambda:r26.run(n,dev)); nm=norm1(psi); del psi; ref26[n]=ms
            P("n=%-2d v26 d'origine B=12 L=2     : %2d passages : %8.1f ms  |norme-1|=%.1e   (reference rechronometree)"%(n,len(ps),ms,nm))
        except Exception as e: P("n=%-2d v26 : ECHEC %s"%(n,str(e)[:90]))
        for B,Lb in ((12,2),(12,3),(12,4),(12,5),(13,4),(13,5)):
            try:
                ps,ph=schedule_rl(n,pr,B,Lb); dev=rs.upload(n,ps); rs.run(20,rs.upload(20,schedule_rl(20,compile_ops(20,circuit(20,1,1)),B,Lb)[0])); cp.cuda.Device().synchronize()
                ms,psi=tmin(lambda:rs.run(n,dev)); nm=norm1(psi); del psi
                mc,psi=tmin(lambda:rs.run(n,dev,True),2); del psi; cp.get_default_memory_pool().free_all_blocks()
                gbs=len(ps)*(1<<n)*16/1e9; tag="B=%d L=%d"%(B,Lb)
                if ms<best.get(n,(1e30,""))[0]: best[n]=(ms,tag)
                P("n=%-2d v29 %-10s            : %2d passages : %8.1f ms (copie seule %7.1f ms, %4.0f Go/s)  |norme-1|=%.1e  %s"%(n,tag,len(ps),ms,mc,gbs/(mc/1e3),nm,("vs v26 x%.2f"%(ref26[n]/ms)) if n in ref26 else ""))
            except Exception as e:
                P("n=%-2d v29 B=%d L=%d : ECHEC %s"%(n,B,Lb,str(e)[:90])); cp.get_default_memory_pool().free_all_blocks()
    P("-"*104); P("MEILLEUR V29 : "+"  ".join("%dq %.1f ms [%s]%s"%(n,best[n][0],best[n][1],(" x%.2f v26"%(ref26[n]/best[n][0])) if n in ref26 else "") for n in sorted(best)))
    P("NOTE : le vecteur final est en disposition permutee (table phys suivie par l'hote) ; remise en ordre = 1 passage de plus si on veut le vecteur trie.")
    P("="*104)
    f=os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_gpu_v29_results.txt"); open(f,"w").write("\n".join(L)+"\n"); print("JOURNAL="+f)
if __name__=="__main__": main()
