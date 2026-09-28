import time, math, json, numpy as np, duel as DU
R=[]
def run(nom,n,C,chi):
    t0=time.time(); s=DU.delta(n,C,chi); dt=time.time()-t0
    G=DU.nb_portes2(C); fq=(1-0.005)**G
    r=dict(epreuve=nom,n=n,chi=chi,portes2=G,delta_fid_est=s.fid,t=dt,mem=s.memory(),qpu_fid_attendue=fq,qpu_t_phys=len(C)*60e-9)
    R.append(r)
    print(f'{nom:34s} n={n} χ={chi:3d} portes2={G:6d} | Δ fid estimée={s.fid:.6f} en {dt:6.1f} s, {s.memory()/1024:7.0f} Kio | QPU fid attendue={fq:.2e}',flush=True)
run('1 intrication faible',1000,DU.circuit_faible(1000,10,5,0.6),4)
run('2 GHZ',1000,DU.circuit_ghz(1000),2)
run('3 monstre profondeur 8',1000,DU.circuit_monstre(1000,8,7),64)
run('3 monstre profondeur 24',1000,DU.circuit_monstre(1000,24,7),64)
json.dump(R,open('resultats_duel_grand.json','w'),indent=1,default=float)
