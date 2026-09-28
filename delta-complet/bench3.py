import numpy as np, math, time, json, sys
import delta2 as D2, delta3 as D3
def human(b):
    for u in ['o','Kio','Mio','Gio','Tio']:
        if b<1024: return f'{b:.1f} {u}'
        b/=1024
    return f'{b:.1f} Pio'
out={}
print('=== 1. VÉRITÉ TERRAIN n=16, θ=0.6, profondeur 40 (le cas dur) ===')
n=16; imp=[3,12]; L=D2.circuit_faible(n,40,5,0.6); ref=D2.dense_reference(n,L)
r=[]
for label,chib in [('uniforme χ=2',np.r_[1,[2]*(n-1),1]),('uniforme χ=4',np.r_[1,[4]*(n-1),1]),
                   ('uniforme χ=8',np.r_[1,[8]*(n-1),1]),('proportionné χ=8/2',D3.budget_mu(n,imp,8,2)),
                   ('proportionné χ=8/4',D3.budget_mu(n,imp,8,4))]:
    chib=np.minimum(chib, np.r_[[2**min(k,n-k) for k in range(n+1)]])   # χ ne peut dépasser le max physique
    s=D2.run2(D3.Delta3(n,chib),L); v=s.dense(); v=v/np.linalg.norm(v)
    F=abs(np.vdot(ref,v))**2
    fl=np.mean([D3.fid1(D3.reduit(ref,n,q),D3.reduit(v,n,q)) for q in imp])
    r.append(dict(mode=label,mem=s.memory_compacte(),fid_globale=F,fid_imp=fl,fid_est=s.fid))
    print(f'{label:22s} mem compacte={human(s.memory_compacte()):>9s}  fid globale VRAIE={F:.5f}  '
          f'fid qubits importants={fl:.6f}  (erreur {1-fl:.1e})  fid_est={s.fid:.5f}')
out['verite']=r
print('\n=== 2. PALIERS x10 : μ_j proportionné (χ=4 autour de 0,1 % de qubits importants, χ=2 ailleurs) ===')
r2=[]
for n in [int(x) for x in sys.argv[1:]] or [1000,10000,100000,1000000]:
    imp=np.arange(500,n,1000)
    chib=D3.budget_mu(n,imp,4,2)
    L=D2.circuit_faible(n,10,5,0.6)
    t0=time.time(); s=D2.run2(D3.Delta3(n,chib),L); dt=time.time()-t0
    mc=s.memory_compacte(); bq=mc/n
    r2.append(dict(n=n,t=dt,portes_s=s.gates/dt,mem_compacte=mc,octets_par_qubit=bq,logfid=s.logfid,lost_max=s.lost_max))
    print(f'n={n:9d} t={dt:8.1f}s {s.gates/dt:9.0f} portes/s  mem compacte={human(mc):>9s} ({bq:.1f} o/qubit)  '
          f'fid globale={math.exp(s.logfid):.5f}  perte max/lien={s.lost_max:.1e}', flush=True)
out['paliers']=r2
json.dump(out,open('resultats_delta3.json' if len(sys.argv)<2 else f'resultats_delta3_{sys.argv[1]}.json','w'),indent=1,default=float)
