import numpy as np, math, time, sys, json
import delta2 as D2, delta3 as D3
def human(b):
    for u in ['o','Kio','Mio','Gio','Tio']:
        if b<1024: return f'{b:.1f} {u}'
        b/=1024
n16=16; L=D2.circuit_faible(16,40,5,0.6); ref=D2.dense_reference(16,L)
for label,ch in [('uniforme χ=3',np.r_[1,[3]*15,1]),('proportionné χ=4/3',D3.budget_mu(16,[3,12],4,3))]:
    s=D2.run2(D3.Delta3(16,ch),L); v=s.dense(); v/=np.linalg.norm(v)
    print(f'n=16 {label:20s} fid VRAIE={abs(np.vdot(ref,v))**2:.5f} mem={human(s.memory_compacte())}')
res=[]
for label,mk in [('uniforme χ=4',lambda n:np.r_[1,[4]*(n-1),1]),('uniforme χ=3',lambda n:np.r_[1,[3]*(n-1),1]),
                 ('proportionné χ=4/3',lambda n:D3.budget_mu(n,np.arange(500,n,1000),4,3))]:
    n=int(sys.argv[1]) if len(sys.argv)>1 else 1000000
    L=D2.circuit_faible(n,10,5,0.6)
    t0=time.time(); s=D2.run2(D3.Delta3(n,mk(n)),L); dt=time.time()-t0
    mc=s.memory_compacte()
    res.append(dict(mode=label,n=n,t=dt,mem=mc,o_par_qubit=mc/n,fid=math.exp(s.logfid)))
    print(f'n={n} {label:20s} t={dt:6.1f}s mem compacte={human(mc):>9s} ({mc/n:6.1f} o/qubit) fid globale={math.exp(s.logfid):.5f}',flush=True)
json.dump(res,open('resultats_delta3b.json','w'),indent=1)
