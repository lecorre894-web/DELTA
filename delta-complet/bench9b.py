import numpy as np, time, delta8 as D8, delta9 as D9, duel as DU
from bench8 import circuit
print('=== 1. SVD randomisée avec k PRÉDICTIF (n=32, monstre profondeur 10) ===')
n=32; circ=circuit(n,10,11,'monstre')
for alea in [False,True]:
    s=D9.Delta9(n,X_base=512,eps=1e-6,alea=alea,graine=2)
    t0=time.time(); D8.executer(s,circ); dt=time.time()-t0
    print(f'alea={alea!s:5s} X_pic={s.X_pic:4d} fid estimée={s.fid:.8f} t={dt:6.2f} s  décompositions={s.stats}')
print('\n=== 2. REPRISE contre ARRÊT : plus petit budget qui finit (n=16, monstre prof. 12) ===')
n=16; circ=circuit(n,12,7,'monstre')
for pol in ['arret','reprise']:
    seuil=None
    for bud in [256,384,512,768,1024,1536,2048,2560,3072,4096]:
        s=D9.Delta9(n,X_base=8,eps=1e-6,budget_octets=bud*1024,politique=pol,alea=True)
        try:
            D8.executer(s,circ); seuil=(bud,s.fid,len(getattr(s,'reprises',[]))); break
        except D8.SaturationDelta: pass
    print(f'politique={pol:8s} : finit à partir de {seuil[0]} Kio (fid estimée {seuil[1]:.8f}, {seuil[2]} reprises)' if seuil else f'politique={pol}: jamais')
print('\n=== 3. ECC : surcoût de parité après découpe à coût égal ===')
s=D9.Delta9(16,X_base=256,eps=1e-6,alea=True); D8.executer(s,circ)
for N in [2,4,8]:
    noeuds,ecc=s.repartir(N)
    t=sum(len(v['bloc'])*16 for v in noeuds.values())
    print(f'{N} nœuds : données {t/1024:7.1f} Kio, parité {ecc.parite.nbytes/1024:7.1f} Kio -> surcoût {ecc.parite.nbytes/t*100:5.1f} % (idéal {100/N:.0f} %)')
