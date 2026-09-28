import numpy as np, time, math, delta15 as D15
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:7.1f} {u}'
        x/=1024
b=128
print('=== CÔNE DANS LE LOT (corrigé) : on ne calcule que les plages demandées ===')
O=D15.Omni(D15.noyau_lisse,b=b,eps=1e-8,portee_max=40); four=D15.fournisseur(b)
Y,f_tout=O.lot(1000,512,four)
O2=D15.Omni(D15.noyau_lisse,b=b,eps=1e-8,portee_max=40)
Y2,_=O2.lot(1000,512,four,sorties=range(1000,1008))
print(f'512 blocs : {f_tout/1e6:8.2f} Mflop | 8 blocs demandés : {O2.flops/1e6:8.2f} Mflop '
      f'-> x{f_tout/max(O2.flops,1):.0f} de flops en moins')
print('   contrôle : écart sur les 8 blocs =',f'{np.abs(Y[:,:8]-Y2).max():.2e}')

print('\n=== % ET SYMÉTRIE : noyau à décroissance rapide (portée physique finie) ===')
for L,label in [(2000.0,'décroissance lente'),(300.0,'décroissance rapide'),(60.0,'très courte portée')]:
    noy=lambda d,L=L: np.exp(-np.abs(d)/L)
    O3=D15.Omni(noy,b=b,eps=1e-8,portee_max=200,seuil_float32=1e-3)
    n64=sum(1 for c in O3.dtypes if O3.dtypes[c]==np.float64)
    n32=len(O3.dtypes)-n64
    print(f'{label:20s} : {len(O3.U):3d} classes gardées, {O3.ignorees:3d} ignorées (vide), '
          f'float64 {n64:3d} / float32 {n32:3d} | mémoire {h(O3.octets())} | rangs {min(O3.rangs.values())}-{max(O3.rangs.values())}')
    t0=time.time(); Y,f=O3.lot(10**9,2048,four); dt=time.time()-t0
    print(f'{"":20s}   2048 blocs en {dt:5.2f} s -> {O3.gflops():5.2f} Gflops, {O3.inconnues_par_seconde():12,.0f} inconnues/s')
