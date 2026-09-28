import numpy as np, time, json, delta17 as D17
taille=4096
def source_io(a, latence=3e-4):
    """Mémoire lente de type ENTRÉE-SORTIE : elle ATTEND (disque, réseau),
    elle ne consomme pas le processeur. Latence émulée par une attente réelle."""
    time.sleep(latence)
    x=np.arange(a*taille,(a+1)*taille,dtype=np.float64)
    return np.sin(x*1e-7)
flux=list(range(600)); cap=8*1024*1024
print('=== MÉMOIRE LENTE D ENTRÉE-SORTIE (latence 300 µs, cas disque/réseau) ===')
res=[]
for prof in [0,2,4,8,16]:
    if prof==0:
        C=D17.CacheLRU(source_io,cap); nom='LRU sans prédiction'
    else:
        C=D17.DeltaCache(source_io,cap,prefetch=True,profondeur=prof); nom=f'Δ-CACHE profondeur {prof}'
    t0=time.time(); s=0.0
    for a in flux: s+=float(C.lire(a)[0])
    dt=time.time()-t0
    if hasattr(C,'fermer'): C.fermer()
    res.append(dict(nom=nom,t=dt,taux=C.taux(),lat=C.latence_moyenne_us()))
    print(f'{nom:26s} : {dt:5.2f} s | {C.latence_moyenne_us():7.2f} µs/accès | {100*C.taux():5.1f} % de succès',flush=True)
best=min(r['t'] for r in res[1:]); base=res[0]['t']
print(f'\ngain du préchargement sur une source lente : x{base/best:.2f} en temps total, '
      f'latence {res[0]["lat"]:.1f} -> {min(r["lat"] for r in res[1:]):.1f} µs')
json.dump(res,open('resultats_delta17c.json','w'),indent=1,default=float)
