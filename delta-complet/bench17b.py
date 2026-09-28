import numpy as np, time, json, delta17 as D17
taille=4096
def source(a):
    x=np.arange(a*taille,(a+1)*taille,dtype=np.float64)
    return np.sin(x*1e-6)+np.cos(x*3e-7)
def travail(d,intensite):
    s=0.0
    for _ in range(intensite): s+=float(np.sqrt(np.abs(d)).sum())
    return s
flux=list(range(1200))
cap=8*1024*1024
print('=== LE PRÉCHARGEMENT PAIE QUAND LE CONSOMMATEUR TRAVAILLE ENTRE DEUX ACCÈS ===')
print('intensité = nombre de passes de calcul par ligne lue\n')
res=[]
for intensite in [0,1,3,8]:
    L=D17.CacheLRU(source,cap); t0=time.time(); s1=0.0
    for a in flux: s1+=travail(L.lire(a),intensite)
    tl=time.time()-t0
    C=D17.DeltaCache(source,cap,prefetch=True,profondeur=8); t0=time.time(); s2=0.0
    for a in flux: s2+=travail(C.lire(a),intensite)
    tc=time.time()-t0; C.fermer()
    res.append(dict(intensite=intensite,t_lru=tl,t_delta=tc,taux=C.taux(),
                    lat_lru=L.latence_moyenne_us(),lat_delta=C.latence_moyenne_us()))
    print(f'intensité {intensite} | LRU {tl:5.2f} s ({L.latence_moyenne_us():6.2f} µs/accès, {100*L.taux():4.1f} % succès) '
          f'| Δ-CACHE {tc:5.2f} s ({C.latence_moyenne_us():6.2f} µs/accès, {100*C.taux():4.1f} % succès) '
          f'-> x{tl/max(tc,1e-9):4.2f} | écart résultats {abs(s1-s2):.1e}',flush=True)
json.dump(res,open('resultats_delta17b.json','w'),indent=1,default=float)
