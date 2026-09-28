import numpy as np, time, json, delta17 as D17
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:7.1f} {u}'
        x/=1024
taille=4096   # 32 Kio par ligne
def source(a):
    """Mémoire lente : la ligne doit être calculée (cas réel : disque, réseau, recalcul)."""
    x=np.arange(a*taille,(a+1)*taille,dtype=np.float64)
    return np.sin(x*1e-6)+np.cos(x*3e-7)
def flux_sequentiel(n): return list(range(n))
def flux_pas(n,pas=3): return [ (k*pas)%(n) for k in range(n)]
def flux_boucle(n,motif=64,tours=8): return [ (k%motif) for k in range(n)] if tours else []
def flux_irregulier(n,seed=1):
    rng=np.random.default_rng(seed); base=rng.permutation(128).tolist()
    return [base[k%128]+(k//128)*0 for k in range(n)]      # motif irrégulier mais RÉPÉTÉ
def flux_aleatoire(n,seed=2):
    rng=np.random.default_rng(seed); return rng.integers(0,4096,size=n).tolist()

res=[]
cap=8*1024*1024
for nom,flux in [('séquentiel',flux_sequentiel(1500)),('pas de 3',flux_pas(1500)),
                 ('boucle courte (réutilisation)',flux_boucle(1500)),
                 ('irrégulier mais répété',flux_irregulier(1500)),
                 ('aléatoire pur',flux_aleatoire(1500))]:
    L=D17.CacheLRU(source,cap)
    t0=time.time()
    for a in flux: L.lire(a)
    tl=time.time()-t0
    P=lambda a: 1.0 if a%97==0 else 0.0            # quelques adresses à 100 %
    C=D17.DeltaCache(source,cap,P=P,prefetch=True,profondeur=6)
    t0=time.time()
    for a in flux: C.lire(a)
    tc=time.time()-t0
    C.fermer()
    res.append(dict(flux=nom,lru_taux=L.taux(),d_taux=C.taux(),lru_t=tl,d_t=tc,
                    lat_lru=L.latence_moyenne_us(),lat_d=C.latence_moyenne_us(),
                    lat_succes=C.latence_succes_us(),lat_echec=C.latence_echec_us(),
                    precharges=C.precharges,octets=C.octets()))
    print(f'{nom:30s} | LRU : {100*L.taux():5.1f} % de succès, {L.latence_moyenne_us():7.2f} µs/accès, {tl:5.2f} s'
          f' | Δ-CACHE : {100*C.taux():5.1f} %, {C.latence_moyenne_us():7.2f} µs/accès, {tc:5.2f} s'
          f' -> x{tl/max(tc,1e-9):4.2f}',flush=True)

print('\n=== DÉTAIL DES LATENCES Δ-CACHE (flux séquentiel) ===')
C=D17.DeltaCache(source,cap,prefetch=True,profondeur=6)
for a in range(1500): C.lire(a)
C.fermer()
print(f'succès : {C.latence_succes_us():6.2f} µs | échec : {C.latence_echec_us():7.2f} µs '
      f'| moyenne : {C.latence_moyenne_us():6.2f} µs | taux {100*C.taux():.1f} % '
      f'| préchargements {C.precharges} | mémoire {h(C.octets())}')
sans=D17.DeltaCache(source,cap,prefetch=False)
t0=time.time()
for a in range(1500): sans.lire(a)
t_sans=time.time()-t0
print(f'sans prédiction : {100*sans.taux():.1f} % de succès, {sans.latence_moyenne_us():.2f} µs/accès, {t_sans:.2f} s')

print('\n=== PRIORITÉ % : les lignes à 100 % survivent à la pression ===')
petit=1*1024*1024
P=lambda a: 1.0 if a<8 else 0.0
C2=D17.DeltaCache(source,petit,P=P,prefetch=False)
for a in range(400): C2.lire(a)
gardees=[a for a in range(8) if a in C2.lignes]
print(f'cache de {h(petit)}, 400 lignes parcourues : lignes à 100 % encore présentes : {gardees}')
L2=D17.CacheLRU(source,petit)
for a in range(400): L2.lire(a)
print(f'LRU dans les mêmes conditions : lignes 0-7 présentes : {[a for a in range(8) if a in L2.lignes]}')

print('\n=== ECC SUR LE CACHE ===')
print('lignes valides :',C2.valide())
list(C2.lignes.values())[0].data[0]+=1e-3
print('une ligne altérée -> détectée :',not C2.valide())
json.dump(res,open('resultats_delta17.json','w'),indent=1,default=float)
