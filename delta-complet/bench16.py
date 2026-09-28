import numpy as np, time, math, json, resource, delta16 as D16
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:8.1f} {u}'
        x/=1024
    return f'{x:.1f} Tio'
rss=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
rng=np.random.default_rng(3)

print('=== 1. DÉDUPLICATION : un fond répété + quelques pages allumées ===')
b=128; npages=2000
fond=np.outer(np.sin(np.linspace(0,3,b)), np.cos(np.linspace(0,2,b)))+0.3
M=D16.MemoireDelta(eps=1e-6)
for k in range(npages):
    page = fond if k % 50 else fond + rng.standard_normal((b,b))*0.05   # 2 % de pages uniques
    M.ecrire(k, page)
print(f'{npages} pages écrites | pages réellement stockées : {len(M.pages)} '
      f'| facteur de déduplication x{M.dedup():.1f}')
print(f'mémoire brute {h(M.octets_bruts)} -> Δ {h(M.octets())}  (x{M.octets_bruts/M.octets():.0f} moins)')

print('\n=== 2. X + % : compression et précision par page ===')
for eps,P,label in [(1e-12,1.0,'eps=1e-12, 100 %'),(1e-6,1.0,'eps=1e-6, 100 %'),
                    (1e-6,0.5,'eps=1e-6, 50 % (float32)'),(1e-4,0.1,'eps=1e-4, 10 % (float16)')]:
    M2=D16.MemoireDelta(eps=eps)
    for k in range(200):
        M2.ecrire(k, fond + rng.standard_normal((b,b))*0.02, P=P)
    err=max(np.abs(M2.lire(k)-M2.pages[M2.table[k]].lire()).max() for k in range(5))
    ref=fond+0
    e2=np.abs(M2.lire(0)-ref).max()
    print(f'{label:28s} : {h(M2.octets())} | rang moyen {np.mean([p.X for p in M2.pages.values()]):5.1f} '
          f'| écart page/original {e2:.1e}')

print('\n=== 3. CÔNE : on ne matérialise que ce qu on lit ===')
M3=D16.MemoireDelta(eps=1e-6)
for k in range(npages): M3.ecrire(k, fond if k%50 else fond+rng.standard_normal((b,b))*0.05)
t0=time.time()
for k in range(0,npages,100): M3.lire(k)
dt=time.time()-t0
print(f'{npages} pages en mémoire, {M3.materialisees} matérialisées ({100*M3.materialisees/npages:.1f} %) '
      f'en {dt*1000:.1f} ms')

print('\n=== 4. TAMPONS : anneau avec préchargement contre tout charger ===')
tuile=1<<20        # 1 Mio par tuile
nb=512             # 512 Mio de flux
def prod(k):
    x=np.arange(k*tuile//8,(k+1)*tuile//8,dtype=np.float64)
    return np.sin(x*1e-6)
avant=rss()
t0=time.time(); total=0.0; octets=0
for k in range(nb):
    t=prod(k); total+=float(t.sum()); octets+=t.nbytes
t_naif=time.time()-t0; rss_naif=rss()
A=D16.Anneau(prod,nb,profondeur=3)
t0=time.time(); total2=0.0
for t in A.parcourir(): total2+=float(t.sum())
t_ann=time.time()-t0
print(f'sans tampon  : {h(octets)} traités en {t_naif:5.2f} s -> {D16.bande_passante(octets,t_naif):5.2f} Go/s')
print(f'anneau (3)   : {h(A.octets)} traités en {t_ann:5.2f} s -> {D16.bande_passante(A.octets,t_ann):5.2f} Go/s '
      f'| mémoire vive du tampon {h(A.pic)} | écart des résultats {abs(total-total2):.1e}')

print('\n=== 5. TAILLE DE TUILE : trouver l optimum (comme le lot de 512) ===')
for kib in [16,64,256,1024,4096,16384]:
    tt=kib*1024; nbt=max(1,(256*1024*1024)//tt)
    def p2(k,tt=tt):
        x=np.arange(k*tt//8,(k+1)*tt//8,dtype=np.float64); return np.sin(x*1e-6)
    A2=D16.Anneau(p2,nbt,profondeur=3)
    t0=time.time(); s=0.0
    for t in A2.parcourir(): s+=float(t.sum())
    dt=time.time()-t0
    print(f'tuile {kib:6d} Kio : {D16.bande_passante(A2.octets,dt):5.2f} Go/s | '
          f'{nbt/dt:9.0f} tuiles/s | mémoire tampon {h(A2.pic)}')

print('\n=== 6. ECC SUR LA MÉMOIRE ===')
print('toutes les pages valides :',M.valide())
p=next(iter(M.pages.values()))
(p.plein if p.plein is not None else p.U).flat[0]+=1e-3
print('une page altérée -> détectée :', not M.valide())
parts,par=M.repartir(8)
tot=sum(len(v) for v in parts.values())
print(f'8 nœuds : {h(tot)} de pages, parité {h(par.nbytes)} -> surcoût {par.nbytes/tot*100:.1f} %')
