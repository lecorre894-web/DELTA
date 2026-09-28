import numpy as np, time, json, delta19 as D19
def h(x):
    for u in ['o','Kio','Mio','Gio','Tio','Pio','Eio','Zio','Yio']:
        if x<1024: return f'{x:8.1f} {u}'
        x/=1024
    return f'10^{np.log10(max(x,1e-300))+24*np.log10(1024)/24:.0f} Yio'
taille=2048
def source(a):
    x=np.arange(a*taille,(a+1)*taille,dtype=np.float64)
    return np.sin(x*1e-5)+0.4*np.cos(x*3e-6)
verite={}
def src(a):
    if a not in verite: verite[a]=source(a)
    return verite[a]

print('=== 1. LE NOMBRE D ÉTAGES COÛTE-T-IL EN LATENCE ? ===')
flux=list(range(400))*3
for niveaux in [3,16,64,256]:
    C=D19.X3D(src,niveaux=niveaux,capacite_L1=128*1024,facteur=1.25)
    t0=time.time()
    for a in flux: C.lire(a)
    dt=time.time()-t0
    print(f'{niveaux:4d} étages | capacité totale {h(C.capacite_totale())} | occupé {h(C.octets())} | '
          f'{100*C.taux():5.1f} % de succès | {C.latence_us():6.2f} µs/accès | {dt:5.2f} s',flush=True)

print('\n=== 2. L ÉVICTION DEVIENT DÉGRADATION : rien n est perdu ===')
C=D19.X3D(src,niveaux=256,capacite_L1=64*1024,facteur=1.2)
K=D19.CacheClassique(src,caps=(64*1024,256*1024,1024*1024))
flux=list(range(1200))
for a in flux: C.lire(a); K.lire(a)
# on redemande les toutes premières adresses, longtemps après
err=[]
for a in range(12):
    y=C.lire(a); err.append(np.abs(y-verite[a]).max()/np.abs(verite[a]).max())
print(f'Δ-X3D   : {100*C.taux():5.1f} % de succès | erreur sur les plus vieilles lignes '
      f'{np.mean(err):.2e} (min {min(err):.1e}, max {max(err):.1e}) | {C.degradations} dégradations')
print(f'classique : {100*K.taux():5.1f} % de succès | {K.perdues} lignes DÉFINITIVEMENT perdues')
print(f'mémoire Δ {h(C.octets())} contre classique {h(sum(K.octets(k) for k in range(3)))}')

print('\n=== 3. OÙ VIVENT LES DONNÉES (profil de la pile 3D) ===')
for niveau,bits,n,oct_ in C.occupation()[:12]:
    print(f'  étage {niveau:3d} : {bits:2d} bits/chiffre, {n:5d} lignes, {h(oct_)}')
print(f'  ... {len(C.occupation())} étages occupés au total')

print('\n=== 4. LE % PROTÈGE LES LIGNES QUI COMPTENT ===')
P=lambda a: 1.0 if a<6 else 0.0
C2=D19.X3D(src,niveaux=256,capacite_L1=64*1024,facteur=1.2,P=P)
for a in range(1200): C2.lire(a)
niv=[C2.repertoire.get(a) for a in range(6)]
print(f'après 1200 lignes, les lignes à 100 % sont aux étages {niv} (0 = le plus rapide)')
autres=[C2.repertoire.get(a) for a in range(6,18)]
print(f'les lignes ordinaires du même âge sont aux étages {autres}')

print('\n=== 5. CAPACITÉ APPARENTE : jusqu où va la pile ===')
for niveaux,fac in [(64,1.3),(128,1.3),(256,1.3),(256,1.5)]:
    C3=D19.X3D(src,niveaux=niveaux,capacite_L1=256*1024,facteur=fac)
    print(f'{niveaux:4d} étages, facteur {fac} -> capacité totale {h(C3.capacite_totale())}')
