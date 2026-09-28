import numpy as np, time, delta16 as D16
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:8.1f} {u}'
        x/=1024
rng=np.random.default_rng(3); b=256
def page_structuree(k):
    """Page à structure réelle : quelques modes + un peu de bruit (cas physique)."""
    x=np.linspace(0,1,b)
    P=sum(np.outer(np.sin((j+1+0.1*k)*np.pi*x), np.cos((j+1)*np.pi*x))/(j+1) for j in range(4))
    return P + rng.standard_normal((b,b))*1e-7

print('=== 2 bis. X (rang) ET % (précision) SUR DES PAGES STRUCTURÉES ===')
ref=[page_structuree(k) for k in range(100)]
for eps,P,label in [(1e-12,1.0,'eps=1e-12, 100 %'),(1e-8,1.0,'eps=1e-8, 100 %'),
                    (1e-5,1.0,'eps=1e-5, 100 %'),(1e-5,0.5,'eps=1e-5, 50 % float32'),
                    (1e-3,0.1,'eps=1e-3, 10 % float16')]:
    M=D16.MemoireDelta(eps=eps)
    for k,p in enumerate(ref): M.ecrire(k,p,P=P)
    err=max(np.abs(M.lire(k)-ref[k]).max()/np.abs(ref[k]).max() for k in range(0,100,10))
    print(f'{label:24s} : {h(M.octets())} contre {h(M.octets_bruts)} bruts '
          f'(x{M.octets_bruts/M.octets():5.1f}) | rang moyen {np.mean([p.X for p in M.pages.values()]):5.1f} '
          f'| erreur relative {err:.1e}')

print('\n=== 4 bis. TAMPON EN ANNEAU AVEC UN VRAI CALCUL À RECOUVRIR ===')
tuile=1<<20; nb=256
def prod(k):
    x=np.arange(k*tuile//8,(k+1)*tuile//8,dtype=np.float64)
    return np.sin(x*1e-6)+np.cos(x*3e-7)          # production coûteuse (LOAD)
def calcul(t):
    return float(np.sqrt(np.abs(t)).sum()+np.tanh(t[:4096]).sum())   # COMPUTE coûteux
t0=time.time(); s1=0.0; oct1=0
for k in range(nb):
    t=prod(k); oct1+=t.nbytes; s1+=calcul(t)
t_naif=time.time()-t0
A=D16.Anneau(prod,nb,profondeur=3)
t0=time.time(); s2=0.0
for t in A.parcourir(): s2+=calcul(t)
t_ann=time.time()-t0
print(f'séquentiel (charger puis calculer) : {h(oct1)} en {t_naif:5.2f} s -> {D16.bande_passante(oct1,t_naif):5.2f} Go/s')
print(f'anneau LOAD || COMPUTE             : {h(A.octets)} en {t_ann:5.2f} s -> {D16.bande_passante(A.octets,t_ann):5.2f} Go/s '
      f'| gain x{t_naif/t_ann:4.2f} | tampon {h(A.pic)} | écart {abs(s1-s2):.1e}')
for prof in [1,2,4,8]:
    A2=D16.Anneau(prod,nb,profondeur=prof)
    t0=time.time(); s=0.0
    for t in A2.parcourir(): s+=calcul(t)
    d=time.time()-t0
    print(f'   profondeur {prof} : {d:5.2f} s -> {D16.bande_passante(A2.octets,d):5.2f} Go/s, tampon {h(A2.pic)}')
