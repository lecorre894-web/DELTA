import numpy as np, time, math, json, delta13 as D13
def h(b):
    for u in ['o','Kio','Mio','Gio']:
        if b<1024: return f'{b:6.1f} {u}'
        b/=1024
    return f'{b:.1f} Tio'
res={}
print('=== 1. X À LA DEMANDE SUR DES DONNÉES (rang des blocs) ===')
n=2048
K=D13.kernel_lisse(n)
v=np.random.default_rng(1).standard_normal(n)
t0=time.time(); yref=K@v; t_dense=time.time()-t0
f_dense=2*n*n
print(f'dense {n}x{n} : {h(K.nbytes)}, matvec {t_dense*1000:6.2f} ms, {f_dense/1e6:.1f} Mflop')
for eps in [1e-4,1e-8,1e-12]:
    A=D13.MatriceDelta(K,taille=256,eps=eps)
    y,f=A.matvec(v)
    err=np.linalg.norm(y-yref)/np.linalg.norm(yref)
    t0=time.time()
    for _ in range(5): A.matvec(v)
    t=(time.time()-t0)/5
    print(f'Δ eps={eps:<6} : {h(A.octets())} (x{A.octets_dense()/A.octets():5.1f} moins), '
          f'rangs {min(A.rangs())}-{max(A.rangs())}, matvec {t*1000:6.2f} ms, {f/1e6:6.2f} Mflop '
          f'(x{f_dense/f:5.1f} moins), erreur relative {err:.2e}')
    res[f'eps{eps}']=dict(octets=A.octets(),flops=f,err=err,t=t)

print('\n=== 2. % = PRÉCISION PAR BLOC (float64 / float32 / float16) ===')
nb=n//256
P=[[1.0 if a==b else (0.5 if abs(a-b)==1 else 0.1) for b in range(nb)] for a in range(nb)]
for label,PP in [('tout en float64',None),('% par bloc (diag 100 %, voisins 50 %, loin 10 %)',P)]:
    A=D13.MatriceDelta(K,taille=256,eps=1e-8,P=PP)
    y,f=A.matvec(v); err=np.linalg.norm(y-yref)/np.linalg.norm(yref)
    print(f'{label:52s} : {h(A.octets())}, erreur {err:.2e}')

print('\n=== 3. CÔNE : ne calculer QUE les sorties demandées ===')
A=D13.MatriceDelta(K,taille=256,eps=1e-8)
y,f_tout=A.matvec(v)
for k in [1,2,4]:
    lignes=set(range(k))
    t0=time.time(); y2,f2=A.matvec(v,lignes=lignes); t=time.time()-t0
    ec=np.linalg.norm(y2[:k*256]-yref[:k*256])/np.linalg.norm(yref[:k*256])
    print(f'{k} bloc(s) de sortie sur {len(A.li)-1} : {f2/1e6:5.2f} Mflop (x{f_tout/f2:4.1f} moins), '
          f'{t*1000:5.2f} ms, erreur sur la part demandée {ec:.2e}')

print('\n=== 4. SYMÉTRIE : blocs nuls par construction ===')
K2=K.copy(); nb=n//256
for a in range(nb):
    for b in range(nb):
        if abs(a-b)>2: K2[a*256:(a+1)*256,b*256:(b+1)*256]=0.0      # matrice à bande
A=D13.MatriceDelta(K2,taille=256,eps=1e-8,nul_tol=0.0)
B=D13.MatriceDelta(K2,taille=256,eps=1e-8,nul_tol=1e-15)
y1,f1=A.matvec(v); y2,f2=B.matvec(v)
print(f'sans détection des blocs nuls : {h(A.octets())}, {f1/1e6:.2f} Mflop')
print(f'avec détection ({B.nuls} blocs nuls ignorés) : {h(B.octets())}, {f2/1e6:.2f} Mflop '
      f'-> x{f1/f2:.1f} de flops en moins, écart sur le résultat {np.abs(y1-y2).max():.1e}')

print('\n=== 5. ECC : empreinte par bloc, parité entre nœuds ===')
A=D13.MatriceDelta(K,taille=256,eps=1e-8)
print('toutes les empreintes valides :',A.verifier_tout())
b=A.blocs[(0,1)]
if b.plein is not None: b.plein.flat[0]+=1e-3
else: b.U.flat[0]+=1e-3
print('un bloc altéré -> détecté :', not A.verifier_tout())
noeuds,parite=A.repartir(4)
print(f'répartition sur 4 nœuds : {[len(x) for x in noeuds.values()]} blocs, parité {h(parite.nbytes)}')

print('\n=== 6. LA LIMITE : données SANS structure (bruit pur) ===')
R=np.random.default_rng(0).standard_normal((1024,1024))
A=D13.MatriceDelta(R,taille=256,eps=1e-8)
y,f=A.matvec(R[0]*0+v[:1024])
print(f'matrice aléatoire 1024 : {h(A.octets())} contre {h(A.octets_dense())} dense, '
      f'rangs {min(A.rangs())}-{max(A.rangs())}, flops {f/1e6:.2f} Mflop contre {2*1024*1024/1e6:.2f} '
      f'-> AUCUN gain, et c est normal : sans structure, rien à compresser.')
json.dump(res,open('resultats_delta13.json','w'),indent=1,default=float)
