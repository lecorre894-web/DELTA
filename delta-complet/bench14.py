import numpy as np, time, math, json, delta14 as D14
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:7.1f} {u}'
        x/=1024
    return f'{x:.1f} Tio'
b=128; eps=1e-8
noy=lambda d: D14.noyau_lisse(d)
print('=== 1. CONTRÔLE : fond + grappes contre opérateur explicite (n = 4096) ===')
n=4096; nb=n//b
fond=D14.FondInfini(noy,b=b,eps=eps,portee_max=nb)
D=D14.DeltaFlops(fond)
rng=np.random.default_rng(7)
grappes={5:rng.standard_normal((b,b))*0.3, 20:rng.standard_normal((b,b))*0.3}
for p,G in grappes.items(): D.allumer(p,G)
four=D14.fournisseur_procedural(b)
v=np.concatenate([four(k) for k in range(nb)])
M=D14.dense_reference(n,b,noy,grappes)
yref=M@v
err=[]
for k in [0,5,10,20,31]:
    y,f=D.bloc(k,lambda j: four(j) if 0<=j<nb else np.zeros(b))
    err.append(np.linalg.norm(y-yref[k*b:(k+1)*b])/max(np.linalg.norm(yref[k*b:(k+1)*b]),1e-30))
print(f'erreur relative max sur 5 blocs de sortie : {max(err):.2e}')
print(f'mémoire : fond {h(fond.octets())} + {len(D.grappes)} grappes {h(sum(g.octets() for g in D.grappes.values()))} '
      f'| opérateur explicite : {h(M.nbytes)}  -> x{M.nbytes/D.octets():.0f} moins')
print(f'portée retenue : {fond.portee} classes de distance, rangs {min(fond.rangs.values())}-{max(fond.rangs.values())}')

print('\n=== 2. COÛT D UNE QUESTION LOCALE : constant quelle que soit la taille ===')
res=[]
for n in [10**4, 10**6, 10**9, 10**12, 10**15]:
    nb=n//b
    D2=D14.DeltaFlops(fond)
    D2.allumer(nb//2, rng.standard_normal((b,b))*0.3)
    four=D14.fournisseur_procedural(b)
    k=nb//2
    for _ in range(20): D2.bloc(k, four)          # 20 questions
    r=dict(n=n,mem=D2.octets(),t_par_question=D2.temps/20,flops=D2.flops/20,
           gflops=D2.gflops(),ops=D2.ops_par_seconde(),
           mem_dense_log10=math.log10(8)+2*math.log10(n))
    res.append(r)
    print(f'n={n:>16,} | mémoire {h(D2.octets())} | {D2.temps/20*1e6:7.1f} µs par question | '
          f'{D2.flops/20/1e3:7.1f} kflop | {D2.gflops():5.2f} Gflops | {D2.ops_par_seconde():9.0f} questions/s | '
          f'opérateur explicite : 10^{math.log10(8)+2*math.log10(n):.0f} o',flush=True)

print('\n=== 3. DÉBIT SOUTENU : beaucoup de blocs de sortie d affilée ===')
nb=10**6//b
D3=D14.DeltaFlops(fond)
for p in range(0,nb,997): D3.allumer(p, rng.standard_normal((b,b))*0.1)
four=D14.fournisseur_procedural(b)
t0=time.time()
N=2000
for k in range(N): D3.bloc(k+nb//3, four)
dt=time.time()-t0
print(f'{N} blocs de sortie ({N*b:,} inconnues calculées) en {dt:.2f} s')
print(f'  -> {D3.gflops():.2f} Gflops soutenus | {D3.ops_par_seconde():,.0f} blocs/s | '
      f'{D3.ops_par_seconde()*b:,.0f} inconnues/s | mémoire totale {h(D3.octets())} '
      f'({len(D3.grappes)} grappes allumées)')

print('\n=== 4. NŒUDS + ECC ===')
noeuds,parite=D3.repartir(4)
print(f'répartition sur 4 nœuds : {[len(v) for v in noeuds.values()]} éléments | parité {h(parite.nbytes)} '
      f'({parite.nbytes/D3.octets()*100:.1f} % de surcoût)')
json.dump(res,open('resultats_delta14.json','w'),indent=1,default=float)
