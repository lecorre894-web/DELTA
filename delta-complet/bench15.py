import numpy as np, time, math, json, delta15 as D15, delta14 as D14
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:7.1f} {u}'
        x/=1024
    return f'{x:.1f} Tio'
b=128; eps=1e-8; noy=D15.noyau_lisse
print('=== 1. CONTRÔLE : le lot donne-t-il EXACTEMENT le même résultat ? ===')
O=D15.Omni(noy,b=b,eps=eps,portee_max=40)
F=D14.FondInfini(D14.noyau_lisse,b=b,eps=eps,portee_max=40)
D=D14.DeltaFlops(F)
four=D15.fournisseur(b)
rng=np.random.default_rng(5)
G=rng.standard_normal((b,b))*0.2
O.allumer(1000,G); D.allumer(1000,G)
Y,_=O.lot(995,12,four)
ec=0
for j,k in enumerate(range(995,1007)):
    y,_=D.bloc(k,four); ec=max(ec,np.abs(Y[:,j]-y).max()/max(np.abs(y).max(),1e-30))
print(f'écart relatif max entre lot (Δ15) et une par une (Δ14) : {ec:.2e}')

print('\n=== 2. LE GAIN DU LOT : même calcul, même flops, débit multiplié ===')
res=[]
for nk in [1,8,64,512,4096]:
    O2=D15.Omni(noy,b=b,eps=eps,portee_max=40); O2.allumer(10**6//b,G)
    t0=time.time()
    tot=0
    for start in range(0,4096,nk):
        Y,f=O2.lot(10**6//b+start,nk,four); tot+=f
    dt=time.time()-t0
    res.append(dict(nk=nk,t=dt,gflops=O2.gflops(),inc=O2.inconnues_par_seconde()))
    print(f'lot de {nk:5d} blocs : 4096 blocs calculés en {dt:6.2f} s -> {O2.gflops():6.2f} Gflops | '
          f'{O2.inconnues_par_seconde():12,.0f} inconnues/s',flush=True)

print('\n=== 3. MAXIMUM : le débit tient-il à l échelle infinie ? ===')
for n in [10**6,10**12,10**18]:
    O3=D15.Omni(noy,b=b,eps=eps,portee_max=40)
    for p in range(0,20): O3.allumer(n//b//2+p*997,G)
    t0=time.time(); Y,f=O3.lot(n//b//2,4096,four); dt=time.time()-t0
    print(f'n={n:>22,} | 4096 blocs ({4096*b:,} inconnues) en {dt:5.2f} s | {O3.gflops():6.2f} Gflops | '
          f'{O3.inconnues_par_seconde():12,.0f} inconnues/s | mémoire {h(O3.octets())} '
          f'| opérateur explicite 10^{math.log10(8)+2*math.log10(n):.0f} o',flush=True)

print('\n=== 4. CÔNE + % + SYMÉTRIE dans le lot ===')
O4=D15.Omni(noy,b=b,eps=eps,portee_max=200)
print(f'classes retenues {len(O4.U)} (ignorées car négligeables : {O4.ignorees}) | '
      f'float64 : {sum(1 for c in O4.dtypes if O4.dtypes[c]==np.float64)} classes, '
      f'float32 : {sum(1 for c in O4.dtypes if O4.dtypes[c]==np.float32)} | mémoire {h(O4.octets())}')
Y,f_tout=O4.lot(1000,512,four)
O5=D15.Omni(noy,b=b,eps=eps,portee_max=200)
Y2,f_cone=O5.lot(1000,512,four,sorties=range(1000,1008))
print(f'512 blocs demandés : {f_tout/1e6:8.2f} Mflop | 8 blocs seulement : {f_cone/1e6:8.2f} Mflop '
      f'(le cône coupe la SORTIE, pas encore le calcul : piste V1.6)')

print('\n=== 5. ECC + NŒUDS (découpe à l octet) ===')
O6=D15.Omni(noy,b=b,eps=eps,portee_max=40)
for p in range(8): O6.allumer(p*1000,G)
print('empreintes valides :',O6.verifier())
noeuds,par=O6.repartir(8)
t=sum(len(v) for v in noeuds.values())
print(f'8 nœuds : données {h(t)}, parité {h(par.nbytes)} -> surcoût {par.nbytes/t*100:.1f} % (idéal 12.5 %)')
json.dump(res,open('resultats_delta15.json','w'),indent=1,default=float)
