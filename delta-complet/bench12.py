import numpy as np, math, time, json
import delta10 as D10, delta11 as D11

def gflops_gemm(N=1024, rep=3):
    A = (np.random.rand(N,N)+1j*np.random.rand(N,N)).astype(np.complex128)
    B = A.copy(); A@B
    t0=time.time()
    for _ in range(rep): A@B
    dt=(time.time()-t0)/rep
    return 8*N**3/dt/1e9, dt      # complexe : 8 flops par multiplication-addition

def gflops_svd(N=512, rep=3):
    A=(np.random.rand(N,N)+1j*np.random.rand(N,N)).astype(np.complex128)
    np.linalg.svd(A,full_matrices=False)
    t0=time.time()
    for _ in range(rep): np.linalg.svd(A,full_matrices=False)
    dt=(time.time()-t0)/rep
    return 22*N**3/dt/1e9, dt

print('=== 1. PUISSANCE CRÊTE DE CETTE MACHINE (2 cœurs, complex128) ===')
g1,d1=gflops_gemm(1024); print(f'produit matriciel 1024x1024 : {g1:7.2f} Gflops   ({d1*1000:.0f} ms)')
g2,d2=gflops_svd(512);    print(f'SVD 512x512                 : {g2:7.2f} Gflops   ({d2*1000:.0f} ms)')
crete=max(g1,g2)

print('\n=== 2. FLOPS SOUTENUS PAR Δ (moteur symétrique Δ10) ===')
res=[]
for n,depth in [(32,12),(64,12),(128,12)]:
    occ=[(1 if k%2==0 else 0) for k in range(n)]
    circ=D10.circuit_symetrique(n,depth,3)
    s=D10.Delta10(n,occ,eps=1e-10,X_max=512)
    t0=time.time()
    for par,g in circ: s.couche(par,g)
    dt=time.time()-t0
    gf=s.flops/dt/1e9
    r=dict(n=n,flops=s.flops,flops_sans_blocs=s.flops_sans_blocs,t=dt,gflops=gf,X=s.X_pic)
    res.append(r)
    print(f'n={n:4d} : {s.flops/1e9:8.2f} Gflop exécutés en {dt:6.2f} s -> {gf:6.2f} Gflops '
          f'({100*gf/crete:5.1f} % de la crête) | évités par la symétrie : '
          f'{(s.flops_sans_blocs-s.flops)/1e9:9.2f} Gflop (x{s.flops_sans_blocs/max(s.flops,1):.2f})',flush=True)

print('\n=== 3. FLOPS ÉVITÉS PAR TOUTE L ARCHITECTURE (référence : vecteur d état 2^n) ===')
for n,depth in [(32,12),(64,12),(128,12)]:
    portes2=sum(len(range(d%2,n-1,2)) for d in range(depth))
    f_dense=portes2*32*(2.0**n)                    # une porte 2 qubits sur 2^n amplitudes
    f_delta=[r for r in res if r['n']==n][0]['flops']
    print(f'n={n:4d} : 2^n exigerait 10^{math.log10(f_dense):.1f} flop | Δ en a exécuté '
          f'{f_delta/1e9:.2f} Gflop -> rapport 10^{math.log10(f_dense/max(f_delta,1)):.1f}')

print('\n=== 4. FLOPS ÉVITÉS PAR LE TRAVAIL À LA DEMANDE (Δ11) ===')
n=4000
zones=[(k,6) for k in range(5,n-10,97)]
circ=D11.circuit_zones(n,10,4,zones)
a=D11.Delta11(n).executer(circ)
q=[z[0] for z in zones if z[0]>n//2][:1]*1
b=D11.Delta11(n).executer(circ,sites=[q[0],q[0]+1,q[0]+2])
print(f'registre de {n} qubits : tout={a.flops/1e6:8.2f} Mflop en {a.temps*1000:6.1f} ms | '
      f'cône={b.flops/1e6:6.3f} Mflop en {b.temps*1000:5.1f} ms | '
      f'flops évités x{a.flops/max(b.flops,1):.0f}')
json.dump(dict(crete=crete,gemm=g1,svd=g2,delta=res),open('resultats_flops.json','w'),indent=1,default=float)
