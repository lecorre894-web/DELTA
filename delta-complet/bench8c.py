import numpy as np, math, time, delta8 as D8, duel as DU
n=16
rng=np.random.default_rng(3)
def circuit_mixte(n,depth,seed):
    """Moitié gauche : portes de Haar (zone chaude). Moitié droite : rotations faibles (zone calme)."""
    rng=np.random.default_rng(seed); C=[]
    for d in range(depth):
        par=d%2; g=[]
        for i in range(par,n-1,2):
            if i < n//2:  g.append(DU.haar4(rng))
            else:
                a=rng.uniform(-0.15,0.15); ph=np.exp(-0.5j*a)
                g.append(np.diag([ph,ph.conj(),ph.conj(),ph]))
        C.append((par,g,None))
    return C
circ=circuit_mixte(n,12,7)
Cd=[(p,{i:U for i,U in zip(range(p,n-1,2),g2)},{}) for p,g2,g1 in circ]
ref=DU.exact(n,Cd)
f=lambda s:(lambda v: abs(np.vdot(ref,v))**2/np.vdot(v,v).real)(s.dense())
print('=== CIRCUIT MIXTE (zone chaude + zone calme) : là où X à la demande doit gagner ===')
for label,kw in [('X fixe = 32',dict(fixe=True,X_base=32)),('X fixe = 64',dict(fixe=True,X_base=64)),
                 ('X fixe = 128',dict(fixe=True,X_base=128)),
                 ('demande eps=1e-6',dict(X_base=256,eps=1e-6)),('demande eps=1e-8',dict(X_base=256,eps=1e-8))]:
    s=D8.Delta8(n,**kw); t0=time.time(); D8.executer(s,circ); dt=time.time()-t0
    Xs=[len(l) for l in s.L[1:-1]]
    print(f'{label:18s} fid={f(s):.6f} mem={s.memoire()/1024:8.1f} Kio t={dt*1000:6.1f} ms  X par lien={Xs}')
