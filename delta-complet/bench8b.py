import numpy as np, math, time, json, delta8 as D8, duel as DU
from bench8 import circuit
n=16
circ=circuit(n,12,7,'monstre')
Cd=[(p,{i:U for i,U in zip(range(p,n-1,2),g2)},{}) for p,g2,g1 in circ]
ref=DU.exact(n,Cd)
def fid(s):
    v=s.dense(); return abs(np.vdot(ref,v))**2/np.vdot(v,v).real
print('=== B (corrigé). X FIXE contre X À LA DEMANDE ===')
for label,kw in [('X fixe = 16',dict(fixe=True,X_base=16)),('X fixe = 64',dict(fixe=True,X_base=64)),
                 ('X fixe = 256',dict(fixe=True,X_base=256)),
                 ('demande eps=1e-4',dict(X_base=256,eps=1e-4)),('demande eps=1e-6',dict(X_base=256,eps=1e-6))]:
    s=D8.Delta8(n,**kw); t0=time.time(); D8.executer(s,circ); dt=time.time()-t0
    print(f'{label:18s} X_pic={s.X_pic:4d} mem={s.memoire()/1024:8.1f} Kio fid={fid(s):.6f} t={dt*1000:7.1f} ms')
print('\n=== C (corrigé). LES 3 ARBITRAGES — budget serré, mode AUDIT (on continue pour comparer) ===')
P=np.zeros(n); P[[3,4,11,12]]=0.999
circ2=circuit(n,10,7,'monstre'); Cd2=[(p,{i:U for i,U in zip(range(p,n-1,2),g2)},{}) for p,g2,g1 in circ2]
ref=DU.exact(n,Cd2)
def fid_local(v,q):
    def red(x):
        t=x.reshape((2,)*n); M=np.moveaxis(t,q,0).reshape(2,-1); return M@M.conj().T
    r,s2=red(ref),red(v/np.linalg.norm(v))
    return float(np.real(np.trace(r@s2))+2*math.sqrt(max(np.linalg.det(r).real,0)*max(np.linalg.det(s2).real,0)))
for bud in [64,128,256]:
    print(f'-- budget {bud} Kio --')
    for arb in ['pourcent','proportionnel','merite']:
        s=D8.Delta8(n,P=P,X_base=256,eps=1e-3,budget_octets=bud*1024,arbitrage=arb,politique='audit')
        D8.executer(s,circ2); v=s.dense()
        fl=np.mean([fid_local(v,q) for q in [3,4,11,12]])
        print(f'   {arb:14s} fid globale={fid(s):.6f}  fid qubits 100%={fl:.6f}  X_pic={s.X_pic:3d} mem={s.memoire()/1024:7.1f} Kio  alertes={len(s.alertes)}')
print('\n=== C bis. Mode ARRÊT (règle de René) : plus petit budget qui finit ===')
for arb in ['pourcent','proportionnel','merite']:
    seuil=None
    for bud in [64,96,128,192,256,384,512,768,1024,1536,2048,3072]:
        s=D8.Delta8(n,P=P,X_base=256,eps=1e-3,budget_octets=bud*1024,arbitrage=arb)
        try:
            D8.executer(s,circ2); seuil=(bud,fid(s)); break
        except D8.SaturationDelta: pass
    print(f'   {arb:14s} finit à partir de {seuil[0]:5d} Kio, fidélité {seuil[1]:.6f}' if seuil else f'   {arb}: jamais')
