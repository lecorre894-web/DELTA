import numpy as np, math, time, json, delta8 as D8, delta9 as D9, duel as DU
from bench8 import circuit
n=16
circ=circuit(n,12,7,'monstre')
Cd=[(p,{i:U for i,U in zip(range(p,n-1,2),g2)},{}) for p,g2,g1 in circ]
ref=DU.exact(n,Cd)
fid=lambda s:(lambda v: abs(np.vdot(ref,v))**2/np.vdot(v,v).real)(s.dense())

print('=== 1. SVD RANDOMISÉE GARDÉE (le garde-fou rattrape les tirages ratés) ===')
for label,kw in [('SVD exacte',dict(alea=False)),
                 ('randomisée, garde 1e-10',dict(alea=True,tol_alea=1e-10)),
                 ('randomisée, garde 1e-6',dict(alea=True,tol_alea=1e-6)),
                 ('randomisée, garde relâchée 1e-2',dict(alea=True,tol_alea=1e-2))]:
    s=D9.Delta9(n,X_base=256,eps=1e-6,**kw); t0=time.time(); D8.executer(s,circ); dt=time.time()-t0
    print(f'{label:32s} fid={fid(s):.8f} t={dt*1000:7.1f} ms  décompositions: {s.stats}')

print('\n=== 2. REPRISE SUR SATURATION (au lieu d abandonner) ===')
P=np.zeros(n)
for pol in ['arret','reprise']:
    s=D9.Delta9(n,P=P,X_base=8,eps=1e-6,budget_octets=4096*1024,politique=pol,alea=True)
    t0=time.time()
    try:
        D8.executer(s,circ); r=f'FINI  fid={fid(s):.8f}  X_pic={s.X_pic}  reprises={len(s.reprises)}'
    except D8.SaturationDelta as e:
        r=f'ARRÊT -> {e}'
    print(f'politique={pol:8s} : {r}   ({(time.time()-t0)*1000:.0f} ms)')
    if pol=='reprise' and s.reprises:
        print('   3 premières reprises :', [{k:v for k,v in d.items() if k in ("couche","lien","X_accorde","X_requis","facteur")} for d in s.reprises[:3]])

print('\n=== 3. ECC DISTRIBUÉ : 4 nœuds + parité, on tue un nœud et on reconstruit ===')
s=D9.Delta9(n,X_base=256,eps=1e-6,alea=True); D8.executer(s,circ)
avant=s.dense().copy()
noeuds,ecc=s.repartir(4)
taille=sum(len(v['bloc'])*16 for v in noeuds.values())
print(f'état réparti : {taille/1024:.1f} Kio sur 4 nœuds + parité {ecc.parite.nbytes/1024:.1f} Kio '
      f'(surcoût {ecc.parite.nbytes/taille*100:.1f} %)')
print('vérification CRC de chaque nœud :', all(ecc.verifier(k,v['bloc']) for k,v in noeuds.items()))
# corruption d un octet sur le nœud 2
v=noeuds[2]['bloc'].copy(); vb=v.view(np.uint8); vb[123]^=0xFF
print('nœud 2 corrompu d un seul octet -> CRC détecte :', not ecc.verifier(2,v))
# panne totale du nœud 1 : reconstruction par XOR
perdu=noeuds.pop(1)
rec,ok=ecc.reconstruire(1,{k:v['bloc'] for k,v in noeuds.items()})
print(f'nœud 1 perdu puis reconstruit : CRC valide={ok}, identique au bit près={np.array_equal(rec,perdu["bloc"])}')
# on remonte l état et on revérifie la fidélité
noeuds[1]=dict(perdu,bloc=rec)
plat=np.concatenate([noeuds[k]['bloc'] for k in sorted(noeuds)])
G2=[];pos=0
for k in sorted(noeuds):
    for f in noeuds[k]['formes']:
        m=int(np.prod(f)); G2.append(noeuds[k]['bloc'][pos:pos+m].reshape(f) if False else None); pos+=m
    pos=0
s.G=[]; pos=0
for k in sorted(noeuds):
    p2=0
    for f in noeuds[k]['formes']:
        m=int(np.prod(f)); s.G.append(noeuds[k]['bloc'][p2:p2+m].reshape(f)); p2+=m
apres=s.dense()
print('état après reconstruction identique :', np.allclose(avant,apres,atol=0,rtol=0))
json.dump({'stats':s.stats},open('resultats_delta9.json','w'),indent=1)
