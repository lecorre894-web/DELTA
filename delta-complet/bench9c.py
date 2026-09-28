import numpy as np, delta8 as D8, delta9 as D9
from bench8 import circuit
n=16; circ=circuit(n,12,7,'monstre')
s=D9.Delta9(n,X_base=256,eps=1e-6); D8.executer(s,circ)
avant=s.dense().copy()
print('=== ECC : découpe en parts d octets égales ===')
for N in [2,4,8,16]:
    noeuds,ecc=s.repartir(N)
    t=sum(len(v['bloc']) for v in noeuds.values())
    print(f'{N:2d} nœuds : données {t/1024:8.1f} Kio, parité {ecc.parite.nbytes/1024:7.1f} Kio -> surcoût {ecc.parite.nbytes/t*100:5.1f} % (idéal {100/N:.1f} %)')
noeuds,ecc=s.repartir(8)
print('\n=== ECC : détection et reconstruction (8 nœuds) ===')
v=noeuds[5]['bloc'].copy(); v[77]^=0xFF
print('corruption d un seul octet détectée par CRC :', not ecc.verifier(5,v))
perdu=noeuds.pop(3)
rec,ok=ecc.reconstruire(3,{k:v['bloc'] for k,v in noeuds.items()})
print(f'nœud 3 perdu -> reconstruit : CRC valide={ok}, identique au bit près={np.array_equal(rec,perdu["bloc"])}')
noeuds[3]=dict(perdu,bloc=rec)
s.remonter(noeuds,ecc)
print('état remonté identique au bit près :', np.array_equal(avant,s.dense()))
