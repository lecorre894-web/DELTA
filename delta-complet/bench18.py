import numpy as np, time, json, hashlib, delta18 as D18
def h(x):
    for u in ['o','Kio','Mio','Gio']:
        if x<1024: return f'{x:7.1f} {u}'
        x/=1024
rng=np.random.default_rng(4)
print('=== 1. CONDENSATEUR : plusieurs chiffres par cellule ===')
x=np.sin(np.linspace(0,50,1_000_000))+0.3*np.cos(np.linspace(0,13,1_000_000))
for bits,P in [(16,1.0),(8,0.5),(4,0.1)]:
    t0=time.time(); c=D18.Condensateur(x,bits=bits); t_pack=time.time()-t0
    t0=time.time(); y=c.lire(); t_unpack=time.time()-t0
    err=np.abs(y-x).max()/np.abs(x).max()
    print(f'{bits:2d} bits par chiffre ({c.par_mot} chiffres par mot de 64 bits) : '
          f'{h(c.octets())} contre {h(x.nbytes)} (x{x.nbytes/c.octets():4.1f}) | '
          f'erreur max {err:.2e} | empaquetage {t_pack*1000:5.0f} ms, lecture {t_unpack*1000:5.0f} ms')

print('\n=== 2. SCEAU À CLÉ contre CRC : la falsification ===')
import zlib
d=rng.standard_normal(100000)
o=np.ascontiguousarray(d).view(np.uint8).ravel()
crc=zlib.crc32(o.tobytes())
o2=o.copy(); o2[5]^=0xFF
# un attaquant ajuste 4 octets pour retrouver le même CRC : c est un calcul direct
print(f'CRC32 : 4 octets bien choisis suffisent à le retrouver -> détection contournable')
cle=hashlib.blake2b(b'cle de Rene', digest_size=32).digest()
s1=D18.sceau(cle,o); s2=D18.sceau(cle,o2)
print(f'sceau à clé : identique après falsification ? {s1==s2}  -> falsification impossible sans la clé')
t0=time.time()
for _ in range(20): D18.sceau(cle,o)
dt=(time.time()-t0)/20
print(f'débit du sceau : {o.nbytes/dt/1e6:.0f} Mo/s')

print('\n=== 3. ARBRE DE MERKLE : une racine pour tout l état ===')
blocs=[rng.standard_normal(20000) for _ in range(64)]
t0=time.time(); M=D18.Merkle([np.ascontiguousarray(b).view(np.uint8) for b in blocs],cle); t_m=time.time()-t0
tot=sum(b.nbytes for b in blocs)
print(f'{len(blocs)} blocs, {h(tot)} : racine construite en {t_m*1000:.0f} ms ({tot/t_m/1e6:.0f} Mo/s)')
print(f'racine : {M.racine.hex()[:32]}...')
oct_blocs=[np.ascontiguousarray(b).view(np.uint8).copy() for b in blocs]
oct_blocs[37][11]^=0x01
ok,mauvais=M.verifier(oct_blocs)
print(f'un octet modifié dans le bloc 37 -> tout bon ? {ok} | blocs fautifs désignés : {mauvais}')
pr=M.preuve(10)
print(f'preuve d authenticité d un bloc : {len(pr)} empreintes ({len(pr)*16} octets) pour {len(blocs)} blocs '
      f'-> vérifiée : {D18.Merkle.verifier_preuve(np.ascontiguousarray(blocs[10]).view(np.uint8),pr,M.racine,cle)}')

print('\n=== 4. COFFRE : chiffré, scellé, réparti, parité sur le CHIFFRÉ ===')
blocs=[rng.standard_normal(50000) for _ in range(8)]
t0=time.time(); C=D18.CoffreDelta(cle,blocs); t_c=time.time()-t0
tot=sum(b.nbytes for b in blocs)
print(f'{len(blocs)} blocs ({h(tot)}) chiffrés et scellés en {t_c*1000:.0f} ms -> {tot/t_c/1e6:.0f} Mo/s')
print('sceaux valides :',C.verifier())
C.chiffres[3][100]^=0xFF
print('un octet altéré dans le bloc 3 -> sceaux valides ?',C.verifier())
C.chiffres[3][100]^=0xFF
par=C.parite()
perdu=C.chiffres[2].copy()
rec=C.reconstruire(2,par)
print(f'bloc 2 perdu puis reconstruit SANS déchiffrer les autres : identique {np.array_equal(rec,perdu)}')
clair=C.dechiffrer(2).view(np.float64)
print(f'déchiffré = original : {np.array_equal(clair,blocs[2])} | parité {h(par.nbytes)} pour {h(tot)} ({par.nbytes/tot*100:.1f} %)')

print('\n=== 5. CE QU UN QPU NE PEUT PAS FAIRE ===')
print("un état quantique cohérent ne peut être ni copié (non-clonage) ni lu sans être détruit :")
print("  -> pas de sceau, pas d arbre de Merkle, pas de parité, pas de reconstruction.")
print("Δ étant classique, son état porte sa propre preuve d intégrité. C est un avantage")
print("de l émulation sur la machine, pas l inverse.")
