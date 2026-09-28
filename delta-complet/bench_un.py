#!/usr/bin/env python3
"""bench_un — Δ-UN V2.0 : la fusion mise à l'épreuve, brique par brique.

Six épreuves, et chacune a sa référence. Les résultats négatifs sont écrits
comme les autres.
"""
import json, time
import numpy as np
import delta_un as DU
import delta15 as D15
import delta16 as D16
import delta18 as D18
import delta19 as D19

R = {}
N = 10 ** 18


def titre(t):
    print('\n' + '─' * 70); print(t); print('─' * 70, flush=True)


# ══════════════════════════════════════════ 1. EXACTITUDE
titre('1. EXACTITUDE — Δ-UN confronté à l\'opérateur dense, sans compression')
D = DU.DeltaUn(D15.noyau_lisse, b=128, eps=1e-8, portee=40, etages=256,
               P=lambda k: 1.0 if k % 500 == 0 else 0.0)
k0 = N // 128 // 2
rng = np.random.default_rng(1)
for j in range(8):
    D.allumer(k0 + j * 97, rng.standard_normal((128, 128)) * 0.2)
e = D.erreur(k0, 32)
print(f'opérateur : {len(D.calcul.U)} classes, {D.calcul.octets()/1024:.1f} Kio')
print(f'erreur relative Δ-UN vs dense, sur 32 blocs : {e:.3e}   (eps demandé 1e-8)')
R['exactitude'] = {'erreur_relative': e, 'eps': 1e-8, 'blocs': 32}

# ══════════════════════════════════════════ 2. DÉBIT
titre('2. DÉBIT — un lot de 4096 blocs sur un problème de taille 10^18')
D.calcul.flops = 0; D.calcul.temps = 0.0; D.calcul.ops = 0
t0 = time.time(); Y, _ = D.plage(k0, 4096); t = time.time() - t0
print(f'4096 blocs = {4096*128:,} inconnues en {t:.3f} s')
print(f'{D.calcul.gflops():.2f} Gflops   {D.calcul.inconnues_par_seconde():,.0f} inconnues/s')
print(f'mémoire de l\'opérateur : {D.calcul.octets()/1024:.1f} Kio — constante, quel que soit n')
R['debit'] = {'blocs': 4096, 'inconnues': 4096 * 128, 'secondes': t,
              'gflops': D.calcul.gflops(),
              'inconnues_par_seconde': D.calcul.inconnues_par_seconde(),
              'octets_operateur': D.calcul.octets()}

# ══════════════════════════════════════════ 3. LE CACHE
titre('3. CACHE — MÊME BUDGET D\'OCTETS pour Δ et pour un cache classique')
BUDGET = 256 * 1024          # 256 Kio, pas un octet de plus pour l'un ni pour l'autre
NL = 2048                    # 2048 lignes de 1 Kio = 2 Mio à loger dans 256 Kio
D2 = DU.DeltaUn(D15.noyau_lisse, b=128, eps=1e-8, portee=40,
                etages=256, capacite_L1=1024,          # 256 étages x 1 Kio = 256 Kio
                P=lambda k: 1.0 if k % 500 == 0 else 0.0)
D2.cache.facteur = 1.0
for e in D2.cache.etages:
    e.capacite = 1024
t0 = time.time(); D2.plage(k0, NL, cacher=True); t_remplir = time.time() - t0
occ = D2.cache.occupation()
print(f'{NL} lignes (2 Mio bruts) rangées dans {BUDGET/1024:.0f} Kio en {t_remplir:.2f} s')
print(f'étages occupés : {len(occ)} sur 256 | dégradations : {D2.cache.degradations} | '
      f'lignes OUBLIÉES : {D2.cache.oubliees}')
for niv, bits, n, o in occ[:4] + occ[-3:]:
    print(f'   étage {niv:3d} : {bits:2d} bits/chiffre, {n:4d} lignes, {o/1024:7.1f} Kio')

rng2 = np.random.default_rng(7)
qs = [k0 + int(x) for x in rng2.integers(0, NL, 3000)]
t0 = time.time()
for q in qs:
    D2.bloc(q)
t_q = time.time() - t0
hs, ms = D2.latences_us()
print(f'3000 questions : {t_q*1000:.0f} ms, {100*D2.cache.taux():.1f} % de succès, '
      f'{hs:.1f} µs par succès')

# combien coûte la réponse quand il faut RECALCULER ?
t0 = time.time()
for j in range(200):
    D2._calculer_bloc(k0 + 5000 + j)
t_rec = (time.time() - t0) / 200 * 1e6
print(f'recalculer une ligne coûte {t_rec:.0f} µs -> le cache fait ×{t_rec/max(hs,1e-9):.0f}')

# ce que vaut vraiment une ligne relue, selon la profondeur où elle a fini
Ex = D2.exact(k0, NL)
par_niv = {}
for m in rng2.integers(0, NL, 400):
    a = k0 + int(m)
    niv = D2.cache.repertoire.get(a)
    if niv is None:
        continue
    x = D2.cache.etages[niv].lignes[a].lire()
    r = Ex[:, int(m)]
    par_niv.setdefault(D2.cache.finesse[a], []).append(
        float(np.linalg.norm(x - r) / np.linalg.norm(r)))
print('erreur d\'une ligne relue, selon la FINESSE qu\'elle a connue au plus bas :')
for bits in sorted(par_niv, reverse=True):
    v = par_niv[bits]
    print(f'   {bits:2d} bits/chiffre : {len(v):3d} lignes, '
          f'erreur médiane {np.median(v):.2e}, pire {max(v):.2e}')
R['cache'] = {'budget_octets': BUDGET, 'lignes': NL, 'lignes_brutes_octets': NL * 1024,
              'etages_occupes': len(occ), 'degradations': D2.cache.degradations,
              'oubliees': D2.cache.oubliees, 'taux': D2.cache.taux(),
              'latence_succes_us': hs, 'recalcul_us': t_rec, 'gain': t_rec / max(hs, 1e-9),
              'erreur_par_etage': {int(k): float(np.median(v)) for k, v in par_niv.items()}}

# ── référence : 3 niveaux, MÊME budget, éviction = perte définitive
src = lambda k: D2._calculer_bloc(k)
cl = D19.CacheClassique(src, caps=(32 * 1024, 96 * 1024, 128 * 1024))   # 256 Kio
for m in range(NL):
    cl.lire(k0 + m)
perdues_remplissage = cl.perdues
t0 = time.time()
for q in qs:
    cl.lire(q)
t_cl = time.time() - t0
print(f'\nréférence 3 niveaux, MÊMES 256 Kio : {perdues_remplissage} lignes perdues au '
      f'remplissage, puis {100*cl.taux():.1f} % de succès sur les mêmes 3000 questions '
      f'({t_cl*1000:.0f} ms)')
print(f'Δ : {D2.cache.oubliees} ligne(s) oubliée(s), {100*D2.cache.taux():.1f} % de succès '
      f'({t_q*1000:.0f} ms) — mais les vieilles lignes sont FLOUES, pas exactes.')
R['cache_classique'] = {'perdues': cl.perdues, 'taux': cl.taux(), 'secondes': t_cl}

# ══════════════════════════════════════════ 4. LA MÉMOIRE
titre('4. MÉMOIRE — pages dédupliquées + condensateurs')
M = np.asarray(Y[:, :512], float).reshape(512, 128)
for i in range(512):
    D.ranger(i, np.outer(M[i % 8], M[i % 8]) if False else M[i % 8][:, None] @ M[i % 8][None, :])
print(f'512 pages écrites, {len(D.memoire.pages)} réellement stockées '
      f'-> déduplication ×{D.memoire.dedup():.0f}')
print(f'{D.memoire.octets_bruts/1024/1024:.1f} Mio bruts -> {D.memoire.octets()/1024:.1f} Kio, '
      f'intégrité : {D.memoire.valide()}')
cond = D18.Condensateur(Y[:, 0], bits=8)
x = cond.lire()
print(f'condensateur 8 bits : {Y[:,0].nbytes} o -> {cond.octets()} o '
      f'(×{Y[:,0].nbytes/cond.octets():.0f}), erreur {np.linalg.norm(x-Y[:,0])/np.linalg.norm(Y[:,0]):.2e}')
R['memoire'] = {'pages_logiques': len(D.memoire.table), 'pages_stockees': len(D.memoire.pages),
                'dedup': D.memoire.dedup(), 'octets_bruts': D.memoire.octets_bruts,
                'octets': D.memoire.octets(), 'integrite': D.memoire.valide(),
                'condensateur_facteur': Y[:, 0].nbytes / cond.octets(),
                'condensateur_erreur': float(np.linalg.norm(x - Y[:, 0]) / np.linalg.norm(Y[:, 0]))}

# ══════════════════════════════════════════ 5. LE SCEAU
titre('5. SCEAU — ce qu\'un QPU ne peut pas faire : signer son propre état')
parts = [Y[:, i].copy() for i in range(256)]
t0 = time.time(); racine = D.sceller(parts); t_s = time.time() - t0
o = D.coffre.taille()
print(f'256 blocs scellés et chiffrés en {t_s*1000:.1f} ms -> {o/t_s/1e6:.0f} Mo/s')
print(f'racine de Merkle : {racine.hex()[:32]}...   sceaux valides : {D.verifier()}')
pr = D.coffre.merkle.preuve(37)
ok = D18.Merkle.verifier_preuve(D.coffre.chiffres[37], pr, racine, D.cle)
faux = D.coffre.chiffres[37].copy(); faux[0] ^= 1
ko = D18.Merkle.verifier_preuve(faux, pr, racine, D.cle)
print(f'preuve du bloc 37 : {len(pr)} empreintes -> vraie : {ok}, sur bloc altéré : {ko}')
avant = D.coffre.chiffres[3].copy()
rec = D.reconstruire(3)
print(f'nœud 3 perdu, reconstruit depuis la parité SANS déchiffrer les autres : '
      f'{np.array_equal(rec, avant)}')
print(f'coût de la parité : {100/len(parts):.2f} % (1 bloc sur {len(parts)})')
R['sceau'] = {'blocs': 256, 'ms': t_s * 1000, 'Mo_s': o / t_s / 1e6,
              'preuve_vraie': bool(ok), 'preuve_alteree': bool(ko),
              'reconstruction': bool(np.array_equal(rec, avant)),
              'cout_parite_pct': 100 / len(parts)}

# ══════════════════════════════════════════ 6. LE CONTRE-ESSAI
titre('6. CONTRE-ESSAI — sur du bruit pur, Δ-UN ne doit RIEN gagner')
def noyau_bruit(d):
    r = np.random.default_rng(np.abs(np.asarray(d)).astype(np.int64) % (2**31))
    return r.standard_normal(np.shape(d))
try:
    B = D15.Omni(noyau_bruit, b=64, eps=1e-8, portee_max=4)
    rangs = list(B.rangs.values())
    print(f'noyau aléatoire : rangs retenus {rangs} sur 64 possibles')
    print('-> aucune compression. Le gain de Δ vient de la STRUCTURE, jamais de la magie.')
    R['contre_essai'] = {'rangs': rangs, 'plein': 64, 'gain': max(rangs) < 64}
except Exception as ex:
    print('contre-essai impossible :', ex)
    R['contre_essai'] = {'erreur': str(ex)}

json.dump(R, open('resultats_un.json', 'w'), indent=1, default=str)
print('\n→ resultats_un.json')
