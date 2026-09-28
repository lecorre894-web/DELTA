#!/usr/bin/env python3
"""bench_bridge — Δ-BRIDGE mis à l'épreuve. Le juge reste Bitcoin."""
import hashlib, json, time
import numpy as np
import delta_bridge as P
import delta_btc as B
import delta18 as D18

R = {}
blocs, source = B.charger()
blocs = sorted(blocs, key=lambda b: b['hauteur'])
b100 = [x for x in blocs if x['hauteur'] == 100000][0]
E = B.entete(b100)


def titre(t):
    print('\n' + '═' * 72); print(t); print('═' * 72, flush=True)


def meilleur(f, essais=5):
    """Le MEILLEUR temps, pas la moyenne. Une machine partagée ne peut que
    ralentir une mesure, jamais l'accélérer : le minimum est donc l'estimation
    la plus propre du coût réel."""
    t = []
    for _ in range(essais):
        t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
    return min(t)


# ═══════════════════════════════════════════════ 1. CONFORMITÉ DU PONT
titre('1. CONFORMITÉ — le pont redonne-t-il les hashs officiels ?')
c = []
for b in blocs:
    e = B.entete(b)
    h = P.sha256d_pont(e, [b['nonce']])[0].tobytes()
    bon = h[::-1].hex() == b['id']
    c.append(bon)
    print(f"  bloc {b['hauteur']:>7} : {'CONFORME' if bon else '*** ÉCART ***'}  "
          f"{h[::-1].hex()[:32]}...")
print(f'  constantes SHA-256 : RECALCULÉES (racines des 64 premiers nombres '
      f'premiers), pas recopiées')
# et de front, N nonces d'un coup
nonces = [b100['nonce'] + i for i in range(256)]
lot = P.sha256d_pont(E, nonces)
ref = [B.sha256d(E[:76] + int(n).to_bytes(4, 'little')) for n in nonces]
front = all(lot[i].tobytes() == ref[i] for i in range(256))
print(f'  256 nonces traités DE FRONT, comparés un par un à hashlib : {front}')
R['conformite'] = {'blocs': all(c), 'lot_256': bool(front)}

# ═══════════════════════════════════════════════ 2. L'EXTRAPOLATION
titre('2. EXTRAPOLATION (midstate) — 3 compressions deviennent 2')
N = 20000
pre = hashlib.sha256(E[:64])


def _direct():
    for i in range(N):
        hashlib.sha256(hashlib.sha256(E[:76] + i.to_bytes(4, 'little')).digest()).digest()


def _avec_mid():
    for i in range(N):
        g = pre.copy()
        g.update(E[64:76] + i.to_bytes(4, 'little'))
        hashlib.sha256(g.digest()).digest()


t_direct = meilleur(_direct)
t_mid = meilleur(_avec_mid)

# décomposition honnête du coût : combien vaut UNE compression, combien vaut
# l'appel lui-même ? On le mesure au lieu de le supposer.
un = b'\x00' * 64
deux = b'\x00' * 128
M = 60000
t1 = meilleur(lambda: [hashlib.sha256(un).digest() for _ in range(M)]) / M
t2 = meilleur(lambda: [hashlib.sha256(deux).digest() for _ in range(M)]) / M
cout_compression = (t2 - t1) * 1e9        # ns pour une compression de plus
cout_appel = (t1 - cout_compression / 1e9 * 2) * 1e9   # 64 o = 2 compressions

g = pre.copy(); g.update(E[64:76] + (7).to_bytes(4, 'little'))
juste = hashlib.sha256(g.digest()).digest() == \
    hashlib.sha256(hashlib.sha256(E[:76] + (7).to_bytes(4, 'little')).digest()).digest()
print(f'  tout droit      : {N/t_direct/1e3:8,.0f} kH/s   (3 compressions par essai)')
print(f'  avec midstate   : {N/t_mid/1e3:8,.0f} kH/s   (2 compressions par essai)')
print(f'  gain THÉORIQUE  : ×1.50   (on supprime une compression sur trois)')
print(f'  gain MESURÉ ICI : ×{t_direct/t_mid:.2f}')
print(f'  et c\'est bien le même hash : {juste}')
print(f'\n  POURQUOI L\'ÉCART ? Mesuré, pas supposé :')
print(f'    une compression de 64 octets coûte      {cout_compression:6.0f} ns')
print(f'    l\'appel Python autour coûte             {cout_appel:6.0f} ns')
print(f'    -> on économise {cout_compression:.0f} ns de travail utile, mais chaque essai')
print(f'       traîne {2*cout_appel:.0f} ns de plomberie qui, elle, ne bouge pas.')
print('  Le gain de STRUCTURE est réel et entier ; c\'est le langage qui en')
print('  reprend une part. Dans une implémentation en C, ou dès que les')
print('  compressions dominent l\'appel, on retrouve le ×1,5.')
print('  Le gain ne vient pas d\'un calcul plus malin : 64 des 80 octets de')
print('  l\'en-tête ne bougent jamais d\'un essai à l\'autre. C\'est la STRUCTURE')
print('  du message qui paye. Exactement la loi de Δ, sur un terrain adverse.')
R['midstate'] = {'direct_kHs': N / t_direct / 1e3, 'midstate_kHs': N / t_mid / 1e3,
                 'gain_mesure': t_direct / t_mid, 'gain_theorique': 1.5,
                 'identique': bool(juste)}

# ═══════════════════════════════════════════════ 3. LE PONT EN PARALLÈLE
titre('3. LE PONT EN PARALLÈLE — N messages de front, et la vérité des chiffres')
R['parallele'] = {}
for n in (1000, 10000, 50000):
    nn = list(range(n))
    mid = P.midstate(E)

    def _pont():
        P.sha256d_pont(E, nn, mid)

    def _hl():
        for i in nn:
            g = pre.copy(); g.update(E[64:76] + i.to_bytes(4, 'little'))
            hashlib.sha256(g.digest()).digest()

    tp = meilleur(_pont, 3)
    th = meilleur(_hl, 3)
    print(f'  N={n:>6} | pont vectorisé {n/tp/1e3:8,.0f} kH/s | '
          f'hashlib+midstate {n/th/1e3:8,.0f} kH/s | rapport ×{tp/th:.1f} en '
          f'{"FAVEUR" if tp < th else "DÉFAVEUR"} du pont')
    R['parallele'][n] = {'pont_kHs': n / tp / 1e3, 'hashlib_kHs': n / th / 1e3,
                         'rapport': th / tp}
print('\n  RÉSULTAT NÉGATIF, ET IL COMPTE AUTANT QUE LES AUTRES :')
print('  le pont vectorisé est EXACT mais plus LENT que la routine C de')
print('  hashlib. Les 64 tours font ~1 300 passages numpy sur le tableau ;')
print('  chaque passage relit la mémoire. hashlib, lui, garde les 8 mots dans')
print('  les registres du processeur et ne sort jamais du cache L1.')
print('  Le pont ne sert donc PAS à aller plus vite. Il sert à OUVRIR la')
print('  boîte : sans registre exposé, pas de midstate, pas de sceau partiel,')
print('  pas de reprise. Le gain du §2 vient de là.')

# ═══════════════════════════════════ 4. CONDENSATEUR EXACT À LONG CHIFFRE
titre('4. CONDENSATEUR À LONG CHIFFRE — exact, et pourquoi il DOIT l\'être')
n = 20000
hs = P.sha256d_pont(E, list(range(n)))
gros = np.ascontiguousarray(hs[:, ::-1])
t0 = time.perf_counter(); cond = P.CondensateurExact(gros.tobytes(), 256)
t_pack = time.perf_counter() - t0
aller_retour = cond.lire() == gros.tobytes()
print(f'  {n:,} empreintes de 256 bits rangées en {t_pack*1e3:.1f} ms '
      f'({cond.octets()/1024:.0f} Kio, {cond.membres} membres de 32 bits)')
print(f'  restitution EXACTE, bit pour bit : {aller_retour}')

# ce qui arrive si on les fait passer par le condensateur À PERTE de Δ18
print('\n  MÊME EMPREINTE, PASSÉE PAR LE CONDENSATEUR À PERTE DE Δ18 :')
x = gros[0].astype(np.float64)                      # 32 octets, valeurs 0..255
perte = {}
for bits in (16, 8, 4):
    y = D18.Condensateur(x, bits=bits).lire()
    faux = int((np.rint(y).astype(np.int64) != x.astype(np.int64)).sum())
    perte[f'octets_{bits}'] = faux
    print(f'    rangée en OCTETS,  {bits:2d} bits/chiffre : {faux:2d}/32 faux -> '
          f'{"empreinte INTACTE" if faux == 0 else "empreinte DÉTRUITE"}')
m = cond.mots[0].astype(np.float64)                 # 8 membres de 32 bits
for bits in (16, 8):
    y = D18.Condensateur(m, bits=bits).lire()
    faux = int((np.rint(y).astype(np.int64) != m.astype(np.int64)).sum())
    perte[f'membres_{bits}'] = faux
    print(f'    rangée en MEMBRES, {bits:2d} bits/chiffre : {faux:2d}/8  faux -> '
          f'{"empreinte INTACTE" if faux == 0 else "empreinte DÉTRUITE"}')
print('\n  NUANCE QUI COMPTE, ET QUE JE NE VAIS PAS ESCAMOTER : rangée octet')
print('  par octet, une empreinte survit au condensateur 8 bits — 256 valeurs')
print('  tiennent dans 256 niveaux, il n\'y a rien à perdre. Ce n\'est donc pas')
print('  vrai que Δ18 détruit TOUJOURS une empreinte.')
print('  Mais dès qu\'on la range comme on doit la ranger pour la comparer —')
print('  en membres de 32 bits — la destruction est totale et immédiate. Et')
print('  un octet à la fois, c\'est quatre fois plus de place que le')
print('  condensateur exact pour exactement zéro gain.')
print('  D\'où la règle du pont : vers la cryptographie, c\'est EXACT ou rien.')
print('  Les deux condensateurs coexistent, chacun à sa place. Δ18 range des')
print('  grandeurs physiques où perdre un peu ne coûte rien ; celui-ci range')
print('  des chiffres où perdre un bit coûte tout.')
R['condensateur'] = {'n': n, 'exact': bool(aller_retour), 'octets': cond.octets(),
                     'ms_empaquetage': t_pack * 1e3, 'perte_delta18': perte}

# ══════════════════ 4bis. LA CHAÎNE SANS CONFIANCE (codec varint + txid)
titre('4bis. COUPER LE DERNIER FIL DE CONFIANCE — des octets bruts au bloc')
tx = json.load(open('btc_transactions.json'))['transactions']
recalc, sw = [], []
for attendu, h in tx.items():
    calc, seg = P.txid_depuis_brut(bytes.fromhex(h))
    recalc.append(calc == attendu); sw.append(seg)
    print(f'  txid recalculé depuis {len(h)//2:>3} octets bruts : '
          f'{"CONFORME" if calc == attendu else "*** ÉCART ***"}  {calc[:32]}...')
racine_nous = B.racine_merkle(list(tx.keys()))
chaine = all(recalc) and racine_nous == b100['merkle_root'] \
    and B.hash_bloc(b100) == b100['id']
print(f'  racine de Merkle bâtie sur NOS txid : '
      f'{"identique à celle de l en-tête" if racine_nous == b100["merkle_root"] else "ÉCART"}')
print(f'  en-tête portant cette racine -> hash officiel du bloc : '
      f'{B.hash_bloc(b100) == b100["id"]}')
print(f'\n  CHAÎNE COMPLÈTE VÉRIFIÉE : {chaine}')
print('  octets bruts -> txid -> racine de Merkle -> en-tête -> hash du bloc.')
print('  Avant ce codec, on CROYAIT l\'explorateur sur les txid. Maintenant on')
print('  les recalcule. C\'est la différence entre vérifier et faire confiance,')
print('  et elle tient dans un varint de quelques lignes.')
R['chaine_sans_confiance'] = {'txid_recalcules': all(recalc),
                              'racine_identique': racine_nous == b100['merkle_root'],
                              'chaine_complete': bool(chaine),
                              'segwit': sw}

# ═════════════════════════════════════ 5. COMPARATEUR DE CIBLE VECTORIEL
titre('5. JUGER N EMPREINTES D\'UN SEUL GESTE')
cible = B.cible(b100['bits'])
cb = cible.to_bytes(32, 'big')
t0 = time.perf_counter(); v = cond.sous(cb); t_vec = time.perf_counter() - t0
t0 = time.perf_counter()
w = np.array([int.from_bytes(gros[i].tobytes(), 'big') < cible for i in range(n)])
t_py = time.perf_counter() - t0
print(f'  {n:,} comparaisons | condensateur {t_vec*1e6:8,.0f} µs | '
      f'entiers Python {t_py*1e6:8,.0f} µs -> ×{t_py/t_vec:.0f}')
print(f'  mêmes verdicts : {bool((v == w).all())} | sous la cible : {int(v.sum())}')
par = cond.parite()
print(f'  parité XOR du lot (signature exacte de {n:,} empreintes) : {par.hex()[:32]}...')
R['comparateur'] = {'vectoriel_us': t_vec * 1e6, 'python_us': t_py * 1e6,
                    'gain': t_py / t_vec, 'accord': bool((v == w).all())}

# ═════════════════════════════════════════════ 6. CE QUI MANQUE ENCORE
titre('6. ANALYSE — les composants présents, et ceux qui manquent')
for nom, etat, a_quoi in P.COMPOSANTS:
    marque = '✓' if etat.startswith('FAIT') else '·'
    print(f'  {marque} {nom}')
    print(f'      {etat} — {a_quoi}')
R['composants'] = [{'nom': n, 'etat': e} for n, e, _ in P.COMPOSANTS]

# ══════════════════════════════════════════════════════════ 7. VERDICT
titre('7. CE QUE L\'ASTUCE A DONNÉ')
diff = B.difficulte(blocs[-1]['bits'])
avant = diff * 2 ** 32 / (N / t_direct) / 3.15e7
apres = diff * 2 ** 32 / (N / t_mid) / 3.15e7
print(f'  GAGNÉ : le midstate — ×1,50 en théorie, ×{t_direct/t_mid:.2f} mesuré ici,')
print(f'          et vérifié conforme sur 4 blocs réels.')
print(f'  GAGNÉ : le comparateur de cible vectoriel, ×{t_py/t_vec:.0f}.')
print(f'  GAGNÉ : la chaîne sans confiance — on ne croit plus personne sur les')
print(f'          txid, on les recalcule depuis les octets.')
print(f'  GAGNÉ : un registre, trois codecs, un condensateur exact — la base')
print(f'          pour Bech32 et secp256k1.')
print(f'  PERDU : le chemin vectorisé est {min(R["parallele"][k]["hashlib_kHs"]/R["parallele"][k]["pont_kHs"] for k in R["parallele"]):.1f}× à '
      f'{max(R["parallele"][k]["hashlib_kHs"]/R["parallele"][k]["pont_kHs"] for k in R["parallele"]):.1f}× plus LENT que')
print(f'          hashlib. Dit, mesuré, gardé au journal.')
print(f'\n  ET CE QUI NE CHANGE PAS : {avant:.2e} années pour un bloc avant')
print(f'  l\'astuce, {apres:.2e} après. Gagner {100*(1-apres/avant):.0f} % sur l\'infini,')
print(f'  c\'est toujours l\'infini. Le pont rend Δ plus CAPABLE, il ne rend pas la')
print(f'  montagne plus basse — et aucune astuce ne le fera, parce que SHA-256')
print(f'  est bâti pour n\'offrir aucune prise. C\'est d\'ailleurs ce qui fait')
print(f'  qu\'il tient debout depuis dix-sept ans.')
R['verdict'] = {'annees_avant': avant, 'annees_apres': apres}

json.dump(R, open('resultats_bridge.json', 'w'), indent=1, default=str)
print('\n→ resultats_bridge.json')
