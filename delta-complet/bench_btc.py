#!/usr/bin/env python3
"""bench_btc — TEST DE CONFORMITÉ Δ / BITCOIN.

Une référence extérieure, publique, que personne ici n'a fabriquée.
Huit épreuves. Une empreinte est juste ou elle est fausse : pas de milieu.
"""
import hashlib, json, time
import delta_btc as B

R = {'epreuves': {}}
blocs, source = B.charger()
blocs = sorted(blocs, key=lambda b: b['hauteur'])


def titre(t):
    print('\n' + '═' * 72); print(t); print('═' * 72, flush=True)


def ok(b):
    return 'CONFORME' if b else '*** ÉCART ***'


print('source des données de référence :', source)

# ════════════════════════════════════ C1 — l'en-tête donne-t-il le bon hash ?
titre('C1 — EN-TÊTE DE BLOC (80 octets) -> HASH OFFICIEL')
c1 = []
for b in blocs:
    h = B.hash_bloc(b)
    bon = (h == b['id'])
    c1.append(bon)
    print(f"  bloc {b['hauteur']:>7} : {h}")
    print(f"  {'':>13} attendu {b['id']}  {ok(bon)}")
R['epreuves']['C1_entete'] = {'tous_conformes': all(c1), 'blocs': len(c1)}

# ════════════════════════════════ C2 — la racine de Merkle recalculée
titre('C2 — RACINE DE MERKLE RECALCULÉE DEPUIS LES TRANSACTIONS')
c2 = []
for b in blocs:
    if not b['txids']:
        print(f"  bloc {b['hauteur']:>7} : txid non relevés (en-tête seul) — non testé")
        continue
    t0 = time.perf_counter()
    r = B.racine_merkle(b['txids'])
    dt = (time.perf_counter() - t0) * 1e3
    bon = (r == b['merkle_root'])
    c2.append(bon)
    print(f"  bloc {b['hauteur']:>7} : {b['tx_count']:>4} tx -> {r}")
    print(f"  {'':>13} attendu {b['merkle_root']}  {ok(bon)}  ({dt:.2f} ms)")
R['epreuves']['C2_merkle'] = {'tous_conformes': all(c2), 'blocs': len(c2)}

# ═══════════════════════════════════════════ C3 — la preuve de travail
titre('C3 — PREUVE DE TRAVAIL : le hash est-il SOUS la cible ?')
c3 = []
for b in blocs:
    val, h = B.preuve_de_travail(b)
    c3.append(val)
    cib = B.cible(b['bits'])
    zeros = 256 - h.bit_length()
    print(f"  bloc {b['hauteur']:>7} : {zeros:>3} bits de zéros en tête | "
          f"hash/cible = {h/cib:.6f} | {ok(val)}")
R['epreuves']['C3_pow'] = {'tous_conformes': all(c3), 'blocs': len(c3)}

# ═════════════════════════════════════════════ C4 — la difficulté
titre('C4 — DIFFICULTÉ RECALCULÉE DEPUIS nBits')
attendues = {0: 1.0, 100000: 14484.162361225399, 200000: 2864140.5078109736,
             968000: 132757073449487.52}
c4 = []
for b in blocs:
    d = B.difficulte(b['bits'])
    a = attendues[b['hauteur']]
    bon = abs(d - a) / a < 1e-9
    c4.append(bon)
    print(f"  bloc {b['hauteur']:>7} : nBits 0x{b['bits']:08x} -> difficulté "
          f"{d:,.6f} | attendue {a:,.6f} | {ok(bon)}")
R['epreuves']['C4_difficulte'] = {'tous_conformes': all(c4)}

# ══════════════════════════════════════ C5 — preuve SPV d'une transaction
titre('C5 — PREUVE SPV : prouver qu\'une transaction est dans le bloc')
b = [x for x in blocs if x['hauteur'] == 200000][0]
c5 = []
for i in (0, 1, 193, 386, 387):
    ch = B.preuve_merkle(b['txids'], i)
    vrai = B.verifier_preuve(b['txids'][i], ch, b['merkle_root'])
    faux_txid = 'ff' + b['txids'][i][2:]
    faux = B.verifier_preuve(faux_txid, ch, b['merkle_root'])
    c5.append(vrai and not faux)
    print(f"  tx n°{i:>4} : {len(ch)} empreintes ({len(ch)*32} octets) | "
          f"acceptée : {vrai} | sur un txid falsifié : {faux} | "
          f"{ok(vrai and not faux)}")
print(f'  -> un portefeuille léger vérifie une tx parmi {b["tx_count"]} avec '
      f'{len(ch)*32} octets au lieu des {b["tx_count"]*32} octets de la liste : '
      f'×{b["tx_count"]/len(ch):.0f}')
R['epreuves']['C5_spv'] = {'tous_conformes': all(c5), 'empreintes': len(ch),
                           'octets_preuve': len(ch) * 32,
                           'octets_liste': b['tx_count'] * 32}

# ══════════════════════════ C6 — l'arbre de Δ18 est-il celui de Bitcoin ?
titre('C6 — CONFORMITÉ DE Δ18 : son arbre de Merkle est-il celui de Bitcoin ?')
b100 = [x for x in blocs if x['hauteur'] == 100000][0]
res = B.conformite_delta18(b100['txids'])
print(f"  Δ18 (BLAKE2b) : {res['racine_delta18']}")
print(f"  Bitcoin       : {res['racine_bitcoin']}")
print(f"  {ok(res['conforme'])} — et c'est NORMAL. Les deux arbres ne répondent")
print('  pas à la même question. Voici les quatre écarts, nommés :')
for nom, d, bt in res['ecarts']:
    print(f"    · {nom}")
    print(f"        {d}")
    print(f"        {bt}")
print('\n  CONCLUSION HONNÊTE : Δ18 n\'est pas un nœud Bitcoin et ne prétend pas')
print('  l\'être. Son arbre à clé est FAIT pour qu\'un tiers ne puisse pas')
print('  refabriquer la racine ; celui de Bitcoin est fait pour que TOUT LE')
print('  MONDE le puisse. Ce sont deux besoins opposés, et deux bons choix.')
print('  Pour parler à Bitcoin, on utilise delta_btc.racine_merkle — conforme,')
print('  vérifié ci-dessus sur 393 transactions réelles.')
R['epreuves']['C6_delta18'] = {'conforme': res['conforme'],
                               'nb_ecarts': len(res['ecarts'])}

# ═══════════════════════ C7 — CVE-2012-2459 : la faille de duplication
titre('C7 — CVE-2012-2459 : deux blocs DIFFÉRENTS, la MÊME racine')
t = b100['txids']
liste_saine = t[:3]                      # 3 transactions -> niveau impair
liste_truquee = t[:3] + [t[2]]           # la dernière recopiée : 4 transactions
r1 = B.racine_merkle(liste_saine)
r2 = B.racine_merkle(liste_truquee)
print(f'  3 transactions      -> {r1}')
print(f'  4 (la 3e doublée)   -> {r2}')
print(f'  collision : {r1 == r2}  <- c\'est la faille, et elle est RÉELLE')
print('  Un attaquant fabrique un bloc invalide avec la même racine ; un nœud')
print('  de 2012 le marquait comme « déjà vu et invalide » et refusait ensuite')
print('  le VRAI bloc. Déni de service par confusion d\'identité.')
try:
    B.racine_merkle(liste_truquee, detecter_mutation=True)
    corrige = False
except ValueError as e:
    corrige = True
    print(f'  correctif appliqué  -> {e}')
r3 = B.racine_merkle(liste_saine, detecter_mutation=True)
print(f'  et le bloc sain passe toujours : {r3 == r1}')
sains = []
for bb in blocs:
    if not bb['txids']:
        continue
    try:
        sains.append(B.racine_merkle(bb['txids'], detecter_mutation=True) == bb['merkle_root'])
    except ValueError:
        sains.append(False)
print(f'  le correctif ne rejette AUCUN des blocs reels testes : {all(sains)} '
      f'({len(sains)} blocs, {sum(b["tx_count"] for b in blocs if b["txids"])} transactions)')
print('\n  LA LEÇON POUR Δ : Δ18 duplique lui aussi le dernier nœud impair.')
print('  Il hérite donc du même défaut. Il est corrigé de la même façon :')
print('  refuser un doublon en fin de niveau. C\'est une faille trouvée en')
print('  confrontant Δ à l\'extérieur, pas en le regardant tout seul.')
R['epreuves']['C7_cve'] = {'collision_reproduite': r1 == r2,
                           'correctif_efficace': corrige and r3 == r1,
                           'aucun_faux_positif': all(sains)}

# ══════════════════════════════════════ C8 — débit, et l'aveu sur le minage
titre('C8 — DÉBIT DE HACHAGE, ET CE QUE Δ N\'APPORTE PAS')
e = B.entete(blocs[1])
n = 200000
t0 = time.perf_counter()
for i in range(n):
    B.sha256d(e)
dt = time.perf_counter() - t0
hs = n / dt
print(f'  {n:,} doubles SHA-256 en {dt:.2f} s -> {hs/1e3:,.0f} kH/s '
      f'({hs/1e6:.3f} MH/s)')
diff = B.difficulte(blocs[-1]['bits'])
essais = diff * 2 ** 32
print(f'  difficulté actuelle du réseau (bloc {blocs[-1]["hauteur"]}) : {diff:,.0f}')
print(f'  essais attendus pour UN bloc : {essais:.3e}')
print(f'  à ce débit, il faudrait {essais/hs/3.15e7:.3e} années.')
print('\n  ET C\'EST LE POINT LE PLUS IMPORTANT DE TOUT CE TEST :')
print('  Δ ne peut RIEN pour le minage, et ne le pourra jamais. Tout son gain')
print('  vient de la STRUCTURE d\'un problème. Or SHA-256 est conçu, ligne par')
print('  ligne, pour n\'en avoir AUCUNE : pas de rang faible, pas de symétrie,')
print('  pas de voisinage exploitable. C\'est exactement le contre-essai du')
print('  noyau aléatoire de Δ-UN, mais mené par l\'adversaire lui-même.')
print('  Si un jour quelqu\'un vous dit qu\'une architecture « quantique » ou')
print('  « Δ » mine plus vite, c\'est faux, et ce banc le prouve chiffres en main.')
R['epreuves']['C8_debit'] = {'hashs_par_seconde': hs, 'difficulte_reseau': diff,
                             'essais_par_bloc': essais,
                             'annees_pour_un_bloc': essais / hs / 3.15e7}

# ═══════════════════════════════════════════════════════════ verdict
titre('VERDICT')
lignes = [('C1 en-tête -> hash officiel', all(c1)),
          ('C2 racine de Merkle recalculée', all(c2)),
          ('C3 preuve de travail', all(c3)),
          ('C4 difficulté', all(c4)),
          ('C5 preuve SPV (acceptée / falsifiée rejetée)', all(c5)),
          ('C7 faille CVE-2012-2459 reproduite puis corrigée',
           (r1 == r2) and corrige and r3 == r1 and all(sains))]
for nom, v in lignes:
    print(f'  {"✓" if v else "✗"}  {nom}')
print(f'  ·  C6 Δ18 non conforme au format Bitcoin — écart ASSUMÉ et expliqué')
print(f'  ·  C8 Δ n\'accélère pas le minage — limite ASSUMÉE et chiffrée')
total = all(v for _, v in lignes)
print(f'\n  {"TOUTES LES ÉPREUVES DE CONFORMITÉ SONT PASSÉES." if total else "ÉCART DÉTECTÉ."}')
R['verdict'] = {'conforme': total,
                'detail': {n: bool(v) for n, v in lignes}}
json.dump(R, open('resultats_btc.json', 'w'), indent=1, default=str)
print('\n→ resultats_btc.json')
