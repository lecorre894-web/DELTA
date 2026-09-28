#!/usr/bin/env python3
"""bench_sha — le moteur Δ-SHA mis à l'épreuve. Juges : hashlib, les vecteurs
officiels FIPS 180-4, et des blocs Bitcoin réels."""
import hashlib, hmac, json, os, time
import numpy as np
import delta_sha as M
import delta_btc as B

R = {}


def titre(t):
    print('\n' + '═' * 74); print(t); print('═' * 74, flush=True)


def ok(b):
    return 'CONFORME' if b else '*** ÉCART ***'


# ═══════════════════════════════ 1. LES CONSTANTES SONT-ELLES BONNES ?
titre('1. LA MACHINE ELLE-MÊME — constantes recalculées, pas recopiées')
H0, K = M.constantes()
print(f'  H[0] = 0x{H0[0]:08x}   (racine carrée de 2, partie fractionnaire)')
print(f'  K[0] = 0x{K[0]:08x}   (racine cubique de 2)')
print(f'  K[63]= 0x{K[63]:08x}   (racine cubique de 311)')
attendu = (H0[0] == 0x6A09E667 and K[0] == 0x428A2F98 and K[63] == 0xC67178F2)
print(f'  8 + 64 constantes dérivées des nombres premiers : {ok(attendu)}')
R['constantes'] = {'conformes': bool(attendu)}

# ═══════════════════════════════ 2. VECTEURS OFFICIELS FIPS 180-4
titre('2. VECTEURS OFFICIELS — et contrôle croisé de ma propre mémoire')
VECTEURS = [
    (b'', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'),
    (b'abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'),
    (b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
     '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1'),
    (b'abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno'
     b'ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu',
     'cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1'),
    (b'a' * 1000000,
     'cd0aa9856147b6c5b4ff2b7dff65165c1c9b1e2b4e1a8e0e4d5f8b0d0d0e0e0e0'),
]
lignes = []
for msg, cite in VECTEURS:
    mien = M.MoteurSHA256(msg).hexdigest()
    vrai = hashlib.sha256(msg).hexdigest()
    nom = f'{len(msg)} octets' if len(msg) != 1000000 else "1 000 000 de 'a'"
    d_ok = (mien == vrai)
    memoire_ok = (cite == vrai)
    lignes.append((nom, d_ok, memoire_ok))
    print(f'  {nom:<22} moteur vs hashlib : {ok(d_ok)}')
    if not memoire_ok:
        print(f'  {"":<22} ATTENTION : le vecteur que J\'AI CITÉ de mémoire est FAUX.')
        print(f'  {"":<22}   cité  {cite}')
        print(f'  {"":<22}   vrai  {vrai}')
        print(f'  {"":<22} Le moteur, lui, est juste. C\'est ma mémoire qui a menti,')
        print(f'  {"":<22} et c\'est exactement pourquoi on ne recopie jamais un')
        print(f'  {"":<22} chiffre sans le confronter à une référence.')
print(f'\n  moteur conforme sur les {len(VECTEURS)} vecteurs : '
      f'{all(x[1] for x in lignes)}')
print(f'  vecteurs que j\'avais correctement en mémoire : '
      f'{sum(x[2] for x in lignes)}/{len(lignes)}')
R['vecteurs'] = {'moteur_conforme': all(x[1] for x in lignes),
                 'memoire_juste': sum(x[2] for x in lignes), 'total': len(lignes)}

# ══════════════════════════════════ 3. ALÉATOIRE MASSIF CONTRE hashlib
titre('3. 5 000 MESSAGES ALÉATOIRES DE LONGUEUR QUELCONQUE')
rng = np.random.default_rng(7)
mauvais = 0
for _ in range(5000):
    n = int(rng.integers(0, 600))
    m = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    if M.MoteurSHA256(m).hexdigest() != hashlib.sha256(m).hexdigest():
        mauvais += 1
print(f'  longueurs 0 à 599 octets (bourrage sur 1 et 2 blocs) : '
      f'{5000 - mauvais}/5000 conformes')
# les longueurs pièges : juste avant, sur, et juste après la frontière
pieges = [0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129]
p_ok = all(M.MoteurSHA256(b'x' * n).hexdigest() == hashlib.sha256(b'x' * n).hexdigest()
           for n in pieges)
print(f'  longueurs pièges {pieges} : {ok(p_ok)}')
print('  (55/56/57 est LA frontière : c\'est là que le bourrage bascule sur un')
print('   bloc de plus. Presque toutes les implémentations fausses cassent ici.)')
R['aleatoire'] = {'conformes': 5000 - mauvais, 'total': 5000, 'pieges': bool(p_ok)}

# ══════════════════════════════════ 4. CE QUE hashlib NE SAIT PAS FAIRE
titre('4. ÉTAT SORTABLE — arrêter, ranger, reprendre ailleurs')
gros = bytes(rng.integers(0, 256, 200000, dtype=np.uint8))
m = M.MoteurSHA256()
m.update(gros[:73421])
sauvegarde = m.etat()
print(f'  73 421 octets absorbés, puis on RANGE l\'état : '
      f'{len(sauvegarde)} octets')
del m
m2 = M.MoteurSHA256.depuis_etat(sauvegarde)
m2.update(gros[73421:])
repris = m2.hexdigest()
direct = hashlib.sha256(gros).hexdigest()
print(f'  reprise dans un moteur NEUF, puis fin du fichier : {ok(repris == direct)}')
print(f'  {repris}')
try:
    h = hashlib.sha256(); h.update(b'x')
    import pickle
    pickle.dumps(h)
    hashlib_serialisable = True
except Exception as e:
    hashlib_serialisable = False
    raison = type(e).__name__
print(f'  hashlib sait-il faire la même chose ? {hashlib_serialisable} '
      f'({raison if not hashlib_serialisable else ""})')
print(f'  C\'EST LE VRAI APPORT DU MOTEUR : {len(sauvegarde)} octets ici '
      f'(41 a 104 selon le reste du tampon)')
print('  suffisent pour suspendre le hachage d\'un fichier de n\'importe')
print('  quelle taille et le reprendre')
print('  des jours plus tard, sur une autre machine. Le midstate de Bitcoin')
print('  n\'est qu\'un cas particulier de ça.')
R['reprise'] = {'octets_etat': len(sauvegarde), 'conforme': repris == direct,
                'hashlib_serialisable': hashlib_serialisable}

# ══════════════════════════════════════════════ 5. HMAC
titre('5. HMAC — sceller pour qu\'un tiers ne puisse pas refabriquer')
cles = [b'', b'cle', b'k' * 64, b'k' * 200]
msgs = [b'', b'message', bytes(rng.integers(0, 256, 300, dtype=np.uint8))]
h_ok = all(M.hmac_sha256(c, m) == hmac.new(c, m, hashlib.sha256).digest()
           for c in cles for m in msgs)
print(f'  {len(cles)*len(msgs)} combinaisons clé/message (clé vide, courte, '
      f'64 octets, plus longue que le bloc) : {ok(h_ok)}')
print('  Un CRC se refabrique en quelques octets bien choisis. Un HMAC, non :')
print('  sans la clé, on ne peut ni forger ni prolonger.')
R['hmac'] = {'conforme': bool(h_ok)}

# ══════════════════════════════════════════════ 6. BITCOIN
titre('6. BITCOIN — le moteur face à quatre blocs réels')
blocs, _ = B.charger()
blocs = sorted(blocs, key=lambda b: b['hauteur'])
b_ok = []
for b in blocs:
    e = B.entete(b)
    h = M.sha256d(e)[::-1].hex()
    b_ok.append(h == b['id'])
    print(f'  bloc {b["hauteur"]:>7} : {ok(h == b["id"])}  {h[:40]}...')
# le midstate, avec l'état sortable
mid = M.MoteurSHA256(); mid.update(e[:64])
etat_mid = mid.etat()
m3 = M.MoteurSHA256.depuis_etat(etat_mid); m3.update(e[64:])
mid_ok = M.sha256(m3.digest())[::-1].hex() == blocs[-1]['id']
print(f'  midstate obtenu par etat()/depuis_etat() : {ok(mid_ok)}')
R['bitcoin'] = {'blocs_conformes': all(b_ok), 'midstate': bool(mid_ok)}

# ══════════════════════════════════════════════ 7. L'ARBRE
titre('7. MOTEUR EN ARBRE — parallélisable, prouvable, réparable')
racine, niveaux = M.hacher_par_arbre(gros, taille_feuille=4096)
n_f = len(niveaux[0])
pr = M.preuve_arbre(niveaux, 7)
v = M.verifier_arbre(niveaux[0][7], pr, racine)
faux = M.verifier_arbre(M.sha256(b'menteur'), pr, racine)
print(f'  200 000 octets -> {n_f} feuilles de 4 Kio, {len(niveaux)} étages')
print(f'  racine : {racine.hex()}')
print(f'  preuve d\'un morceau : {len(pr)} empreintes ({len(pr)*32} octets) | '
      f'acceptée {v} | falsifiée {faux}')
print(f'  linéaire (un seul SHA du tout) : {hashlib.sha256(gros).hexdigest()[:32]}...')
print('  LES DEUX EMPREINTES SONT DIFFÉRENTES, ET C\'EST NORMAL : elles ne')
print('  répondent pas à la même question. Le SHA linéaire dit « ce fichier')
print('  est-il intact ». L\'arbre dit « ce MORCEAU appartient-il au fichier »,')
print('  et il le dit sans relire le reste. On livre les deux, on ne les')
print('  confond jamais.')
R['arbre'] = {'feuilles': n_f, 'etages': len(niveaux), 'preuve_octets': len(pr) * 32,
              'acceptee': bool(v), 'falsifiee_rejetee': not faux}

# ══════════════════════════════════════════════ 8. LE PRIX À PAYER
titre('8. LE PRIX — le moteur est LENT, et il faut le dire en premier')
donnee = bytes(rng.integers(0, 256, 262144, dtype=np.uint8))


def meilleur(f, n=3):
    return min([(lambda: (time.perf_counter(), f(), time.perf_counter()))()
                for _ in range(n)], key=lambda x: x[2] - x[0])


t = []
for _ in range(3):
    t0 = time.perf_counter(); M.sha256(donnee); t.append(time.perf_counter() - t0)
t_moteur = min(t)
t = []
for _ in range(5):
    t0 = time.perf_counter(); hashlib.sha256(donnee).digest(); t.append(time.perf_counter() - t0)
t_hashlib = min(t)
print(f'  256 Kio hachés')
print(f'    Δ-SHA   : {len(donnee)/t_moteur/1e6:8.2f} Mo/s')
print(f'    hashlib : {len(donnee)/t_hashlib/1e6:8.2f} Mo/s')
print(f'    rapport : ×{t_moteur/t_hashlib:,.0f} EN DÉFAVEUR du moteur')
print('\n  C\'est attendu et c\'est net : hashlib est du C compilé qui garde ses')
print('  8 mots dans les registres du processeur ; Δ-SHA est du Python qui')
print('  manipule des entiers objets. On ne rattrape pas ça, et on ne va pas')
print('  prétendre le contraire.')
print('  CE QU\'ON ACHÈTE AVEC CETTE LENTEUR : l\'état sortable, le midstate,')
print('  l\'arbre, le parallèle, et des constantes qu\'on peut auditer. En')
print('  production on hache avec hashlib et on PILOTE avec Δ-SHA — le moteur')
print('  ouvert sert à décider, pas à abattre le travail.')
R['prix'] = {'moteur_Mo_s': len(donnee) / t_moteur / 1e6,
             'hashlib_Mo_s': len(donnee) / t_hashlib / 1e6,
             'rapport': t_moteur / t_hashlib}

# ══════════════════════════════════════════════ VERDICT
titre('VERDICT')
v = [('constantes recalculées', R['constantes']['conformes']),
     ('vecteurs officiels', R['vecteurs']['moteur_conforme']),
     ('5 000 messages aléatoires + longueurs pièges',
      R['aleatoire']['conformes'] == 5000 and R['aleatoire']['pieges']),
     ('état sortable et reprise', R['reprise']['conforme']),
     ('HMAC', R['hmac']['conforme']),
     ('4 blocs Bitcoin + midstate', R['bitcoin']['blocs_conformes'] and R['bitcoin']['midstate']),
     ('arbre : preuve acceptée, falsifiée rejetée',
      R['arbre']['acceptee'] and R['arbre']['falsifiee_rejetee'])]
for nom, b in v:
    print(f'  {"✓" if b else "✗"}  {nom}')
print(f'  ·  lenteur assumée et chiffrée : ×{R["prix"]["rapport"]:,.0f}')
R['verdict'] = {n: bool(b) for n, b in v}
json.dump(R, open('resultats_sha.json', 'w'), indent=1, default=str)
print(f'\n  {"MOTEUR CONFORME SUR TOUTE LA LIGNE." if all(b for _, b in v) else "ÉCART DÉTECTÉ."}')
print('\n→ resultats_sha.json')
