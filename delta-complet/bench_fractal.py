#!/usr/bin/env python3
"""bench_fractal — chaque grain porte le concept entier. Ce que ça donne,
ce que ça coûte, et où ça cesse d'être raisonnable."""
import hashlib, json, math, time
import numpy as np
import delta_fractal as F

R = {}
CLE = hashlib.blake2b(b'Delta-Rene', digest_size=32).digest()


def titre(t):
    print('\n' + '═' * 72); print(t); print('═' * 72, flush=True)


# ═══════════════════════════════════════ 1. LA RÉCURSION EST-ELLE JUSTE ?
titre('1. CONFORMITÉ — la récursion rend-elle la même valeur que le calcul direct ?')
b = 128
f = F.fournisseur_grain(b)
A = F.Fractal(1024, f, CLE, branchement=2)
direct = np.concatenate([f(k) for k in range(1024)])
par_larbre = A.racine.valeur()
err = float(np.abs(direct - par_larbre).max())
print(f'  1 024 grains de {b} nombres, arbre de profondeur {A.profondeur()}')
print(f'  écart maximal entre l\'arbre entier et le calcul direct : {err:.1e}')
print(f'  racine : {A.racine.sceau().hex()}')
print(f'  construction : {A.t_construction*1e3:.0f} ms')
R['conformite'] = {'ecart_max': err, 'profondeur': A.profondeur(),
                   'grains': 1024, 'ms': A.t_construction * 1e3}

# ═══════════════════════════════════════ 2. LA PREUVE, D'UN BOUT À L'AUTRE
titre('2. LA PREUVE TRAVERSE TOUS LES ÉTAGES')
bons, mauvais = [], []
for i in (0, 1, 511, 1000, 1023):
    ch = A.preuve(i)
    v = A.verifier(i, ch)
    faux = F.Noeud.verifier_chemin(hashlib.blake2b(b'menteur', key=CLE,
                                                   digest_size=16).digest(),
                                   ch, A.racine.sceau(), CLE)
    bons.append(v); mauvais.append(faux)
    print(f'  grain {i:>5} : {len(ch):2d} empreintes ({len(ch)*16:4d} octets) | '
          f'acceptée {v} | sur un sceau inventé {faux}')
print(f'  -> prouver l\'appartenance d\'un grain parmi 1 024 coûte '
      f'{A.taille_preuve(0)} octets ; relire les 1 024 grains en coûterait '
      f'{A.utile():,} : ×{A.utile()/A.taille_preuve(0):,.0f}')
R['preuve'] = {'toutes_vraies': all(bons), 'aucune_fausse': not any(mauvais),
               'octets': A.taille_preuve(0), 'octets_donnees': A.utile()}

# ════════════════════════════ 3. LE PRIX DU CONCEPT COMPLET SUR CHAQUE GRAIN
titre('3. À QUELLE FINESSE LE CONCEPT COMPLET TIENT-IL ENCORE ?')
print('  Chaque grain porte 16 octets de sceau, quoi qu\'il transporte.')
print('  Plus le grain est fin, plus cette signature pèse lourd devant lui.\n')
print(f"  {'grain':>8} {'utile':>12} {'total':>12} {'surcoût':>10}  verdict")
lignes = []
for taille in (1, 2, 4, 8, 16, 64, 256, 1024, 4096):
    T = F.Fractal(256, F.fournisseur_grain(taille), CLE, branchement=2)
    u, o = T.utile(), T.octets()
    s = (o - u) / u
    verdict = ('le concept écrase la donnée' if s > 1.0 else
               'coûteux' if s > 0.25 else
               'raisonnable' if s > 0.05 else 'quasi gratuit')
    print(f'  {taille:>8} {u:>12,} {o:>12,} {100*s:>9.1f}%  {verdict}')
    lignes.append({'grain': taille, 'utile': u, 'total': o, 'surcout': s})
print('\n  LECTURE : sous 16 nombres par grain, la signature et l\'ossature de')
print('  l\'arbre pèsent plus que ce qu\'elles protègent. Au-dessus de 256, le')
print('  concept complet devient quasi gratuit — mais on perd en finesse de')
print('  désignation : on ne peut plus montrer du doigt qu\'un gros morceau.')
print('  L\'infini ne se gagne donc pas en descendant toujours plus fin. Il se')
print('  gagne à la BONNE finesse, et cette finesse se mesure.')
R['grain'] = lignes

# ═══════════════════════════════ 4. LA PREUVE GRANDIT-ELLE ? À PEINE.
titre('4. LA PREUVE FACE AU NOMBRE — c\'est ici que l\'infini se gagne')
print(f"  {'grains':>12} {'profondeur':>11} {'preuve':>10} {'données':>14}  rapport")
croissance = []
for n in (16, 256, 4096, 65536):
    T = F.Fractal(n, F.fournisseur_grain(16), CLE, branchement=2)
    p, u = T.taille_preuve(0), T.utile()
    print(f'  {n:>12,} {T.profondeur():>11} {p:>7} o {u:>12,} o  ×{u/p:>10,.0f}')
    croissance.append({'grains': n, 'profondeur': T.profondeur(),
                       'preuve_octets': p, 'donnees_octets': u})
print('\n  La donnée est multipliée par 4 096 ; la preuve, elle, passe de 64 à')
print('  256 octets. Elle grandit comme le LOGARITHME, pas comme le nombre.')
print('  C\'est exactement ça, « le concept complet fera l\'infini » : ce n\'est')
print('  pas qu\'on range l\'infini, c\'est qu\'on le PROUVE en quelques octets.')
R['croissance'] = croissance

# ═════════════════════════════════════ 5. JUSQU'OÙ, VRAIMENT (extrapolation)
titre('5. JUSQU\'OÙ — et ce qui est mesuré, ce qui est extrapolé')
print(f"  {'grains':>26} {'preuve':>9}   statut")
extra = []
for e in (16, 20, 32, 64, 128, 256):
    n = 2 ** e
    p = e * 16
    statut = 'MESURÉ' if e <= 16 else 'extrapolé (log2(N)×16 o)'
    print(f'  2^{e:<3} = {n:>18,.0f} {p:>7} o   {statut}' if e <= 64
          else f'  2^{e:<3} = {float(n):>18.3e} {p:>7} o   {statut}')
    extra.append({'exposant': e, 'preuve_octets': p, 'mesure': e <= 16})
print('\n  2^64 grains, soit dix-huit milliards de milliards : la preuve tient')
print('  en UN kilo-octet. 2^256 : quatre kilo-octets.')
print('  HONNÊTEMENT : au-delà de 2^16 (65 536), ce tableau est une')
print('  EXTRAPOLATION de la loi log2(N)×16, pas une mesure. La loi est')
print('  vérifiée sur les quatre points mesurés du §4 ; elle n\'est pas')
print('  devinée. Mais matérialiser 2^64 grains n\'est possible sur aucune')
print('  machine, et je ne vais pas faire semblant du contraire.')
R['extrapolation'] = extra

# ═════════════════════════════════ 6. UN GRAIN CORROMPU EST-IL DÉSIGNÉ ?
titre('6. UN GRAIN MENT — l\'arbre le montre-t-il du doigt ?')
T = F.Fractal(512, F.fournisseur_grain(64), CLE, branchement=2)
avant = T.verifier(300)
sain = T.intact()
rien, _ = T.designer_fautif()
T.cellules[300].v[7] += 1e-9                    # une altération minuscule
t0 = time.time(); fautif, chemin = T.designer_fautif(); dt = time.time() - t0
apres = T.verifier(300)
print(f'  arbre sain : intact ? {sain} | un fautif désigné à tort ? '
      f'{rien is not None}')
print(f'  avant altération, le grain 300 se vérifie : {avant}')
print(f'  on modifie UN nombre de 1e-9 dans le grain 300 (le sceau, lui,')
print(f'  n\'est PAS retouché — c\'est le cas le plus sournois)')
print(f'  l\'arbre le voit : {not T.intact()}')
print(f'  et il désigne : grain k={fautif.k if fautif is not None else None} '
      f'(attendu 300), chemin de {len(chemin)} branches sur 512 grains')
print(f'  la preuve du grain 300 est maintenant REFUSÉE : {not apres}')
print(f'  trouvé en {dt*1e3:.1f} ms')
print('\n  CE DÉFAUT ÉTAIT RÉEL ET IL ÉTAIT À MOI : la vérification comparait')
print('  le sceau RANGÉ au lieu de le RECALCULER sur la donnée. Résultat :')
print('  on pouvait modifier un grain sans que rien ne bronche. Un sceau qui')
print('  ne relit pas ce qu\'il scelle ne protège rien — il confirme seulement')
print('  qu\'on l\'a bien rangé. Corrigé, puis remesuré ci-dessus.')
R['corruption'] = {'designe': (fautif.k if fautif is not None else None),
                   'attendu': 300, 'branches': len(chemin), 'grains': 512,
                   'ms': dt * 1e3, 'sain_avant': bool(sain),
                   'aucun_faux_positif': rien is None,
                   'preuve_refusee_apres': not apres}

# ══════════════════════════════════════════════════ 7. CONTRE-ESSAI
titre('7. CONTRE-ESSAI — et si deux grains sont identiques ?')
class Faux:
    def __init__(s, sc): s._sceau = sc
    def sceau(s): return s._sceau
    def poids(s): return 1
    def octets(s): return 16
    def utile(s): return 0
    def valeur(s): return np.zeros(1)
c = F.Cellule(0, F.fournisseur_grain(8), CLE)
try:
    F.Noeud([c, Faux(c.sceau())], CLE, strict=True)
    rejet = False
except ValueError as e:
    rejet = True
    print(f'  deux grains de sceau identique -> {e}')
print(f'  rejeté : {rejet}')
print('  Le correctif trouvé hier sur Bitcoin protège maintenant CHAQUE étage')
print('  de la fractale. Une faille corrigée une fois est corrigée partout :')
print('  c\'est le seul vrai bénéfice d\'une architecture qui se répète.')
R['contre_essai'] = {'mutation_rejetee': rejet}

# ═══════════════════════════════════════════════════════ 8. VERDICT
titre('8. CE QUE LA FRACTALE DONNE, ET CE QU\'ELLE NE DONNE PAS')
print('  DONNE  une preuve en log2(N) : 1 Kio pour 2^64 grains.')
print('  DONNE  la désignation du grain fautif : 9 branches suivies sur 512.')
print('         (détecter exige de tout relire — O(N), aucune structure n\'y')
print('          échappe — mais dire LEQUEL, au bout de cette même lecture,')
print('          ne coûte rien de plus. Un sceau global, lui, coûte autant')
print('          et ne donne que l\'alarme.)')
print('  DONNE  une seule correction qui protège tous les étages à la fois.')
print('  DONNE  la même interface à tous les niveaux : on peut empiler sans fin.')
print('  NE DONNE PAS  de la place : l\'arbre ne range rien de plus, il PROUVE.')
print('  NE DONNE PAS  de gain sous 16 nombres par grain : en dessous, le')
print('                concept complet pèse plus que ce qu\'il porte.')
print('  NE DONNE PAS  2^64 grains sur une machine réelle — la loi est')
print('                vérifiée, la matérialisation ne l\'est pas.')
json.dump(R, open('resultats_fractal.json', 'w'), indent=1, default=str)
print('\n→ resultats_fractal.json')
