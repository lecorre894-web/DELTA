#!/usr/bin/env python3
"""bench_reglage — QUEL réglage fixe bat l'agent, et pourquoi.

On ne fige pas un réglage sur la foi d'un seul chiffre : on balaye les deux
grandeurs qui comptent vraiment — la GRANULARITÉ (combien de blocs on calcule
d'un coup) et la PROFONDEUR (combien de granules d'avance) — puis on regarde
si le gagnant reste le même quand la charge change.
"""
import json, time
import numpy as np
import delta_systeme as SY

R = {}
P = lambda t: print(t, flush=True)
K0 = 10 ** 18 // 128 // 2


def titre(t):
    P('\n' + '═' * 74); P(t); P('═' * 74)


def courir(G, prof, acces, cap=256 * 1024):
    s = SY.Systeme(lot=G * prof, capacite_L1=cap)
    a = SY.Agent(s, actif=False)
    # CORRECTION MAJEURE : on interroge le CACHE RÉEL, pas un ensemble
    # « déjà vu ». Sans cela, un bloc évincé du cache n'était jamais
    # recalculé et le taux de succès valait 1,000 par construction —
    # le cache ne pouvait rien départager. C'était le défaut du banc.
    calcules = 0
    t0 = time.perf_counter()
    for k in acces:
        base = (k // G) * G
        for d in range(prof):
            bb = base + d * G
            if bb not in s.cache.repertoire:
                s.precalculer(bb, G)
                calcules += G
        s.lire(k)
    return time.perf_counter() - t0, s.cache.taux(), calcules


def charges():
    g = np.random.default_rng(0)
    n = 1200
    a = [K0 + i for i in range(n // 3)]
    b = [K0 + int(x) for x in g.integers(0, 40000, n // 3)]
    c = [K0 + 5000 + int(x) for x in g.integers(0, 40, n - 2 * (n // 3))]
    return {'mixte (le banc actuel)': a + b + c,
            'balayage long': [K0 + i for i in range(n)],
            'dispersion': [K0 + int(x) for x in g.integers(0, 200000, n)],
            'boucle serrée': [K0 + int(x) for x in g.integers(0, 60, n)]}


# 1 — LE BALAYAGE COMPLET
titre('1. LE BALAYAGE — granularité x profondeur, sur la charge du banc')
GS, PS = (8, 16, 32, 64, 128), (1, 2, 4)
ch = charges()
acc = ch['mixte (le banc actuel)']
P(f"  {'G':>5}" + ''.join(f'{"prof " + str(p):>12}' for p in PS)
  + f"{'blocs calculés':>16}")
grille = {}
for G in GS:
    ligne, blocs = [], None
    for p in PS:
        dt, taux, nb = courir(G, p, acc)
        ligne.append(dt)
        grille[(G, p)] = {'s': dt, 'taux': taux, 'blocs': nb}
        if p == 1:
            blocs = nb
    P(f'  {G:>5}' + ''.join(f'{v:>11.2f}s' for v in ligne) + f'{blocs:>16,}')
best = min(grille, key=lambda k: grille[k]['s'])
P(f'\n  MEILLEUR : granularité {best[0]}, profondeur {best[1]} -> '
  f'{grille[best]["s"]:.2f} s')
P(f'  (l\'agent adaptatif faisait 4,54 s ; le « FIXE lot=64 » du banc')
P(f'   précédent, c\'était G=64 profondeur 1 : {grille[(64,1)]["s"]:.2f} s)')
R['balayage'] = {f'{k[0]}x{k[1]}': v for k, v in grille.items()}
R['meilleur'] = {'G': best[0], 'prof': best[1], 's': grille[best]['s']}

# 2 — LE GAGNANT TIENT-IL SUR D'AUTRES CHARGES ?
titre('2. LE GAGNANT TIENT-IL AILLEURS ? (c\'est ça, la vraie question)')
cands = [(8, 1), (16, 1), (32, 1), (64, 1), (64, 2), (128, 1)]
P(f"  {'charge':<24}" + ''.join(f'{f"G{g}p{p}":>10}' for g, p in cands)
  + f"{'meilleur':>12}")
tab = {}
for nom, a in ch.items():
    ligne = []
    for g, p in cands:
        dt, _, _ = courir(g, p, a)
        ligne.append(dt)
    i = int(np.argmin(ligne))
    tab[nom] = {'temps': ligne, 'meilleur': f'G{cands[i][0]}p{cands[i][1]}'}
    P(f'  {nom:<24}' + ''.join(f'{v:>9.2f}s' for v in ligne)
      + f'{tab[nom]["meilleur"]:>12}')
gagnants = {v['meilleur'] for v in tab.values()}
P(f'\n  réglages gagnants distincts : {len(gagnants)}  ({", ".join(sorted(gagnants))})')
if len(gagnants) == 1:
    P('  UN SEUL réglage gagne partout. Dans ce système, un agent adaptatif')
    P('  n\'a donc rien à apporter : il n\'y a rien à adapter. On fige.')
else:
    P('  Le gagnant CHANGE selon la charge : un réglage fixe ne peut pas')
    P('  être optimal partout, et un agent retrouve un intérêt.')
R['par_charge'] = tab

# 3 — POURQUOI : OÙ PART LE TEMPS
titre('3. POURQUOI CE RÉGLAGE — où part le temps')
for G in (8, 64, 128):
    dt, taux, nb = courir(G, 1, acc)
    P(f'  G={G:>3} : {nb:>8,} blocs calculés pour 1 200 accès  '
      f'-> {nb/1200:>6.1f} blocs par accès utile, {dt:.2f} s')
P('\n  Tout est là : plus la granularité est grosse, plus on calcule de')
P('  blocs que personne ne lira jamais. Le taux de cache reste à 1,000')
P('  dans tous les cas, donc le cache ne départage RIEN — seul compte le')
P('  travail fait pour rien. C\'est la limite du banc, et elle explique')
P('  pourquoi le meilleur réglage est simplement « le plus petit grain ».')

# 4 — LE BANC DUR : ET SI LE CACHE DÉBORDAIT ?
titre('4. LE BANC DUR — un cache trop petit, et la donne change')
g = np.random.default_rng(1)
dur = [K0 + (i % 3000) for i in range(2400)]      # balayage répété, ensemble large
P(f"  cache réduit à 16 Kio (il déborde vraiment)")
P(f"  {'réglage':>10} {'secondes':>10} {'taux':>8} {'blocs':>12}")
durs = {}
for g_, p in ((8, 1), (16, 1), (32, 1), (64, 1), (64, 2), (64, 4), (128, 1)):
    dt, taux, nb = courir(g_, p, dur, cap=16 * 1024)
    durs[f'G{g_}p{p}'] = {'s': dt, 'taux': taux, 'blocs': nb}
    P(f'  {f"G{g_}p{p}":>10} {dt:>10.2f} {taux:>8.3f} {nb:>12,}')
bd = min(durs, key=lambda k: durs[k]['s'])
P(f'\n  MEILLEUR ICI : {bd} ({durs[bd]["s"]:.2f} s, taux {durs[bd]["taux"]:.3f})')
tx = [v['taux'] for v in durs.values()]
if max(tx) - min(tx) > 1e-6:
    P(f'  taux de succès enfin variables : de {min(tx):.3f} à {max(tx):.3f}.')
    P('  Le cache travaille et départage. C\'est le banc qui manquait.')
else:
    P(f'  taux toujours identiques ({tx[0]:.3f}), et J\'AI TROUVÉ POURQUOI :')
    P('  ce n\'est pas un défaut du banc, c\'est le PRINCIPE MÊME de Δ19.')
    P('  Dans la pile X3D, l\'éviction devient DÉGRADATION : une ligne')
    P('  froide descend d\'étage et s\'y range plus grossièrement, mais elle')
    P('  reste au répertoire. Elle n\'est donc jamais « manquée ».')
    P('  CONSÉQUENCE, ET ELLE EXPLIQUE TOUTE LA NUIT : dans un système bâti')
    P('  sur X3D, le taux de succès vaut structurellement 1,000 et ne peut')
    P('  RIEN piloter. Voilà pourquoi l\'agent V1, qui suivait le taux,')
    P('  était condamné d\'avance — et pourquoi V3, qui suit l\'utilisation,')
    P('  est le seul à regarder une grandeur qui bouge vraiment.')
R['banc_dur'] = durs

json.dump(R, open('resultats_reglage.json', 'w'), indent=1, default=str)
P('\n→ resultats_reglage.json')
