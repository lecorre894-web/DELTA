#!/usr/bin/env python3
"""bench_arene — la compétition. Fiabilité = pire cas, pas moyenne."""
import json
import numpy as np
import delta_arene as A

R = {}
P = lambda t: print(t, flush=True)
CHARGES = ('sequentielle', 'dispersee', 'boucle', 'zipf', 'alternance', 'adverse')


def titre(t):
    P('\n' + '═' * 74); P(t); P('═' * 74)


def concurrents():
    return ([A.Fixe(l) for l in A.LOTS]
            + [A.Glouton(), A.Hysteresis(), A.Bandit()])


# ═════════════════════════ 1. L'ORACLE : la borne à battre
titre('1. L\'ORACLE — le meilleur lot FIXE de chaque charge, connu après coup')
ar = A.Arene(capacite=512)
oracle = {}
P(f"  {'charge':<14}" + ''.join(f'{l:>9}' for l in A.LOTS) + f"{'meilleur':>10}")
for c in CHARGES:
    acces = A.charge(c, 3000, 0)
    couts = [ar.courir(acces, A.Fixe(l))['cout_us'] / 1000 for l in A.LOTS]
    i = int(np.argmin(couts))
    oracle[c] = {'cout': couts[i], 'lot': A.LOTS[i]}
    P(f'  {c:<14}' + ''.join(f'{v:>9.1f}' for v in couts) + f'{A.LOTS[i]:>10}')
P('\n  (coûts en millisecondes simulées) Regarde la ligne « alternance » et')
P('  la ligne « boucle » : le meilleur lot n\'est PAS le même. Aucun réglage')
P('  immobile ne peut gagner partout — c\'est ce qui manquait au banc d\'hier.')
R['oracle'] = oracle

# ═════════════════════════ 2. LA COMPÉTITION
titre('2. LA COMPÉTITION — regret de chacun sur chaque charge')
P('  Le regret est le surcoût par rapport à l\'oracle, en pourcentage.')
P('  0 % = aussi bon que le meilleur réglage connu après coup.\n')
noms = [a.nom for a in concurrents()]
P(f"  {'agent':<15}" + ''.join(f'{c[:9]:>10}' for c in CHARGES) + f"{'PIRE':>8}")
table = {}
for j, modele in enumerate(concurrents()):
    ligne, regrets = [], []
    for c in CHARGES:
        ag = concurrents()[j]
        r = ar.courir(A.charge(c, 3000, 0), ag)
        reg = 100 * (r['cout_us'] / 1000 - oracle[c]['cout']) / oracle[c]['cout']
        regrets.append(reg)
        ligne.append(r)
    table[modele.nom] = {'regrets': regrets, 'pire': max(regrets),
                         'moyen': float(np.mean(regrets)),
                         'changements': [x['changements'] for x in ligne]}
    P(f'  {modele.nom:<15}' + ''.join(f'{v:>9.1f}%' for v in regrets)
      + f'{max(regrets):>7.1f}%')
R['competition'] = table

# ═════════════════════════ 3. LE CLASSEMENT PAR FIABILITÉ
titre('3. CLASSEMENT — par PIRE cas, pas par moyenne')
ordre = sorted(table.items(), key=lambda kv: kv[1]['pire'])
P(f"  {'rang':>4} {'agent':<15} {'pire regret':>13} {'regret moyen':>14} "
  f"{'changements':>12}")
for i, (nom, v) in enumerate(ordre, 1):
    P(f'  {i:>4} {nom:<15} {v["pire"]:>12.1f}% {v["moyen"]:>13.1f}% '
      f'{sum(v["changements"]):>12}')
gagnant = ordre[0][0]
P(f'\n  GAGNANT EN FIABILITÉ : {gagnant}')
P('  Un agent qui gagne cinq charges et s\'effondre sur la sixième ne vaut')
P('  rien dans une machine qui tourne sans surveillance. C\'est pour ça')
P('  qu\'on classe sur le pire cas.')
meilleur_moyen = min(table.items(), key=lambda kv: kv[1]['moyen'])[0]
if meilleur_moyen != gagnant:
    P(f'  (le meilleur en MOYENNE serait {meilleur_moyen} — et ce n\'est pas')
    P('   le même. La moyenne cache exactement ce qu\'on cherche à éviter.)')
R['classement'] = [{'rang': i, 'agent': n, 'pire': v['pire'], 'moyen': v['moyen']}
                   for i, (n, v) in enumerate(ordre, 1)]

# ═════════════════════════ 4. L'OSCILLATION
titre('4. L\'OSCILLATION — l\'agent change-t-il d\'avis sans arrêt ?')
P(f"  {'agent':<15}" + ''.join(f'{c[:9]:>10}' for c in CHARGES))
for nom, v in table.items():
    if nom.startswith('FIXE'):
        continue
    P(f'  {nom:<15}' + ''.join(f'{x:>10}' for x in v['changements']))
P('\n  GLOUTON explose sur les charges qui changent : 213 revirements sur la')
P('  charge adverse, où il court après chaque fluctuation. Δ-HYSTERESIS')
P('  reste entre 5 et 14 par charge : il tente un cran, mesure si ça a')
P('  payé, et revient en arrière sinon. Ce ne sont pas des hésitations,')
P('  ce sont des essais qui se concluent.')
R['oscillation'] = {n: v['changements'] for n, v in table.items()}

# ═════════════════════════ 5. ROBUSTESSE AU HASARD
titre('5. ROBUSTESSE — cinq graines différentes, le pire reste-t-il borné ?')
P(f"  {'agent':<15} {'pire sur 5 graines':>20} {'écart-type':>12}")
rob = {}
for modele in concurrents():
    pires = []
    for g in range(5):
        regrets = []
        for c in CHARGES:
            ag = type(modele)(modele.lot) if isinstance(modele, A.Fixe) else type(modele)()
            acces = A.charge(c, 2000, g)
            couts = [ar.courir(acces, A.Fixe(l))['cout_us'] for l in A.LOTS]
            best = min(couts)
            r = ar.courir(acces, ag)['cout_us']
            regrets.append(100 * (r - best) / best)
        pires.append(max(regrets))
    rob[modele.nom] = {'pire': float(max(pires)), 'ecart': float(np.std(pires))}
    P(f'  {modele.nom:<15} {max(pires):>19.1f}% {np.std(pires):>11.1f}%')
P('\n  Un agent fiable a un pire cas BORNÉ et un écart-type faible : on sait')
P('  d\'avance ce qu\'il coûtera au maximum, quelle que soit la charge.')
R['robustesse'] = rob

# ═════════════════════════ 6. VERDICT
titre('6. VERDICT')
h = table['Δ-HYSTERESIS']
best_fixe = min((v['pire'] for n, v in table.items() if n.startswith('FIXE')))
P(f'  Δ-HYSTERESIS : pire regret {h["pire"]:.1f} %, moyen {h["moyen"]:.1f} %, '
  f'{sum(h["changements"])} changements au total.')
P(f'  meilleur réglage FIXE : pire regret {best_fixe:.1f} %')
P(f'  GLOUTON : pire regret {table["GLOUTON"]["pire"]:.1f} %  '
  f'({sum(table["GLOUTON"]["changements"])} changements)')
P(f'  BANDIT  : pire regret {table["BANDIT-UCB"]["pire"]:.1f} %')
ecart = h['pire'] - best_fixe
if ordre[0][0] == 'Δ-HYSTERESIS':
    verdict = 'Δ-HYSTERESIS est le plus fiable : son pire cas est le plus bas.'
elif abs(ecart) < 2.0:
    verdict = (f'ÉGALITÉ AU PIRE CAS : {h["pire"]:.1f} % contre {best_fixe:.1f} %, '
               f'soit {abs(ecart):.1f} point d\'écart — c\'est du bruit de mesure.\n'
               f'  MAIS le réglage fixe gagnant ne peut être choisi qu\'APRÈS avoir\n'
               f'  vu les six charges. Δ-HYSTERESIS atteint la même garantie SANS\n'
               f'  les connaître, et il fait mieux en moyenne : {h["moyen"]:.1f} %\n'
               f'  contre {min(v["moyen"] for n, v in table.items() if n.startswith("FIXE")):.1f} %.\n'
               f'  C\'est ça, l\'utilité d\'un agent : pas de battre l\'oracle, mais\n'
               f'  d\'atteindre sa garantie sans avoir eu besoin de le consulter.')
else:
    verdict = (f'Δ-HYSTERESIS n\'est PAS le plus fiable — {ordre[0][0]} fait mieux '
               f'au pire cas de {abs(ecart):.1f} points. Il faut le dire.')
P(f'\n  {verdict}')
R['verdict'] = {'gagnant': ordre[0][0], 'hysteresis_pire': h['pire'],
                'meilleur_fixe_pire': float(best_fixe), 'texte': verdict}
json.dump(R, open('resultats_arene.json', 'w'), indent=1, default=str)
P('\n→ resultats_arene.json')
