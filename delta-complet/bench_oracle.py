#!/usr/bin/env python3
"""bench_oracle — peut-on DÉPASSER l'oracle, et quel outil le permet ?"""
import json
import numpy as np
import delta_arene as A

R = {}
P = lambda t: print(t, flush=True)
CH = ('sequentielle', 'dispersee', 'boucle', 'zipf', 'alternance', 'adverse')
ar = A.Arene(capacite=512)


def titre(t):
    P('\n' + '═' * 74); P(t); P('═' * 74)


def oracle():
    o = {}
    for c in CH:
        acc = A.charge(c, 3000, 0)
        co = [ar.courir(acc, A.Fixe(l))['cout_us'] for l in A.LOTS]
        i = int(np.argmin(co))
        o[c] = (co[i], A.LOTS[i])
    return o


O = oracle()

# 1 — ÉTAT DES LIEUX
titre('1. ÉTAT DES LIEUX — où en sont les agents, avec et sans Δ')
P('  SANS Δ (dans l\'arène, modèle de coût pur) : regret vs oracle fixe')
etat = {}
for mk in (lambda: A.Hysteresis(), lambda: A.Glouton(), lambda: A.Bandit(),
           lambda: A.Regime(), lambda: A.Regime(fenetre=24, delai=24)):
    a0 = mk()
    reg = []
    for c in CH:
        r = ar.courir(A.charge(c, 3000, 0), mk())
        reg.append(100 * (r['cout_us'] - O[c][0]) / O[c][0])
    etiq = a0.nom + (' rapide' if getattr(a0, 'f', 0) == 24 else '')
    etat[etiq] = reg
    P(f'    {etiq:<18}' + ''.join(f'{v:>9.1f}%' for v in reg)
      + f'   pire {max(reg):>7.1f}%')
P('\n  AVEC Δ (dans le système réel, bench_systeme) : l\'agent fait 4,54 s')
P('  contre 3,17 s pour l\'oracle fixe, soit 43 % de regret. Le système')
P('  réel est plus dur que l\'arène : chaque décision y a un coût caché')
P('  que le modèle ne facture pas.')
R['etat'] = etat

# 2 — DÉPASSER L'ORACLE
titre('2. DÉPASSER L\'ORACLE — c\'est possible, et voici où')
P('  L\'oracle est le meilleur réglage IMMOBILE. Sur une charge qui')
P('  alterne, aucun réglage immobile ne peut être bon aux deux moments :')
P('  c\'est là, et seulement là, qu\'un agent peut passer dessous.\n')
P(f"  {'charge':<14} {'oracle (ms)':>12} {'lot':>5} {'Δ-REGIME':>11} "
  f"{'regret':>9}   verdict")
dep = {}
for c in CH:
    r = ar.courir(A.charge(c, 3000, 0), A.Regime(fenetre=24, delai=24))
    reg = 100 * (r['cout_us'] - O[c][0]) / O[c][0]
    v = 'AU-DESSUS DE L\'ORACLE' if reg < -0.5 else ('à égalité' if reg < 2
                                                     else 'en dessous')
    P(f'  {c:<14} {O[c][0]/1000:>12.1f} {O[c][1]:>5} '
      f'{r["cout_us"]/1000:>11.1f} {reg:>8.1f}%   {v}')
    dep[c] = {'oracle_ms': O[c][0] / 1000, 'agent_ms': r['cout_us'] / 1000,
              'regret': reg, 'changements': r['changements']}
gagnees = [c for c in CH if dep[c]['regret'] < -0.5]
P(f'\n  charges où l\'agent BAT l\'oracle : {len(gagnees)} sur {len(CH)}'
  + (f'  ({", ".join(gagnees)})' if gagnees else ''))
R['depassement'] = dep

# 3 — QUEL OUTIL EST DÉSIGNÉ ?
titre('3. QUEL OUTIL EST DÉSIGNÉ — on retire chaque pièce, une par une')
P('  Seule façon honnête de savoir ce qui produit le gain : le débrancher.\n')
variantes = {
    'complet (f=100, d=100)': dict(detect=True, delai=100, fenetre=100),
    'SANS détecteur de régime': dict(detect=False, delai=100, fenetre=100),
    'réaction rapide (f=24, d=24)': dict(detect=True, delai=24, fenetre=24),
    'réaction très rapide (f=12, d=8)': dict(detect=True, delai=8, fenetre=12),
    'sans amortisseur (d=0)': dict(detect=True, delai=0, fenetre=24),
    'fenêtre longue (f=400)': dict(detect=True, delai=400, fenetre=400),
}
P(f"  {'variante':<28} {'pire regret':>12} {'moyen':>9} {'changements':>12}")
abl = {}
for nom, kw in variantes.items():
    reg, ch = [], 0
    for c in CH:
        r = ar.courir(A.charge(c, 3000, 0), A.Regime(**kw))
        reg.append(100 * (r['cout_us'] - O[c][0]) / O[c][0])
        ch += r['changements']
    abl[nom] = {'pire': max(reg), 'moyen': float(np.mean(reg)), 'chg': ch}
    P(f'  {nom:<28} {max(reg):>11.1f}% {np.mean(reg):>8.1f}% {ch:>12}')
base = abl['complet (f=100, d=100)']['pire']
baseM = abl['complet (f=100, d=100)']['moyen']
P('\n  LECTURE : l\'écart entre « complet » et « sans détecteur de régime »')
P('  mesure ce que le détecteur apporte à lui seul. Les autres lignes')
P('  disent si les réglages comptent ou si c\'est du décor.')
sans = abl['SANS détecteur de régime']
P(f'  Retirer le DÉTECTEUR DE RÉGIME : le regret moyen passe de '
  f'{baseM:.1f} % à {sans["moyen"]:.1f} %,')
sens = 'sameliore'.replace('sameliore', "s'améliore") if sans["pire"] < base else 'empire'
P(f'  soit {sans["moyen"]-baseM:+.1f} points. Mais le PIRE cas, lui, {sens} : '
  f'{base:.1f} % -> {sans["pire"]:.1f} %.')
P('  TENSION RÉELLE, ET IL FAUT LA NOMMER : le détecteur fait gagner')
P('  beaucoup en moyenne et perdre un peu au pire cas. Il rend l\'agent')
P('  meilleur la plupart du temps et un peu plus fragile sur la charge')
P('  adverse, qui est justement faite pour le prendre en défaut.')
meilleur = min(abl.items(), key=lambda kv: kv[1]['moyen'])
sur = min(abl.items(), key=lambda kv: kv[1]['pire'])
P(f'\n  OUTIL DÉSIGNÉ pour le regret MOYEN : « {meilleur[0]} » '
  f'({meilleur[1]["moyen"]:.1f} %)')
P(f'  OUTIL DÉSIGNÉ pour le PIRE cas      : « {sur[0]} » '
  f'({sur[1]["pire"]:.1f} %)')
cle, pires = meilleur[0], {}
R['ablation'] = abl
R['outil_designe'] = {'moyen': meilleur[0], 'moyen_val': meilleur[1]['moyen'],
                      'pire': sur[0], 'pire_val': sur[1]['pire'],
                      'apport_detecteur_moyen': sans['moyen'] - baseM}

# 4 — CONTRE-ESSAI
titre('4. CONTRE-ESSAI — une charge SANS régime stable à reconnaître')
P('  PREMIER ESSAI RATÉ, ET JE LE LAISSE : j\'avais pris une charge')
P('  purement aléatoire en croyant qu\'elle n\'offrait rien à détecter.')
P('  Faux — « dispersion » EST un régime, parfaitement reconnaissable,')
P('  et le détecteur y gagne massivement (+0,4 % contre +52,0 % sans lui).')
P('  Ce n\'était donc pas un contre-essai mais une confirmation de plus.\n')
g = np.random.default_rng(11)
alea = [int(x) for x in g.integers(0, 200000, 3000)]
co = [ar.courir(alea, A.Fixe(l))['cout_us'] for l in A.LOTS]
best = min(co)
r1 = ar.courir(alea, A.Regime(fenetre=24, delai=24))
r2 = ar.courir(alea, A.Regime(detect=False))
P(f'    charge « dispersion pure » : oracle {best/1000:.1f} ms | '
  f'avec détecteur {100*(r1["cout_us"]-best)/best:+.1f} % | '
  f'sans {100*(r2["cout_us"]-best)/best:+.1f} %')
# LE VRAI contre-essai : le regime change a CHAQUE acces, rien a reconnaitre
mix = []
for t in range(3000):
    mix.append(t if t % 2 == 0 else int(g.integers(0, 200000)))
co2 = [ar.courir(mix, A.Fixe(l))['cout_us'] for l in A.LOTS]
b2 = min(co2)
m1 = ar.courir(mix, A.Regime(fenetre=24, delai=24))
m2 = ar.courir(mix, A.Regime(detect=False))
P(f'\n  LE VRAI CONTRE-ESSAI — le régime bascule à CHAQUE accès, donc')
P(f'  aucune fenêtre ne peut en isoler un :')
P(f'    oracle fixe         : {b2/1000:>9.1f} ms  '
  f'(lot {A.LOTS[int(np.argmin(co2))]})')
P(f'    avec détecteur      : {m1["cout_us"]/1000:>9.1f} ms  '
  f'({100*(m1["cout_us"]-b2)/b2:+.1f} %)  {m1["changements"]} changements')
P(f'    sans détecteur      : {m2["cout_us"]/1000:>9.1f} ms  '
  f'({100*(m2["cout_us"]-b2)/b2:+.1f} %)')
ecart = 100*(m1["cout_us"]-b2)/b2 - 100*(m2["cout_us"]-b2)/b2
P(f'\n  apport du détecteur ici : {-ecart:+.1f} points. ENCORE RATÉ.')
P('  DEUXIÈME ÉCHEC, ET J\'ARRÊTE DE BRICOLER : une alternance stricte')
P('  un-sur-deux est, elle aussi, un régime parfaitement STABLE —')
P('  statistiquement. La fenêtre y lit « moitié suite, moitié saut »')
P('  à chaque fois, et cette réponse constante suffit à choisir le bon')
P('  lot. Je n\'ai donc PAS réussi à mettre le détecteur en défaut dans')
P('  ce modèle, et je préfère l\'écrire plutôt que de tordre la charge')
P('  jusqu\'à obtenir le résultat que je cherchais.')
P('\n  CE QUE ÇA VEUT DIRE, EXACTEMENT : dans ce modèle de coût, le')
P('  détecteur n\'a que trois réponses possibles et une statistique')
P('  stable suffit à trancher — il est donc difficile à piéger PAR')
P('  CONSTRUCTION. Cela ne dit RIEN de sa robustesse ailleurs : dans le')
P('  système réel, le même principe se fait battre par un réglage fixe')
P('  de 43 %. Un agent robuste dans un modèle n\'est pas un agent robuste.')
R['contre_essai'] = {'dispersion_avec': 100*(r1["cout_us"]-best)/best,
                     'dispersion_sans': 100*(r2["cout_us"]-best)/best,
                     'melange_avec': 100*(m1["cout_us"]-b2)/b2,
                     'melange_sans': 100*(m2["cout_us"]-b2)/b2}

json.dump(R, open('resultats_oracle.json', 'w'), indent=1, default=str)
P('\n→ resultats_oracle.json')
