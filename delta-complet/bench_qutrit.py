#!/usr/bin/env python3
"""bench_qutrit — Δ en dimension 3, et la mesure d'ATLAS rejouée dedans.

Référence extérieure : ATLAS, arXiv 2603.26463, « Measurements of Z-boson
pair entanglement in decays of Higgs bosons », 4,7 sigma observés.
"""
import json
import numpy as np
import delta_qutrit as Q
import delta1 as D1  # noqa: F401  (Δ1 reste la référence conceptuelle du découpage)

R = {}


def titre(t):
    print('\n' + '═' * 74); print(t); print('═' * 74, flush=True)


# ═══════════════════════════════════ 1. L'ÉTAT, ET SON X
titre('1. L\'ÉTAT DU HIGGS — et le X que la nature lui impose')
psi = Q.etat_higgs()
s = Q.schmidt(psi)
print('  |H> = ( |+1,-1>  -  |0,0>  +  |-1,+1> ) / racine(3)')
print(f'  norme : {np.linalg.norm(psi):.15f}')
print(f'  coefficients de Schmidt : {np.round(s, 12)}')
print(f'  tous égaux à 1/racine(3) = {1/np.sqrt(3):.12f} : '
      f'{np.allclose(s, 1/np.sqrt(3))}')
print(f'  X = {Q.X_de(psi)}   (maximum possible pour deux qutrits : 3)')
print(f'  entropie d\'intrication : {Q.entropie(psi):.12f} nats')
print(f'  ln(3) = {np.log(3):.12f}  ->  écart {abs(Q.entropie(psi)-np.log(3)):.2e}')
print('\n  L\'état est MAXIMALEMENT intriqué. En dimension 2 on n\'aurait pas pu')
print('  dépasser X = 2 ; ici la nature monte à 3, et pas plus haut. Le X de Δ')
print('  n\'est pas un réglage libre : c\'est la dimension locale qui le borne.')
R['etat'] = {'X': Q.X_de(psi), 'entropie': Q.entropie(psi), 'ln3': float(np.log(3)),
             'schmidt': [float(x) for x in s]}

# ═══════════════════════════════════ 2. LE TEST D'ATLAS, DANS Δ
titre('2. LE TEST D\'ATLAS REJOUÉ — Δ a-t-il le droit de découper ?')
sep = Q.etat_separable()
for nom, e in (('état du Higgs (H -> ZZ*)', psi), ('état produit (contre-essai)', sep)):
    print(f'  {nom:<32} X = {Q.X_de(e)}   '
          f'coupure autorisée : {Q.peut_couper(e)}')
print('\n  C\'est EXACTEMENT l\'hypothèse qu\'ATLAS a testée sur le ciel :')
print('  « l\'état est-il séparable ? », c\'est-à-dire « X vaut-il 1 ? ».')
print('  Le détecteur a répondu NON à 4,7 sigma. Le test de découpage de Δ1')
print('  répond NON aussi, et pour la même raison : le deuxième coefficient')
print(f'  de Schmidt vaut {Q.schmidt(psi)[1]:.6f}, il ne tombe pas sous la tolérance.')
R['test'] = {'higgs_coupable': Q.peut_couper(psi), 'separable_coupable': Q.peut_couper(sep),
             'X_higgs': Q.X_de(psi), 'X_separable': Q.X_de(sep)}

# ═══════════════════════════════════ 3. LA SYMÉTRIE, FAÇON Δ10
titre('3. LA SYMÉTRIE m1+m2 = 0 — Δ10 appliqué à la physique réelle')
sect = Q.secteurs(psi)
plein, garde, gain = Q.gain_symetrie(psi)
for charge in sorted(sect, reverse=True):
    sg = lambda m: ('+1' if m > 0 else ('-1' if m < 0 else ' 0'))
    termes = ' , '.join(f'|{sg(m1)},{sg(m2)}> x {a:+.5f}' for m1, m2, a in sect[charge])
    print(f'  charge m1+m2 = {charge:+d} : {termes}')
print(f'\n  amplitudes possibles : {plein}   réellement non nulles : {garde}'
      f'   -> ×{gain:.0f}')
print('  Le Higgs a un spin NUL : la conservation du moment cinétique interdit')
print('  toute configuration où m1+m2 est différent de zéro. Six amplitudes sur')
print('  neuf n\'existent pas — non pas « petites », INEXISTANTES.')
print('  Dans Δ10 on rejette une porte qui viole la charge ; ici c\'est la')
print('  nature qui refuse, et c\'est cette contrainte qui FABRIQUE l\'intrication.')
print('  L\'état n\'est pas intriqué par hasard : il l\'est parce qu\'une symétrie')
print('  lui interdit d\'être autre chose.')
R['symetrie'] = {'amplitudes': plein, 'non_nulles': garde, 'gain': gain,
                 'charges': sorted(sect)}

# ═══════════════════════════════════ 4. NÉGATIVITÉ ET BRUIT
titre('4. JUSQU\'OÙ L\'INTRICATION RÉSISTE — et où Δ récupère le droit de couper')
rho = Q.densite(psi)
n0 = Q.negativite(rho)
print(f'  négativité de l\'état pur : {n0:.12f}')
print(f'  valeur théorique (d-1)/2 = {(3-1)/2:.1f}  ->  écart {abs(n0-1.0):.2e}')
print('\n  {:>24} {:>12}   verdict'.format("fraction p d'état pur", "négativité"))
lignes = []
for p in (1.0, 0.8, 0.6, 0.4, 0.3, 0.26, 0.25, 0.24, 0.2, 0.1):
    n = Q.negativite(Q.isotrope(psi, p))
    v = 'INTRIQUÉ' if n > 1e-12 else 'séparable — Δ peut couper'
    print(f'  {p:>24.2f} {n:>12.6f}   {v}')
    lignes.append({'p': p, 'negativite': n})
seuil = Q.seuil_separabilite(psi)
print(f'\n  seuil CHERCHÉ par dichotomie : p = {seuil:.6f}')
print(f'  seuil THÉORIQUE 1/(d+1)     : p = {1/4:.6f}')
print(f'  écart : {abs(seuil-0.25):.2e}')
print('\n  Voilà le pont entre la physique et le canon de Δ : eps, c\'est ce')
print('  qu\'on a le droit de PERDRE. Ici la nature fixe le point exact où la')
print('  perte devient totale. Au-dessus de p = 1/4, Δ n\'a PAS le droit de')
print('  découper l\'état, quelle que soit la tolérance qu\'on lui donne.')
print('  En dessous, le découpage devient légitime — l\'intrication a disparu')
print('  dans le bruit. Ce seuil n\'est pas un réglage : c\'est une frontière.')
R['bruit'] = {'negativite_pure': n0, 'seuil_mesure': seuil, 'seuil_theorique': 0.25,
              'courbe': lignes}

# ═══════════════════════════════════ 5. CE QU'IL MANQUE POUR 5 SIGMA
titre('5. COMBIEN DE DONNÉES MANQUE-T-IL À ATLAS ?')
f, pct, _ = Q.donnees_pour(5.0, 4.7)
f3, pct3, _ = Q.donnees_pour(3.0, 4.7)
print(f'  observé : 4,7 sigma   (attendu : 4,9)')
print(f'  seuil de DÉCOUVERTE en physique des particules : 5 sigma')
print(f'  la significativité croît comme racine(N) :')
print(f'    pour atteindre 5,0 sigma -> ×{f:.3f} de données, soit +{pct:.0f} %')
print(f'    pour 7 sigma             -> ×{Q.donnees_pour(7.0, 4.7)[0]:.2f}')
print(f'    pour 10 sigma            -> ×{Q.donnees_pour(10.0, 4.7)[0]:.2f}')
print('\n  Treize pour cent. C\'est tout ce qui sépare « évidence forte » de')
print('  « découverte ». Le HiLumi n\'apportera pas une idée neuve : il')
print('  apportera du VOLUME, et les barres d\'erreur se resserreront toutes')
print('  seules. C\'est la même loi que partout chez nous — on ne multiplie')
print('  pas ce qu\'on n\'a pas, on accumule ce qui existe.')
R['sigma'] = {'observe': 4.7, 'attendu': 4.9, 'facteur_pour_5': f, 'pourcent': pct}

# ═══════════════════════════════════ 6. CE QUE Δ NE PEUT PAS DIRE
titre('6. CE QUE CE CALCUL NE DIT PAS — et il faut le dire')
print('  · Δ reconstruit ici l\'état THÉORIQUE imposé par la symétrie, et il le')
print('    fait exactement. Il ne rejoue PAS l\'analyse d\'ATLAS : reconstruire')
print('    les coefficients C à partir des angles des quatre leptons demande')
print('    leur convention de normalisation des opérateurs tensoriels, que je')
print('    n\'ai pas. Leurs C_(2,1,2,-1) = -0,71 ± 0,45 ne sont donc PAS')
print('    comparables tels quels aux nombres ci-dessus.')
print('  · l\'étoile de ZZ* compte : l\'un des deux Z est virtuel (125 GeV ne')
print('    suffisent pas pour deux Z réels à 91 GeV chacun). L\'état réel n\'est')
print('    donc pas le singulet pur, mais un mélange — d\'où le modèle isotrope')
print('    du §4, qui est une approximation déclarée, pas la vérité du détecteur.')
print('  · Δ ne prouve rien sur la nature. Il montre qu\'un état contraint par')
print('    une symétrie est forcément intriqué, et à quel point. C\'est ATLAS')
print('    qui a fait la mesure ; Δ ne fait que retrouver ce que la structure')
print('    imposait d\'avance.')

json.dump(R, open('resultats_qutrit.json', 'w'), indent=1, default=str)
print('\n→ resultats_qutrit.json')
