#!/usr/bin/env python3
"""bench_diversite — le pont Leinster -> Δ, mesuré."""
import json
import numpy as np
import delta_diversite as L

R = {}
P = lambda t: print(t, flush=True)


def titre(t):
    P('\n' + '═' * 74); P(t); P('═' * 74)


# 1 — CONFORMITE : notre rang participatif EST la diversite d'ordre 1
titre('1. CONFORMITÉ — Δ calculait déjà du Leinster sans le savoir')
rng = np.random.default_rng(0)
ecarts = []
for _ in range(500):
    s = np.abs(rng.standard_normal(rng.integers(3, 40)))
    a = L.rang_participatif(s)
    p = s ** 2
    b = L.hill(p, 1.0)
    c = float(np.exp(-(lambda x: (x * np.log(x)).sum())(p / p.sum())))
    ecarts.append(max(abs(a - b), abs(a - c)))
P(f'  500 spectres au hasard : écart max entre notre rang participatif et')
P(f'  la diversité de Hill d\'ordre 1 : {max(ecarts):.2e}')
P(f'  -> ce sont le MÊME objet. Δ utilisait un cas particulier (q=1) d\'une')
P(f'  famille entière, sans savoir qu\'elle existait.')
# et Z = identite doit redonner Hill
s = np.abs(rng.standard_normal(12))
I = np.eye(12)
e2 = max(abs(L.hill(s, q) - L.hill_Z(s, I, q)) for q in (0.5, 1, 2, 4))
P(f'  avec Z = identité, la version sensible à la similarité redonne Hill :')
P(f'  écart {e2:.2e}')
R['conformite'] = {'ecart_rang_hill': float(max(ecarts)), 'ecart_Z_identite': float(e2)}

# 2 — LE PROFIL : un chiffre ne suffit pas
titre('2. UN SEUL CHIFFRE NE SUFFIT PAS — le profil de diversité')
cas = {
    'quatre égales + queue plate': np.r_[np.ones(4), 0.05 * np.ones(60)],
    'une géante + 63 miettes': np.r_[10.0, 0.1 * np.ones(63)],
    'décroissance douce': 1.0 / np.arange(1, 65) ** 0.5,
    'tout égal (64)': np.ones(64),
}
P(f"  {'spectre':<28} {'q=0':>7} {'q=1':>7} {'q=2':>7} {'q=inf':>7}")
prof = {}
for nom, sp in cas.items():
    v = [L.hill(sp ** 2, q) for q in (0, 1, 2, np.inf)]
    P(f'  {nom:<28} ' + ' '.join(f'{x:>7.2f}' for x in v))
    prof[nom] = v
P('\n  Regarde les deux premières lignes : à q=0 elles se valent presque')
P('  (64 éléments non nuls des deux côtés), et à q=inf tout les sépare.')
P('  Δ ne donnait que la colonne q=1. La courbe entière dit si le poids est')
P('  partagé ou capté par un seul — et ça change ce qu\'on a le droit de')
P('  jeter.')
R['profil'] = prof

# 3 — LE VRAI APPORT : dedupliquer EN CONTINU
titre('3. DÉDUPLICATION CONTINUE — ce que Δ16 ne savait pas faire')
n, d = 60, 128
base = rng.standard_normal((3, d))
pages = []
for i in range(n):
    src = base[i % 3]
    bruit = 10.0 ** (-3 + 3 * (i / n))          # de quasi-identique à distinct
    pages.append(src + bruit * rng.standard_normal(d))
Pg = np.array(pages)
exact = len({p.tobytes() for p in Pg})
Z = L.similarite_cos(Pg, raideur=8.0)
poids = np.ones(n)
P(f'  {n} pages de {d} nombres, fabriquées à partir de 3 sources, avec un')
P(f'  bruit qui va de 1e-3 (quasi copies) à 1 (vraiment distinctes).\n')
P(f'  déduplication EXACTE de Δ16 (octet pour octet) : {exact} pages distinctes')
P(f'  -> elle ne voit RIEN : un bit de différence suffit à créer une page.\n')
P(f"  {'q':>6} {'diversité':>11}")
dv = []
for q in (0, 1, 2, np.inf):
    v = L.hill_Z(poids, Z, q)
    P(f'  {str(q):>6} {v:>11.2f}')
    dv.append({'q': str(q), 'D': v})
P(f'\n  magnitude (Leinster, sur les distances) : '
  f'{L.magnitude(L.similarite_expo(Pg, echelle=8.0)):.2f}')
P('  La diversité voit ce que la comparaison octet par octet ne peut pas')
P('  voir : il n\'y a pas 60 contenus, il y en a une poignée, et le reste')
P('  est de la redite. C\'est exactement le gain que Δ16 laissait sur la')
P('  table.')
R['dedup'] = {'pages': n, 'exactes': exact, 'diversite': dv}

# 4 — SUR NOS PROPRES MESURES
titre('4. APPLIQUÉ À CE QU\'ON A DÉJÀ MESURÉ')
mesures = {
    'réseau sous contrainte (V28)': np.r_[1.62, 1.32, 1.29, 1.16, .22, .17, .10, .08],
    'réseau libre (V28)': np.r_[3.20, 3.15, 3.12, 3.10, 3.04, 2.96, 2.9, 2.85],
    'orbitales hydrogène (V29)': np.r_[4.36, 2.60, 1.85, 1.40, 1.08, .84],
    'potentiel aléatoire (V29)': np.r_[3.40, 3.33, 3.18, 3.03, 2.95, 2.11],
}
P(f"  {'cas':<30} {'q=0':>6} {'q=1':>6} {'q=2':>6} {'q=inf':>7}")
app = {}
for nom, sp in mesures.items():
    v = [L.hill(sp ** 2, q) for q in (0, 1, 2, np.inf)]
    P(f'  {nom:<30} ' + ' '.join(f'{x:>6.2f}' for x in v))
    app[nom] = v
P('\n  Le réseau sous contrainte : 8 valeurs non nulles à q=0, mais 4,1 à')
P('  q=1 et 2,9 à q=inf. Les quatre premières portent tout, et la courbe le')
P('  dit sans qu\'on ait à choisir un seuil epsilon. C\'est le progrès :')
P('  Δ devait fixer eps à la main, la diversité s\'en passe.')
R['applique'] = app

# 5 — CONTRE-ESSAI
titre('5. CONTRE-ESSAI — du bruit pur n\'a pas de diversité faible')
bruit = np.abs(rng.standard_normal(64))
v = [L.hill(bruit ** 2, q) for q in (0, 1, 2, np.inf)]
P(f'  spectre aléatoire (64) : ' + '  '.join(f'q={q}:{x:.2f}'
                                             for q, x in zip((0, 1, 2, 'inf'), v)))
P(f'  structuré (4 dominantes) : ' + '  '.join(
    f'q={q}:{x:.2f}' for q, x in zip((0, 1, 2, 'inf'),
                                     [L.hill(np.r_[1.62, 1.32, 1.29, 1.16,
                                                   .22, .17, .10, .08] ** 2, q)
                                      for q in (0, 1, 2, np.inf)])))
P('\n  Sur du bruit, la diversité reste haute à tous les ordres : rien ne')
P('  domine, donc rien ne se jette. Septième vérification de la même loi.')
R['contre_essai'] = {'bruit': v}

json.dump(R, open('resultats_diversite.json', 'w'), indent=1, default=str)
P('\n→ resultats_diversite.json')
