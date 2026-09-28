#!/usr/bin/env python3
"""Δ-ATOME V1.0 — LE PREMIER ATOME DE Δ, ET LA NATURE POUR JUGE.

« ATOME ARTIFICIEL » n'est pas une image : c'est le nom que les physiciens
donnent aux boîtes quantiques et aux qubits supraconducteurs — des systèmes
fabriqués qui possèdent, comme un atome, des niveaux d'énergie DISCRETS.

Δ n'a encore jamais fait de physique. Il a fait du calcul, du cache, de la
cryptographie, des neurones. Ici il doit produire une quantité que personne
n'a choisie : l'énergie de l'atome d'hydrogène.

LE JUGE, ET IL EST IMPLACABLE
  On ne compare pas Δ à un autre programme. On le compare à la NATURE :
    - l'énergie d'ionisation de l'hydrogène : 13,6057 eV, mesurée depuis
      plus d'un siècle
    - les raies de Balmer, celles qu'on voit dans le spectre du Soleil et
      de toute étoile : 656,3 / 486,1 / 434,0 / 410,2 nanomètres
  Δ part de RIEN : pas de formule des niveaux, pas de table. Seulement
  l'équation de Schrödinger discrétisée, et une matrice à diagonaliser.
  Si les chiffres tombent, ils tombent parce que la physique est juste.

CE QUE Δ APPORTE À L'ATOME
  Une fois les niveaux obtenus, on pose la question de Δ : combien de
  fonctions de base faut-il VRAIMENT garder ? C'est le X, appliqué à un
  atome. Et comme toujours, l'erreur est mesurée et déclarée.

UNITÉS ATOMIQUES : hbar = m_e = e = 1. L'énergie est en hartree.
  1 hartree = 27,211386245988 eV  (CODATA)
  1 bohr    = 0,529177210903 angström
  Dans ces unités, la théorie dit E_n = -1/(2 n^2), exactement.

Dépendances : numpy. Termux OK.
"""
import numpy as np

HARTREE_EV = 27.211386245988          # CODATA
BOHR_NM = 0.0529177210903             # nanomètres
# constante de structure fine et vitesse de la lumière en unités atomiques
ALPHA = 7.2973525693e-3
C_UA = 1.0 / ALPHA
# longueur d'onde (nm) d'un photon d'énergie E (hartree) : hc/E
HC_NM_EV = 1239.841984               # eV.nm  (h c)


def hamiltonien_radial(l=0, Z=1, r_max=120.0, n_points=6000):
    """L'équation de Schrödinger radiale, discrétisée en différences finies.

        -1/2 u'' + [ l(l+1)/(2 r^2) - Z/r ] u = E u,   avec u = r R(r)

    La matrice est tridiagonale symétrique : la diagonaliser, c'est
    résoudre l'atome. Rien d'autre n'est injecté — surtout pas la réponse."""
    r = np.linspace(r_max / n_points, r_max, n_points)
    h = r[1] - r[0]
    diag = 1.0 / h ** 2 + l * (l + 1) / (2.0 * r ** 2) - Z / r
    hors = -0.5 / h ** 2 * np.ones(n_points - 1)
    return r, diag, hors


def niveaux(l=0, Z=1, combien=6, r_max=120.0, n_points=6000):
    """Les énergies propres liées (négatives), avec leurs fonctions d'onde.

    PREMIER GAIN DE STRUCTURE, ET IL EST GRATUIT : l'opérateur de dérivée
    seconde ne relie que des points VOISINS. La matrice est donc
    tridiagonale : 3 diagonales au lieu de n^2 éléments. On ne la construit
    jamais en entier — un solveur tridiagonal suffit."""
    r, diag, hors = hamiltonien_radial(l, Z, r_max, n_points)
    try:
        from scipy.linalg import eigh_tridiagonal
        E, V = eigh_tridiagonal(diag, hors, select='i',
                                select_range=(0, combien + 2))
    except Exception:
        E, V = np.linalg.eigh(np.diag(diag) + np.diag(hors, 1)
                              + np.diag(hors, -1))
    garde = E < 0
    E, V = E[garde], V[:, garde]
    return r, E[:combien], V[:, :combien]


def cout_matrice(n_points):
    """Ce que coûterait la matrice pleine, contre ce que coûtent ses trois
    diagonales. C'est la LOCALITÉ de l'opérateur qui paye — exactement le
    cône de lumière de Δ11, mais sur un atome."""
    plein = n_points * n_points
    creux = 2 * n_points - 1
    return plein, creux, plein / creux


def energie_exacte(n, Z=1):
    """La théorie : E_n = -Z^2 / (2 n^2) hartree. C'est CE chiffre-là que
    Δ doit retrouver sans l'avoir jamais lu."""
    return -Z ** 2 / (2.0 * n ** 2)


def longueur_onde_nm(E_haut, E_bas):
    """Longueur d'onde du photon émis entre deux niveaux (hartree -> nm)."""
    dE_eV = (E_haut - E_bas) * HARTREE_EV
    return HC_NM_EV / abs(dE_eV)


def rydberg_nm(n_haut, n_bas=2, masse_reduite=True):
    """Formule de Rydberg, avec ou sans correction de masse réduite.
    Le proton n'est pas infiniment lourd : l'électron et lui tournent autour
    de leur centre de masse commun. La correction vaut 1/(1+m_e/m_p) et se
    voit dès la troisième décimale."""
    R_inf_nm = 1.0 / 1.0973731568160e-2      # nm (R_inf = 1,097...e7 m^-1)
    mu = 1.0 / (1.0 + 1.0 / 1836.15267343) if masse_reduite else 1.0
    return R_inf_nm / (mu * (1.0 / n_bas ** 2 - 1.0 / n_haut ** 2))


# ═══════════════════════════════ le X de Δ, appliqué à un atome
def densite_radiale(r, u):
    """|u|^2 normalisée : la probabilité de trouver l'électron à la
    distance r. C'est l'objet physique, celui qu'on peut comprimer."""
    p = u ** 2
    return p / np.trapezoid(p, r)


def X_effectif(M, eps=1e-3):
    """Combien de modes faut-il garder pour ne perdre qu'une part eps ?
    Exactement le X de Δ8, appliqué cette fois à un atome."""
    s = np.linalg.svd(np.asarray(M), compute_uv=False)
    tot = float((s ** 2).sum())
    cum = np.cumsum(s[::-1] ** 2)[::-1]
    for k in range(1, len(s) + 1):
        if (cum[k] if k < len(cum) else 0.0) / tot <= eps ** 2:
            return k, s
    return len(s), s


def tronquer(M, k):
    u, s, vh = np.linalg.svd(np.asarray(M), full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k, :]


def regle_selection(l1, l2):
    """Un photon emporte un moment cinétique de 1. Une transition n'est
    permise que si l change de 1 exactement. Ce n'est pas une convention :
    c'est encore une SYMÉTRIE qui interdit, comme le spin nul du Higgs."""
    return abs(l1 - l2) == 1


def potentiel_aleatoire(n_points=6000, r_max=120.0, graine=0):
    """Le contre-essai : un potentiel sans aucune structure. Ses niveaux ne
    doivent suivre AUCUNE loi simple, et rien ne doit se comprimer."""
    rng = np.random.default_rng(graine)
    r = np.linspace(r_max / n_points, r_max, n_points)
    h = r[1] - r[0]
    V = -np.abs(rng.standard_normal(n_points)) * 3.0
    diag = 1.0 / h ** 2 + V
    hors = -0.5 / h ** 2 * np.ones(n_points - 1)
    try:
        from scipy.linalg import eigh_tridiagonal
        E, W = eigh_tridiagonal(diag, hors, select='i', select_range=(0, 9))
    except Exception:
        E, W = np.linalg.eigh(np.diag(diag) + np.diag(hors, 1) + np.diag(hors, -1))
    g = E < 0
    return r, E[g][:8], W[:, g][:, :8]
