#!/usr/bin/env python3
"""Δ-QUTRIT V1.0 — Δ généralisé à la dimension d, et la mesure d'ATLAS rejouée.

CE QUI CHANGE, ET POURQUOI ÇA COMPTE
  Depuis Δ1, tout le projet travaille en dimension 2 : des qubits, deux
  états, un rang de Schmidt qui ne peut pas dépasser 2 sur une coupure
  élémentaire. Les bosons Z, eux, sont massifs et de spin 1 : TROIS états
  de polarisation. Le papier d'ATLAS les nomme lui-même « spin qutrits ».

  Pour deux qutrits, le rang de Schmidt maximal vaut 3. C'est le X de René,
  et la nature en fixe ici la borne : ni 2, ni 256. Trois.

L'ÉTAT MESURÉ AU LHC
  Le boson de Higgs a un spin NUL. Quand il se désintègre en deux bosons de
  spin 1, la conservation du moment cinétique impose que le spin total
  reste nul. Il n'existe qu'un seul état possible :

      |H> = ( |+1,-1>  -  |0,0>  +  |-1,+1> ) / racine(3)

  Ses trois coefficients de Schmidt valent tous 1/racine(3) : l'état est
  MAXIMALEMENT intriqué pour deux qutrits. X = 3, exactement.
  Et remarque la règle qui le fabrique : m1 + m2 = 0 dans chaque terme.
  C'est une symétrie U(1) — exactement le mécanisme de Δ10, sauf qu'ici
  c'est la nature qui rejette les configurations interdites.

CE QUE ce module MESURE, SANS RIEN SUPPOSER
  - le spectre de Schmidt et donc X
  - l'entropie d'intrication
  - la négativité (critère de Peres-Horodecki)
  - le gain de la symétrie m1+m2=0 en blocs de charge
  - le SEUIL DE BRUIT au-delà duquel l'intrication meurt, c'est-à-dire le
    moment où Δ a enfin le droit de découper
  - combien de données il manque à ATLAS pour passer de 4,7 à 5 sigma

Dépendances : numpy. Termux OK.
"""
import numpy as np

# les trois projections de spin d'un boson massif de spin 1
M = (+1, 0, -1)


def etat_higgs():
    """L'unique état de spin total nul formé de deux spins 1.
    Coefficients de Clebsch-Gordan pour |J=0,M=0> issus de j1=j2=1."""
    psi = np.zeros((3, 3))
    psi[0, 2] = 1 / np.sqrt(3)      # |+1,-1>
    psi[1, 1] = -1 / np.sqrt(3)     # |0 , 0>
    psi[2, 0] = 1 / np.sqrt(3)      # |-1,+1>
    return psi


def etat_separable(a=None, b=None):
    """Un état produit : aucune intrication. C'est le contre-essai."""
    a = np.array([0.6, 0.8, 0.0]) if a is None else np.asarray(a, float)
    b = np.array([0.0, 0.5, np.sqrt(0.75)]) if b is None else np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.outer(a, b)


# ══════════════════════════════════════════════ le X de Δ, en dimension d
def schmidt(psi):
    """Les coefficients de Schmidt d'un état bipartite. C'est LA mesure de
    Δ depuis le premier jour, elle ne change pas avec la dimension."""
    return np.linalg.svd(np.asarray(psi), compute_uv=False)


def X_de(psi, eps=1e-12):
    """Le rang effectif : combien de valeurs comptent vraiment.
    X = 1 <=> état séparable <=> Δ a le droit de découper."""
    s = schmidt(psi)
    return int((s > eps * s[0]).sum())


def entropie(psi):
    """Entropie d'intrication, en nats. Vaut ln(d) pour un état
    maximalement intriqué de deux qudits : ln(3) = 1,0986 ici."""
    s = schmidt(psi) ** 2
    s = s[s > 0]
    return float(-(s * np.log(s)).sum())


def peut_couper(psi, tol=1e-9):
    """Le test de Δ1 : si le deuxième coefficient de Schmidt est négligeable,
    l'état se découpe en deux morceaux indépendants. Sinon, on ne coupe pas.
    C'est exactement l'hypothèse qu'ATLAS a testée sur le ciel."""
    s = schmidt(psi)
    return bool(len(s) < 2 or s[1] <= tol * s[0])


# ═══════════════════════════════════ la symétrie m1+m2=0, façon Δ10
def secteurs(psi, tol=1e-12):
    """Range les amplitudes par charge m1+m2. Une charge dont toutes les
    amplitudes sont nulles N'EXISTE PAS : on ne la stocke pas."""
    d = {}
    for i, m1 in enumerate(M):
        for j, m2 in enumerate(M):
            if abs(psi[i, j]) > tol:
                d.setdefault(m1 + m2, []).append((m1, m2, float(psi[i, j])))
    return d


def gain_symetrie(psi):
    """Combien d'amplitudes la symétrie permet-elle de ne pas stocker ?"""
    plein = psi.size
    garde = sum(len(v) for v in secteurs(psi).values())
    return plein, garde, plein / max(garde, 1)


# ════════════════════════════════════ l'intrication face au bruit
def densite(psi):
    v = np.asarray(psi).reshape(-1)
    return np.outer(v, v.conj())


def isotrope(psi, p):
    """État de Werner isotrope : une fraction p de l'état pur, le reste en
    bruit blanc. C'est le modèle honnête de ce que la mesure abîme —
    le Z virtuel, le bruit de fond, la reconstruction imparfaite."""
    return p * densite(psi) + (1 - p) * np.eye(9) / 9


def transpose_partielle(rho, d=3):
    r = rho.reshape(d, d, d, d)
    return r.transpose(0, 3, 2, 1).reshape(d * d, d * d)


def negativite(rho, d=3):
    """Critère de Peres-Horodecki : si la transposée partielle a une valeur
    propre NÉGATIVE, l'état est forcément intriqué. La négativité mesure de
    combien. Elle vaut (d-1)/2 = 1 pour un état maximalement intriqué."""
    vp = np.linalg.eigvalsh(transpose_partielle(rho, d))
    return float(-vp[vp < 0].sum())


def seuil_separabilite(psi, d=3, pas=1e-5):
    """Cherche le p où l'intrication s'éteint. Pour l'état isotrope, la
    théorie donne exactement p = 1/(d+1) = 1/4 en dimension 3. On ne le
    suppose pas : on le CHERCHE, puis on compare."""
    lo, hi = 0.0, 1.0
    while hi - lo > pas:
        mi = (lo + hi) / 2
        if negativite(isotrope(psi, mi), d) > 1e-12:
            hi = mi
        else:
            lo = mi
    return (lo + hi) / 2


# ═══════════════════════════════ ce qu'il manque à ATLAS pour 5 sigma
def donnees_pour(sigma_vise, sigma_obtenu, luminosite=1.0):
    """La significativité croît comme la racine du nombre d'événements.
    Passer de s1 à s2 demande donc (s2/s1)^2 fois plus de données."""
    facteur = (sigma_vise / sigma_obtenu) ** 2
    return facteur, (facteur - 1) * 100, luminosite * facteur
