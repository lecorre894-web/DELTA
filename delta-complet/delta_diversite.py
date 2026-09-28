#!/usr/bin/env python3
"""Δ-DIVERSITE V1.0 — LE PONT LEINSTER → Δ.

Δ compte depuis toujours avec UN seul nombre : X, le rang effectif. Leinster
et Cobbold ont montré qu'un seul nombre ne suffit jamais, et qu'il en faut
une FAMILLE, indexée par q :

    D_q(p) = ( somme_i p_i^q )^(1/(1-q))        (nombres de Hill)

      q=0   la richesse brute : combien d'éléments non nuls
      q=1   exp(entropie de Shannon)  <- C'EST DEJA NOTRE RANG PARTICIPATIF
      q=2   inverse de l'indice de Simpson : poids aux dominants
      q=inf 1 / le plus gros poids : seul le géant compte

Et surtout, leur vraie trouvaille : la diversité SENSIBLE À LA SIMILARITÉ.
Compter deux choses presque identiques comme DEUX est une erreur. Avec une
matrice de similarité Z (Z_ij proche de 1 = très semblables) :

    D_q^Z(p) = ( somme_i p_i (Zp)_i^(q-1) )^(1/(1-q))

Si Z = identité (tout est distinct), on retrouve les Hill classiques : c'est
le test de conformité de ce module.

CE QUE ÇA APPORTE À Δ, CONCRÈTEMENT
  La déduplication de Δ16 est BINAIRE : deux pages sont identiques ou elles
  ne le sont pas. Cinquante pages qui diffèrent d'un bit comptent pour
  cinquante. La diversité de Leinster les compte pour ce qu'elles VALENT.
  C'est une déduplication continue, et elle se mesure.

Dépendances : numpy. Termux OK.
"""
import numpy as np


def normaliser(p):
    p = np.asarray(p, float).ravel()
    p = np.clip(p, 0, None)
    s = p.sum()
    return p / s if s > 0 else p


def hill(p, q):
    """Nombre de Hill d'ordre q. Cas q=1 traité à part (limite)."""
    p = normaliser(p)
    p = p[p > 0]
    if abs(q - 1.0) < 1e-12:
        return float(np.exp(-(p * np.log(p)).sum()))
    if np.isinf(q):
        return float(1.0 / p.max())
    return float((p ** q).sum() ** (1.0 / (1.0 - q)))


def hill_Z(p, Z, q):
    """Diversité sensible à la similarité (Leinster & Cobbold)."""
    p = normaliser(p)
    Zp = np.asarray(Z, float) @ p
    m = p > 0
    p, Zp = p[m], Zp[m]
    Zp = np.clip(Zp, 1e-300, None)
    if abs(q - 1.0) < 1e-12:
        return float(np.exp(-(p * np.log(Zp)).sum()))
    if np.isinf(q):
        return float(1.0 / Zp.max())
    return float((p * Zp ** (q - 1.0)).sum() ** (1.0 / (1.0 - q)))


def profil(p, Z=None, qs=(0, 0.5, 1, 2, 4, np.inf)):
    """La COURBE complète, pas un seul chiffre. Elle décroît toujours avec q :
    plus q monte, plus on ne compte que les dominants."""
    if Z is None:
        return [(q, hill(p, q)) for q in qs]
    return [(q, hill_Z(p, Z, q)) for q in qs]


def similarite_cos(V, raideur=1.0):
    """Z construite sur la ressemblance des vecteurs : Z_ij = |cos| ^ raideur.
    Deux directions colinéaires -> 1 (indistinguables). Orthogonales -> 0."""
    V = np.asarray(V, float)
    n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-300)
    return np.abs(n @ n.T) ** raideur


def similarite_expo(V, echelle=1.0):
    """Z_ij = exp(-d_ij/echelle), la forme utilisée pour la MAGNITUDE."""
    V = np.asarray(V, float)
    d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    return np.exp(-d / echelle)


def magnitude(Z):
    """Magnitude de Leinster : la « taille effective » d'un ensemble de
    points. Somme de tous les coefficients de l'inverse de Z. Vaut n si tout
    est distinct, et tend vers 1 si tout se confond."""
    try:
        W = np.linalg.solve(np.asarray(Z, float), np.ones(Z.shape[0]))
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(np.asarray(Z, float)) @ np.ones(Z.shape[0])
    return float(W.sum())


def rang_participatif(spectre):
    """Ce que Δ calculait déjà : exp(entropie du spectre normalisé au carré).
    On va vérifier que c'est EXACTEMENT D_1 des nombres de Hill."""
    s = np.asarray(spectre, float) ** 2
    return hill(s, 1.0)
