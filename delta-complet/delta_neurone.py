#!/usr/bin/env python3
"""Δ-NEURONE V1.0 — L'IMBRICATION DANS UN RÉSEAU DE NEURONES.

L'IDÉE DE RENÉ : passer de l'intrication quantique à l'imbrication neuronale.

Ce n'est pas une analogie molle. La décomposition de Schmidt qui donne le X
de Δ est exactement la décomposition en valeurs singulières d'une matrice.
Une couche de neurones EST une matrice. On peut donc lui demander la même
chose qu'à un état quantique : combien de directions comptent vraiment ?

    état quantique      couche de neurones
    -------------------------------------------------
    amplitudes          poids
    coefficients        valeurs singulières
      de Schmidt
    X = rang            X = rang effectif de la couche
    séparable (X=1)     un seul motif suffit
    intriqué            les entrées sont IMBRIQUÉES : on ne peut pas
                        les traiter séparément sans perdre

LE TEST QUI DÉCIDE, ET IL BOUCLE TOUT LE PROJET
  Depuis Δ1, la même loi revient : le gain vient de la STRUCTURE, jamais de
  la magie. Or l'apprentissage est précisément un procédé qui FABRIQUE de la
  structure. Donc :
      - un réseau entraîné sur un problème structuré devrait devenir
        compressible, et Δ devrait lire cette structure dans ses poids ;
      - un réseau entraîné sur du BRUIT pur ne devrait rien offrir du tout.
  Si le second se comprimait aussi bien que le premier, toute la loi de Δ
  s'effondrerait. Le contre-essai est donc ici obligatoire, pas décoratif.

LE PROTOCOLE
  On CACHE une structure connue : la cible ne dépend que de r directions
  parmi n (r = 4, n = 64). Le réseau ne le sait pas. On l'entraîne, puis on
  regarde si le rang effectif de sa première couche retombe sur r, et si les
  directions qu'il a trouvées sont bien CELLES-LÀ.

Dépendances : numpy. Termux OK.
"""
import numpy as np


# ═════════════════════════════════════════ le problème, à structure cachée
def probleme(n=64, r=4, m=3000, graine=0):
    """La cible ne dépend QUE de r directions de l'espace d'entrée.
    Les n-r autres sont du décor : le réseau doit les ignorer tout seul."""
    rng = np.random.default_rng(graine)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))    # r directions orthonormées
    X = rng.standard_normal((m, n))
    Z = X @ U                                            # les r coordonnées utiles
    y = (np.tanh(1.6 * Z[:, 0]) * np.tanh(1.2 * Z[:, 1])
         + 0.7 * np.sin(1.3 * Z[:, 2]) - 0.5 * Z[:, 3] ** 2 + 0.4 * Z[:, 0] * Z[:, 3])
    y = (y - y.mean()) / y.std()
    return X, y[:, None], U


def bruit_pur(n=64, m=3000, graine=1):
    """Même taille, aucune structure : la cible est tirée au hasard.
    Le réseau ne peut que MÉMORISER. C'est le contre-essai."""
    rng = np.random.default_rng(graine)
    X = rng.standard_normal((m, n))
    y = rng.standard_normal((m, 1))
    return X, y, None


# ═══════════════════════════════════════════════════ le réseau, en numpy nu
class Reseau:
    """Un perceptron à deux couches cachées, entraîné par Adam. Rien
    d'exotique : on veut mesurer la STRUCTURE des poids, pas battre un
    record. Tout est en numpy pour rester lisible et reproductible."""

    def __init__(self, tailles=(64, 64, 32, 1), graine=0):
        rng = np.random.default_rng(graine)
        self.W, self.b = [], []
        for a, c in zip(tailles[:-1], tailles[1:]):
            self.W.append(rng.standard_normal((a, c)) * np.sqrt(2.0 / a))
            self.b.append(np.zeros(c))
        self.depart = [w.copy() for w in self.W]      # les poids AVANT tout
        self.hist = []

    def avant(self, X, garder=False):
        h = X
        actes = [h]
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            h = np.tanh(z) if k < len(self.W) - 1 else z
            actes.append(h)
        if garder:
            return h, actes
        return h

    def perte(self, X, y):
        return float(np.mean((self.avant(X) - y) ** 2))

    def entrainer_sgd(self, X, y, pas=6000, lr=0.2, wd=0.0):
        """Descente de gradient simple, avec PRESSION DE PARCIMONIE (wd).

        C'est la découverte de ce module : Adam, en normalisant chaque
        coordonnée, pousse partout également et produit des poids de rang
        PLEIN. La structure n'apparaît que sous une contrainte explicite
        qui rend le superflu coûteux."""
        n = X.shape[0]
        for t in range(pas):
            out, a = self.avant(X, garder=True)
            if not np.isfinite(out).all():
                return self, False           # divergence : on le dit
            d = 2.0 * (out - y) / n
            gW = [None] * len(self.W); gb = [None] * len(self.W)
            for k in range(len(self.W) - 1, -1, -1):
                gW[k] = a[k].T @ d; gb[k] = d.sum(0)
                if k > 0:
                    d = (d @ self.W[k].T) * (1 - a[k] ** 2)
            for k in range(len(self.W)):
                self.W[k] -= lr * (gW[k] + wd * self.W[k])
                self.b[k] -= lr * gb[k]
        return self, True

    def entrainer(self, X, y, pas=250, lr=6e-3, graine=0):
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, e = 0.9, 0.999, 1e-8
        n = X.shape[0]
        for t in range(1, pas + 1):
            out, actes = self.avant(X, garder=True)
            d = 2.0 * (out - y) / n
            gW, gb = [None] * len(self.W), [None] * len(self.W)
            for k in range(len(self.W) - 1, -1, -1):
                gW[k] = actes[k].T @ d
                gb[k] = d.sum(0)
                if k > 0:
                    d = (d @ self.W[k].T) * (1 - actes[k] ** 2)
            for k in range(len(self.W)):
                mW[k] = b1 * mW[k] + (1 - b1) * gW[k]
                vW[k] = b2 * vW[k] + (1 - b2) * gW[k] ** 2
                mb[k] = b1 * mb[k] + (1 - b1) * gb[k]
                vb[k] = b2 * vb[k] + (1 - b2) * gb[k] ** 2
                self.W[k] -= lr * (mW[k] / (1 - b1 ** t)) / (np.sqrt(vW[k] / (1 - b2 ** t)) + e)
                self.b[k] -= lr * (mb[k] / (1 - b1 ** t)) / (np.sqrt(vb[k] / (1 - b2 ** t)) + e)
            if t % 25 == 0 or t == 1:
                self.hist.append((t, self.perte(X, y)))
        return self


# ══════════════════════════════════ le X de Δ, appliqué à une couche
def spectre(W):
    return np.linalg.svd(np.asarray(W), compute_uv=False)


def X_effectif(W, eps=1e-2):
    """Combien de directions faut-il garder pour ne perdre qu'une part eps
    de l'énergie ? C'est le X de Δ8, mot pour mot, appliqué aux poids."""
    s = spectre(W)
    tot = float((s ** 2).sum())
    cum = np.cumsum(s[::-1] ** 2)[::-1]
    for k in range(1, len(s) + 1):
        reste = cum[k] if k < len(cum) else 0.0
        if reste / tot <= eps ** 2:
            return k
    return len(s)


def entropie_spectre(W):
    """Entropie du spectre normalisé : la mesure d'IMBRICATION de la couche.
    Basse = quelques directions dominent. Haute = tout compte pareil, donc
    rien n'est compressible. C'est l'entropie d'intrication de Δ, transposée."""
    s = spectre(W) ** 2
    s = s / s.sum()
    s = s[s > 0]
    return float(-(s * np.log(s)).sum())


def rang_participatif(W):
    """exp(entropie) : le nombre EFFECTIF de directions, sans seuil à choisir.
    Vaut 1 si une seule direction porte tout, et le rang plein si elles se
    valent toutes."""
    return float(np.exp(entropie_spectre(W)))


def tronquer(W, k):
    u, s, vh = np.linalg.svd(np.asarray(W), full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k, :]


# ═══════════════════════ le réseau a-t-il trouvé LES BONNES directions ?
def alignement(W, U):
    """Angles principaux entre l'espace appris et l'espace VRAI.
    On ne demande pas au réseau de retrouver les directions une à une —
    seulement le bon sous-espace. 1,0 = alignement parfait."""
    r = U.shape[1]
    u, _, _ = np.linalg.svd(np.asarray(W), full_matrices=False)
    A = u[:, :r]
    c = np.linalg.svd(A.T @ U, compute_uv=False)
    return float(np.mean(np.clip(c, 0, 1))), [float(x) for x in c]
