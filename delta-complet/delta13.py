#!/usr/bin/env python3
"""Δ13 V1.3 — Δ-FLOPS : l'architecture Δ appliquée au calcul, pas aux qubits.

L'unité n'est plus le qubit mais le BLOC DE DONNÉES. Tout le reste se traduit :

  intrication  -> dépendance entre blocs
  X            -> RANG du bloc : un bloc lisse se factorise en U(m,r) V(r,n)
  epsilon      -> erreur admise sur le bloc (le rang en découle, à la demande)
  % (P)        -> PRÉCISION du bloc : float64 / float32 / float16
  mu           -> budget accordé au bloc
  cône         -> on ne calcule que les blocs de sortie réellement demandés
  symétrie     -> blocs nuls par construction, jamais stockés ni multipliés
  ECC          -> empreinte par bloc, parité entre nœuds

Un produit matriciel dense coûte 2 n^3 flops. En Δ-FLOPS, un bloc de rang r
coûte 2 r (m + n) par vecteur, et deux blocs de rang r se multiplient en
O(r^2 (m+n)) au lieu de O(m n k). Le gain est un gain d'EXPOSANT quand les
données ont une structure, et nul quand elles n'en ont pas — c'est dit.

Dépendances : numpy. Compatible Termux.
"""
import math, time, zlib
import numpy as np

PRECISIONS = {1.00: np.float64, 0.50: np.float32, 0.10: np.float16}


def precision_de(P):
    """% -> type de flottant. 100 % = float64, 50 % = float32, 10 % = float16."""
    for seuil in sorted(PRECISIONS, reverse=True):
        if P >= seuil:
            return PRECISIONS[seuil]
    return np.float16


class Bloc:
    """Un bloc : soit plein, soit factorisé U V (rang X), avec sa précision."""
    __slots__ = ('U', 'V', 'plein', 'X', 'dtype', 'forme', 'crc')

    def __init__(self, M, eps=1e-8, P=1.0, rang_max=None):
        dt = precision_de(P)
        m, n = M.shape
        self.forme = (m, n)
        self.dtype = dt
        rmax = min(m, n) if rang_max is None else min(rang_max, m, n)
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        tot = float((s ** 2).sum()) + 1e-300
        cum = np.cumsum(s[::-1] ** 2)[::-1]
        X = len(s)
        for k in range(1, len(s) + 1):
            if (cum[k] if k < len(cum) else 0.0) / tot <= eps ** 2:
                X = k
                break
        X = min(X, rmax)
        # on ne factorise que si c'est RENTABLE : X (m+n) < m n
        if X * (m + n) < m * n:
            self.U = (u[:, :X] * s[:X]).astype(dt)
            self.V = vh[:X, :].astype(dt)
            self.plein = None
            self.X = X
        else:
            self.plein = M.astype(dt)
            self.U = self.V = None
            self.X = min(m, n)
        self.crc = self.empreinte()

    def octets(self):
        t = np.dtype(self.dtype).itemsize
        return (self.plein.size if self.plein is not None else self.U.size + self.V.size) * t

    def empreinte(self):
        a = self.plein if self.plein is not None else np.concatenate([self.U.ravel(), self.V.ravel()])
        return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).tobytes()) & 0xFFFFFFFF

    def verifier(self):
        return self.empreinte() == self.crc

    def matvec(self, v):
        """Renvoie (résultat, flops réellement dépensés)."""
        if self.plein is not None:
            return self.plein.astype(np.float64) @ v, 2 * self.plein.size
        t = self.V.astype(np.float64) @ v
        return self.U.astype(np.float64) @ t, 2 * (self.V.size + self.U.size)

    def dense(self):
        return (self.plein if self.plein is not None else self.U @ self.V).astype(np.float64)


class MatriceDelta:
    """Matrice découpée en blocs, chacun compressé à la demande."""

    def __init__(self, M, taille=256, eps=1e-8, P=None, nul_tol=0.0):
        n, m = M.shape
        self.n, self.m, self.taille = n, m, taille
        self.li = list(range(0, n, taille)) + [n]
        self.co = list(range(0, m, taille)) + [m]
        self.blocs = {}
        self.nuls = 0
        self.t_construction = 0.0
        t0 = time.time()
        for a in range(len(self.li) - 1):
            for b in range(len(self.co) - 1):
                S = M[self.li[a]:self.li[a + 1], self.co[b]:self.co[b + 1]]
                if nul_tol > 0 and np.abs(S).max() <= nul_tol:
                    self.nuls += 1               # bloc nul : jamais stocké (symétrie)
                    continue
                p = 1.0 if P is None else float(P[a][b])
                self.blocs[(a, b)] = Bloc(S, eps=eps, P=p)
        self.t_construction = time.time() - t0

    def octets(self):
        return sum(b.octets() for b in self.blocs.values())

    def octets_dense(self):
        return self.n * self.m * 8

    def rangs(self):
        return [b.X for b in self.blocs.values()]

    def matvec(self, v, lignes=None):
        """Produit matrice-vecteur. `lignes` : ne calculer QUE ces blocs de
        sortie (cône de lumière : on ne réveille que ce qui est demandé)."""
        y = np.zeros(self.n)
        flops = 0
        for (a, b), bloc in self.blocs.items():
            if lignes is not None and a not in lignes:
                continue
            r, f = bloc.matvec(v[self.co[b]:self.co[b + 1]])
            y[self.li[a]:self.li[a + 1]] += r
            flops += f
        return y, flops

    def verifier_tout(self):
        return all(b.verifier() for b in self.blocs.values())

    def repartir(self, nb_noeuds):
        """Répartit les blocs sur des nœuds + parité XOR (ECC)."""
        cles = sorted(self.blocs)
        noeuds = {k: [] for k in range(nb_noeuds)}
        for i, c in enumerate(cles):
            noeuds[i % nb_noeuds].append(c)
        flux = {k: np.concatenate([np.frombuffer(
            np.ascontiguousarray(self.blocs[c].dense()).tobytes(), dtype=np.uint8) for c in v])
            for k, v in noeuds.items() if v}
        taille = max(len(x) for x in flux.values())
        parite = np.zeros(taille, np.uint8)
        for x in flux.values():
            parite[:len(x)] ^= x
        return noeuds, parite


def kernel_lisse(n, seed=0, bruit=0.0):
    """Matrice de noyau lisse : structure typique des problèmes physiques
    (interaction, diffusion). Ses blocs hors diagonale sont de rang faible."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n) + 1.5
    K = 1.0 / (0.1 + np.abs(x[:, None] - y[None, :]))
    if bruit > 0:
        K = K + bruit * rng.standard_normal((n, n))
    return K
