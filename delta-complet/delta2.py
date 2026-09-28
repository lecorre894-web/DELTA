#!/usr/bin/env python3
"""Δ2 V0.2 — intrication faible, liens χ, calcul par lots, complex64.

Évolution de Δ1 :
  Δ1 coupait un îlot seulement au RANG 1 (état produit exact ou approché).
     -> sur de grands registres la fidélité s'effondrait (0.62 à 1000 qubits).
  Δ2 garde χ valeurs de Schmidt sur chaque LIEN entre qubits voisins.
     χ = 1  : c'est Δ1 (individualisme pur)
     χ = 2,4,8... : intrication faible gardée, au prix de χ² par site
  C'est la forme de Vidal (Γ, λ) d'un réseau de tenseurs 1D (MPS / TEBD).

Canon Δ :
  - μ_j (multiplicateur de budget) = χ du lien : R_j = R_0 μ_j
  - ε / budget d'erreur : poids de Schmidt jeté à chaque troncature, comptabilisé
  - parallélisme Δ : toutes les portes d'une couche sur des liens disjoints
    sont appliquées EN UN SEUL LOT (einsum + SVD batchée), pas une par une
  - complex64 : 8 octets par nombre au lieu de 16

Limite honnête : une géométrie 1D à voisins proches. Une intrication forte ou
longue portée fait exploser χ requis (χ ~ 2^(n/2) au pire) : on retrouve le mur.

Dépendances : numpy. Compatible Termux.
"""
import math, time
import numpy as np

CT = np.complex64
RT = np.float32


def rx_batch(th):
    c = np.cos(th / 2).astype(RT); s = np.sin(th / 2).astype(RT)
    g = np.zeros((len(th), 2, 2), CT)
    g[:, 0, 0] = c; g[:, 1, 1] = c; g[:, 0, 1] = -1j * s; g[:, 1, 0] = -1j * s
    return g


def zz_batch(th):
    """CNOT · (I ⊗ rz(θ)) · CNOT = exp(-i θ/2 Z⊗Z) : une porte d'intrication réglable."""
    ph = np.exp(-0.5j * th).astype(CT)
    d = np.stack([ph, ph.conj(), ph.conj(), ph], 1)   # |00>,|01>,|10>,|11>
    g = np.zeros((len(th), 4, 4), CT)
    idx = np.arange(4)
    g[:, idx, idx] = d
    return g.reshape(len(th), 2, 2, 2, 2)


class Delta2:
    def __init__(self, n, chi=4):
        self.n, self.chi = n, chi
        self.G = np.zeros((n, chi, 2, chi), CT)
        self.G[:, 0, 0, 0] = 1                              # |0...0>
        self.L = np.zeros((n + 1, chi), RT)
        self.L[:, 0] = 1
        self.logfid = 0.0                                   # log de la fidélité estimée
        self.gates = 0
        self.lost_max = 0.0

    def memory(self):
        return self.G.nbytes + self.L.nbytes

    def layer1(self, gates):
        """Une porte 1-qubit sur chaque site, en un seul lot."""
        self.G = np.einsum('bst,bltr->blsr', gates, self.G, optimize=True)
        self.gates += len(gates)

    def layer2(self, first, gates):
        """Portes 2-qubits sur les liens (i, i+1), i = first, first+2, ... EN LOT."""
        i = np.arange(first, self.n - 1, 2)
        if len(i) == 0:
            return
        chi = self.chi
        Ga, Gb = self.G[i], self.G[i + 1]
        La, Lm, Lb = self.L[i], self.L[i + 1], self.L[i + 2]
        th = np.einsum('bl,blsm,bm,bmtr,br->blstr', La, Ga, Lm, Gb, Lb, optimize=True)
        th = np.einsum('bstuv,bluvr->blstr', gates[: len(i)], th, optimize=True)
        M = th.reshape(len(i), 2 * chi, 2 * chi)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        tot = (S.astype(np.float64) ** 2).sum(1)
        keep = (S[:, :chi].astype(np.float64) ** 2).sum(1)
        lost = np.clip(1.0 - keep / np.maximum(tot, 1e-300), 0.0, 1.0)
        self.logfid += float(np.log1p(-np.minimum(lost, 1 - 1e-15)).sum())
        self.lost_max = max(self.lost_max, float(lost.max()))
        Sk = S[:, :chi] / np.sqrt(np.maximum(keep, 1e-300))[:, None].astype(RT)
        U = U[:, :, :chi].reshape(len(i), chi, 2, chi)
        Vh = Vh[:, :chi, :].reshape(len(i), chi, 2, chi)
        inv = lambda x: np.where(x > 1e-6, 1.0 / np.maximum(x, 1e-30), 0.0).astype(RT)
        self.G[i] = U * inv(La)[:, :, None, None]
        self.G[i + 1] = Vh * inv(Lb)[:, None, None, :]
        self.L[i + 1] = Sk.astype(RT)
        self.gates += len(i)

    @property
    def fid(self):
        return math.exp(self.logfid)

    def dense(self):
        """Contracte le réseau en vecteur d'état. Contrôle, petits n uniquement."""
        psi = np.zeros((1, self.chi), np.complex128)       # (états, lien gauche)
        psi[0, 0] = 1
        for s in range(self.n):
            A = self.G[s].astype(np.complex128) * self.L[s].astype(np.float64)[:, None, None]
            psi = np.einsum('pl,lsr->psr', psi, A).reshape(-1, self.chi)
        return psi[:, 0]


def circuit_faible(n, depth, seed, theta):
    """Mêmes portes que c_faible de Δ1, générées couche par couche."""
    rng = np.random.default_rng(seed)
    layers = []
    for d in range(depth):
        a1 = rng.uniform(-theta, theta, n).astype(RT)
        first = d % 2
        nb = len(range(first, n - 1, 2))
        a2 = rng.uniform(-theta, theta, nb).astype(RT)
        layers.append((a1, first, a2))
    return layers


def run2(sim, layers):
    for a1, first, a2 in layers:
        sim.layer1(rx_batch(a1))
        sim.layer2(first, zz_batch(a2))
    return sim


def dense_reference(n, layers):
    """Simulation exacte 2^n du même circuit (vérité terrain)."""
    t = np.zeros((2,) * n, np.complex128); t[(0,) * n] = 1
    for a1, first, a2 in layers:
        g1 = rx_batch(a1).astype(np.complex128)
        for q in range(n):
            t = np.moveaxis(np.tensordot(g1[q], t, axes=([1], [q])), 0, q)
        g2 = zz_batch(a2).astype(np.complex128)
        for k, q in enumerate(range(first, n - 1, 2)):
            t = np.tensordot(g2[k], t, axes=([2, 3], [q, q + 1]))
            t = np.moveaxis(t, [0, 1], [q, q + 1])
    return t.reshape(-1)
