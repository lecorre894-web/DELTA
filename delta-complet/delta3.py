#!/usr/bin/env python3
"""Δ3 V0.3 — χ PROPORTIONNÉ par lien (μ_j) + traitement par TRANCHES.

  - chaque lien j reçoit son propre budget χ_j = f(μ_j) :
        liens autour des qubits importants (P = 100 %) : χ haut
        liens ailleurs                                  : χ bas
  - calcul en blocs de taille χ_max, mais troncature réelle à χ_j par lien :
    l'effet sur la fidélité est donc exact
  - mémoire rapportée = STOCKAGE COMPACT, somme exacte des tenseurs à leur
    vraie taille χ_gauche · 2 · χ_droite (ce que prendrait un rangement serré)
  - tranches : une couche est traitée par paquets de liens (pavage) ;
    la mémoire de travail reste bornée quelle que soit la taille du registre.
    C'est ce qui rend l'émulation extensible au-delà de la RAM : le réseau est
    local, une tranche ne dépend que de ses voisines immédiates.

Dépendances : numpy. Compatible Termux.
"""
import math
import numpy as np
from delta2 import Delta2, CT, RT, rx_batch, zz_batch


class Delta3(Delta2):
    def __init__(self, n, chi_bond, tranche=200_000):
        chi_bond = np.asarray(chi_bond, dtype=np.int32)      # longueur n+1
        super().__init__(n, int(chi_bond.max()))
        self.chi_bond = chi_bond
        self.tranche = tranche

    def memory_compacte(self):
        cl = self.chi_bond[:-1].astype(np.int64)
        cr = self.chi_bond[1:].astype(np.int64)
        return int((cl * 2 * cr).sum() * 8 + self.chi_bond.sum() * 4)

    def layer1(self, gates):
        for s in range(0, self.n, self.tranche):
            e = min(self.n, s + self.tranche)
            self.G[s:e] = np.einsum('bst,bltr->blsr', gates[s:e], self.G[s:e], optimize=True)
        self.gates += len(gates)

    def layer2(self, first, gates):
        bonds = np.arange(first, self.n - 1, 2)
        chi = self.chi
        inv = lambda x: np.where(x > 1e-6, 1.0 / np.maximum(x, 1e-30), 0.0).astype(RT)
        for s in range(0, len(bonds), self.tranche):
            i = bonds[s:s + self.tranche]
            g = gates[s:s + self.tranche]
            Ga, Gb = self.G[i], self.G[i + 1]
            La, Lm, Lb = self.L[i], self.L[i + 1], self.L[i + 2]
            th = np.einsum('bl,blsm,bm,bmtr,br->blstr', La, Ga, Lm, Gb, Lb, optimize=True)
            th = np.einsum('bstuv,bluvr->blstr', g, th, optimize=True)
            U, S, Vh = np.linalg.svd(th.reshape(len(i), 2 * chi, 2 * chi), full_matrices=False)
            cj = self.chi_bond[i + 1]                              # budget du lien
            mask = (np.arange(2 * chi)[None, :] < cj[:, None]).astype(RT)
            S2 = S.astype(np.float64) ** 2
            tot = S2.sum(1)
            keep = (S2 * mask).sum(1)
            lost = np.clip(1.0 - keep / np.maximum(tot, 1e-300), 0.0, 1.0)
            self.logfid += float(np.log1p(-np.minimum(lost, 1 - 1e-15)).sum())
            self.lost_max = max(self.lost_max, float(lost.max()))
            Sk = (S * mask)[:, :chi] / np.sqrt(np.maximum(keep, 1e-300))[:, None].astype(RT)
            U = (U * mask[:, None, :])[:, :, :chi].reshape(len(i), chi, 2, chi)
            Vh = (Vh * mask[:, :, None])[:, :chi, :].reshape(len(i), chi, 2, chi)
            self.G[i] = U * inv(La)[:, :, None, None]
            self.G[i + 1] = Vh * inv(Lb)[:, None, None, :]
            self.L[i + 1] = Sk.astype(RT)
        self.gates += len(bonds)


def budget_mu(n, importants, chi_haut, chi_bas, rayon=2):
    """μ_j -> χ_j : χ haut sur les liens à moins de `rayon` d'un qubit important."""
    chi = np.full(n + 1, chi_bas, np.int32)
    for q in importants:
        a, b = max(1, q - rayon + 1), min(n - 1, q + rayon)
        chi[a:b + 1] = chi_haut
    chi[0] = chi[n] = 1
    return chi


def reduit(v, n, q):
    t = v.reshape((2,) * n)
    M = np.moveaxis(t, q, 0).reshape(2, -1)
    return M @ M.conj().T


def fid1(r, s):
    """Fidélité entre deux états d'un qubit (matrices 2x2)."""
    return float(np.real(np.trace(r @ s)) +
                 2 * math.sqrt(max(np.linalg.det(r).real, 0) * max(np.linalg.det(s).real, 0)))
