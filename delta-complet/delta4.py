#!/usr/bin/env python3
"""Δ4 V0.4 — la GRAPPE PARTAGÉE : une infinité de qubits en quelques Kio.

Idée (canon Δ, « chaque qubit déclenche une plateforme en grappe ») :
  si toutes les grappes sont identiques, on n'en stocke QU'UNE et toutes les
  autres la référencent. Poussé à la limite, c'est une chaîne INFINIE de qubits
  décrite par une cellule de 2 tenseurs (Γ_A, λ_A, Γ_B, λ_B) : iTEBD.

  Mémoire : constante, indépendante du nombre de qubits.
  Prix    : l'individualisme est suspendu. Toutes les grappes sont identiques ;
            un circuit doit agir pareil partout (invariance par translation).

Contrôle : on compare <Z> sur un qubit de la chaîne infinie à <Z> au CENTRE
d'une chaîne finie Δ2 de grande taille, soumise au même circuit uniforme.

Dépendances : numpy. Compatible Termux.
"""
import math
import numpy as np

C = np.complex128
Z = np.array([1.0, -1.0])


def rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], C)


def zz(t):
    ph = np.exp(-0.5j * t)
    return np.diag([ph, np.conj(ph), np.conj(ph), ph]).reshape(2, 2, 2, 2)


def circuit_uniforme(depth, seed, theta):
    """Mêmes angles sur tout le registre, différents à chaque couche."""
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-theta, theta), d % 2, rng.uniform(-theta, theta)) for d in range(depth)]


class Infini:
    """Chaîne infinie A-B-A-B-... en forme de Vidal. Deux sites stockés."""
    def __init__(self, chi):
        self.chi = chi
        self.G = [np.zeros((chi, 2, chi), C), np.zeros((chi, 2, chi), C)]
        for g in self.G:
            g[0, 0, 0] = 1
        self.L = [np.zeros(chi), np.zeros(chi)]      # L[0] : lien A|B ; L[1] : lien B|A
        self.L[0][0] = self.L[1][0] = 1
        self.logfid_site = 0.0                        # log de fidélité PAR SITE

    def memory(self):
        return sum(g.nbytes for g in self.G) + sum(l.nbytes for l in self.L)

    def g1(self, U):
        self.G = [np.einsum('st,ltr->lsr', U, g) for g in self.G]

    def g2(self, U, parite):
        """parite 0 : liens A-B ; parite 1 : liens B-A. Tous les liens identiques d'un coup."""
        a, b = (0, 1) if parite == 0 else (1, 0)
        chi = self.chi
        Lout, Lmid = self.L[1 - parite], self.L[parite]    # lien extérieur, lien central
        th = np.einsum('l,lsm,m,mtr,r->lstr', Lout, self.G[a], Lmid, self.G[b], Lout)
        th = np.einsum('stuv,luvr->lstr', U, th)
        M = th.reshape(2 * chi, 2 * chi)
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        tot = (s ** 2).sum()
        keep = (s[:chi] ** 2).sum()
        self.logfid_site += 0.5 * math.log(max(keep / tot, 1e-300))   # 1 lien pour 2 sites
        sk = s[:chi] / math.sqrt(keep)
        inv = np.where(Lout > 1e-8, 1.0 / np.maximum(Lout, 1e-30), 0.0)
        self.G[a] = u[:, :chi].reshape(chi, 2, chi) * inv[:, None, None]
        self.G[b] = vh[:chi, :].reshape(chi, 2, chi) * inv[None, None, :]
        self.L[parite] = sk

    def z_moyen(self):
        """<Z> sur le site A. Forme canonique : poids = λ_gauche² · λ_droite²."""
        Lg, Ld = self.L[1], self.L[0]
        w = np.einsum('l,lsr,r->s', Lg ** 2, np.abs(self.G[0]) ** 2, Ld ** 2)
        return float((w * Z).sum() / w.sum())


def run_infini(chi, layers):
    s = Infini(chi)
    for t1, par, t2 in layers:
        s.g1(rx(t1))
        s.g2(zz(t2), par)
    return s


def run_fini_centre(n, chi, layers):
    """Même circuit sur une chaîne finie (Δ2, complex128 pour le contrôle)."""
    import delta2 as D2
    D2.CT, D2.RT = np.complex128, np.float64
    s = D2.Delta2(n, chi)
    s.G = s.G.astype(np.complex128); s.L = s.L.astype(np.float64)
    for t1, par, t2 in layers:
        s.layer1(np.repeat(rx(t1)[None], n, 0))
        nb = len(range(par, n - 1, 2))
        s.layer2(par, np.repeat(zz(t2)[None], nb, 0))
    q = n // 2
    if q % 2 == 1:
        q -= 1                                          # un site « A » (pair)
    w = np.einsum('l,lsr,r->s', s.L[q] ** 2, np.abs(s.G[q]) ** 2, s.L[q + 1] ** 2)
    return float((w * Z).sum() / w.sum()), s.memory()
