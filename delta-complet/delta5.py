#!/usr/bin/env python3
"""Δ5 V0.5 — HYBRIDE : fond infini partagé + grappes individuelles allumées.

Architecture :
  - fond      : chaîne infinie uniforme (Δ4, iTEBD), 2 tenseurs, mémoire constante
  - grappes   : zones où chaque qubit reçoit ses PROPRES portes (individualisme)
  - coût      : seules les grappes allumées sont stockées, chacune sur un segment
                de taille W + 4d (W = largeur de la grappe, d = profondeur)

Garantie par le CÔNE DE LUMIÈRE (portes entre voisins, d couches) :
  - l'influence d'une grappe ne dépasse pas d sites de chaque côté
  - l'effet de bord d'un segment fini ne pénètre pas plus de d sites
  => un segment de W + 4d sites calcule EXACTEMENT la grappe et sa zone
     d'influence ; partout ailleurs, le fond infini est exact.
  => deux grappes séparées de plus de 2d sites sont indépendantes.

Limite : une porte reliant deux grappes lointaines (longue portée) casse la
garantie. Le mur de l'intrication forte reste le même à l'intérieur d'une grappe.

Dépendances : numpy. Compatible Termux.
"""
import math
import numpy as np
import delta4 as D4

Z = np.array([1.0, -1.0])


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], np.complex128)


class Chaine:
    """Chaîne finie en forme de Vidal, complex128, portes site par site possibles."""
    def __init__(self, n, chi):
        self.n, self.chi = n, chi
        self.G = np.zeros((n, chi, 2, chi), np.complex128); self.G[:, 0, 0, 0] = 1
        self.L = np.zeros((n + 1, chi)); self.L[:, 0] = 1
        self.logfid = 0.0

    def memory(self):
        return self.G.nbytes + self.L.nbytes

    def layer1(self, gates):
        self.G = np.einsum('bst,bltr->blsr', gates, self.G, optimize=True)

    def layer2(self, first, gate):
        i = np.arange(first, self.n - 1, 2)
        chi = self.chi
        th = np.einsum('bl,blsm,bm,bmtr,br->blstr', self.L[i], self.G[i], self.L[i + 1],
                       self.G[i + 1], self.L[i + 2], optimize=True)
        th = np.einsum('stuv,bluvr->blstr', gate, th, optimize=True)
        U, S, Vh = np.linalg.svd(th.reshape(len(i), 2 * chi, 2 * chi), full_matrices=False)
        tot = (S ** 2).sum(1); keep = (S[:, :chi] ** 2).sum(1)
        self.logfid += float(np.log(np.maximum(keep / tot, 1e-300)).sum())
        Sk = S[:, :chi] / np.sqrt(keep)[:, None]
        inv = lambda x: np.where(x > 1e-8, 1.0 / np.maximum(x, 1e-30), 0.0)
        self.G[i] = U[:, :, :chi].reshape(len(i), chi, 2, chi) * inv(self.L[i])[:, :, None, None]
        self.G[i + 1] = Vh[:, :chi, :].reshape(len(i), chi, 2, chi) * inv(self.L[i + 2])[:, None, None, :]
        self.L[i + 1] = Sk

    def z_profil(self):
        w = np.einsum('bl,blsr,br->bs', self.L[:-1] ** 2, np.abs(self.G) ** 2, self.L[1:] ** 2)
        return (w * Z).sum(1) / w.sum(1)


def run_chaine(n, chi, layers, portes_indiv, origine):
    """Circuit uniforme + portes individuelles. `portes_indiv` : {(couche, site_absolu): U}.
    `origine` : position absolue du site 0 de la chaîne (doit être paire pour garder A/B)."""
    s = Chaine(n, chi)
    for c, (t1, par, t2) in enumerate(layers):
        g = np.repeat(D4.rx(t1)[None], n, 0)
        for (cc, site), U in portes_indiv.items():
            k = site - origine
            if cc == c and 0 <= k < n:
                g[k] = U @ g[k]
        s.layer1(g)
        s.layer2((par - origine) % 2, D4.zz(t2))
    return s


class Hybride:
    def __init__(self, chi, layers):
        self.chi, self.layers = chi, layers
        self.d = len(layers)
        self.fond = D4.run_infini(chi, layers)
        z_A = self.fond.z_moyen()
        # site B : même calcul avec les liens inversés
        f = self.fond
        w = np.einsum('l,lsr,r->s', f.L[0] ** 2, np.abs(f.G[1]) ** 2, f.L[1] ** 2)
        self.z_fond = (z_A, float((w * Z).sum() / w.sum()))
        self.grappes = []            # (début absolu, fin absolue, profil exact, origine)

    def allumer(self, debut, largeur, portes_indiv):
        """Allume une grappe individuelle sur [debut, debut+largeur)."""
        m = 2 * self.d
        origine = debut - m
        origine -= origine % 2                     # alignement A/B
        n = (debut + largeur + m) - origine
        n += n % 2
        s = run_chaine(n, self.chi, self.layers, portes_indiv, origine)
        prof = s.z_profil()
        # zone exacte : à plus de d sites des bords du segment
        lo, hi = origine + self.d, origine + n - self.d
        self.grappes.append((lo, hi, prof[self.d:n - self.d], s.memory()))
        return s.memory()

    def z(self, site):
        for lo, hi, prof, _ in self.grappes:
            if lo <= site < hi:
                return float(prof[site - lo])
        return self.z_fond[site % 2]

    def memory(self):
        return self.fond.memory() + sum(g[3] for g in self.grappes)
