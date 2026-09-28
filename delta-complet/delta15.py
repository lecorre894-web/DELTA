#!/usr/bin/env python3
"""Δ15 V1.5 — Δ-OMNI : toutes les stratégies fusionnées, traitement par LOTS.

Fusion de tout ce qui a été mesuré depuis Δ1 :

  fond infini partagé (Δ5, Δ14)   : un seul jeu de tenseurs pour une infinité
  grappes allumées (Δ5)            : l'individualité ne coûte que là où elle est
  X à la demande (Δ8)              : le rang de chaque classe suit epsilon
  % = précision (Δ13)              : classes lointaines en float32, cœur en float64
  symétrie / vide (Δ10)            : les classes négligeables n'existent pas
  cône de lumière (Δ11)            : on ne calcule que ce qui est demandé
  ECC (Δ9)                         : empreinte par élément, parité entre nœuds
  TRAITEMENT PAR LOTS (nouveau)    : tous les blocs de sortie d'un coup, en deux
                                     grands produits matriciels par classe, au
                                     lieu d'une nuée de petites multiplications.

Le lot est la seule vraie nouveauté : même nombre de flops, mais confiés à BLAS
en gros blocs. C'est ce qui transforme un débit de 0,35 Gflops en plusieurs
Gflops, sans changer un seul résultat.

Dépendances : numpy. Compatible Termux.
"""
import math, time, zlib
import numpy as np


class Omni:
    def __init__(self, noyau, b=128, eps=1e-8, portee_max=64, seuil_float32=1e-4):
        self.noyau, self.b, self.eps = noyau, b, eps
        self.U, self.V, self.rangs, self.dtypes = {}, {}, {}, {}
        self.grappes = {}
        self.flops = self.ops = 0
        self.temps = 0.0
        self.ignorees = 0
        i = np.arange(b)
        ref = None
        for c in list(range(0, portee_max + 1)) + list(range(-1, -portee_max - 1, -1)):
            M = noyau(i[:, None] - (i[None, :] + c * b))
            nrm = float(np.linalg.norm(M))
            if ref is None:
                ref = nrm
            if nrm <= eps * ref:                       # SYMÉTRIE / VIDE : classe inexistante
                self.ignorees += 1
                continue
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            tot = float((s ** 2).sum()); cum = np.cumsum(s[::-1] ** 2)[::-1]
            r = len(s)
            for k in range(1, len(s) + 1):            # X À LA DEMANDE
                if (cum[k] if k < len(cum) else 0.0) / tot <= eps ** 2:
                    r = k
                    break
            # % : une classe faible devant le cœur n'a pas besoin de float64
            dt = np.float64 if nrm > seuil_float32 * ref else np.float32
            self.U[c] = (u[:, :r] * s[:r]).astype(dt)
            self.V[c] = vh[:r, :].astype(dt)
            self.rangs[c] = r
            self.dtypes[c] = dt
        self.portee = max(abs(c) for c in self.U)
        self.crcs = {c: self._crc(c) for c in self.U}

    # ------------------------------------------------------------------ ECC
    def _crc(self, c):
        a = np.concatenate([self.U[c].ravel().astype(np.float64),
                            self.V[c].ravel().astype(np.float64)])
        return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).tobytes()) & 0xFFFFFFFF

    def verifier(self):
        return all(self._crc(c) == self.crcs[c] for c in self.U)

    def octets(self):
        return (sum(self.U[c].nbytes + self.V[c].nbytes for c in self.U)
                + sum(g.nbytes for g in self.grappes.values()))

    def allumer(self, position, correction):
        self.grappes[int(position)] = correction
        return correction.nbytes

    # ------------------------------------------------------------ le LOT
    def lot(self, k0, nk, fournisseur, sorties=None):
        """Si `sorties` est donné, on découpe la demande en plages contiguës et
        on ne calcule QUE celles-là (le cône coupe le CALCUL, pas la sortie)."""
        if sorties is not None:
            garde = sorted(set(int(k) for k in sorties))
            plages, debut, prec = [], garde[0], garde[0]
            for k in garde[1:]:
                if k == prec + 1:
                    prec = k
                else:
                    plages.append((debut, prec - debut + 1)); debut = prec = k
            plages.append((debut, prec - debut + 1))
            morceaux = [self._lot(a, l, fournisseur)[0] for a, l in plages]
            return np.concatenate(morceaux, axis=1), 0
        return self._lot(k0, nk, fournisseur)

    def _lot(self, k0, nk, fournisseur):
        """Calcule d'un coup les blocs de sortie k0 .. k0+nk-1.

        `sorties` : sous-ensemble demandé (cône de lumière). Le reste n'est
        jamais calculé.
        """
        t0 = time.time()
        p = self.portee
        # 1. on rassemble les tranches d'entrée nécessaires, une seule fois
        cols = np.empty((self.b, nk + 2 * p), np.float64)
        for j, k in enumerate(range(k0 - p, k0 + nk + p)):
            cols[:, j] = fournisseur(k)
        Y = np.zeros((self.b, nk))
        flops = 0
        # 2. deux GRANDS produits par classe, au lieu de nk petits
        for c in self.U:
            bloc = cols[:, p + c: p + c + nk]                 # entrées décalées
            Vc, Uc = self.V[c], self.U[c]
            T = Vc @ bloc.astype(Vc.dtype, copy=False)        # (r, nk)
            Y += (Uc @ T).astype(np.float64, copy=False)      # (b, nk)
            flops += 2 * Vc.shape[0] * Vc.shape[1] * nk + 2 * Uc.shape[0] * Uc.shape[1] * nk
        # 3. les grappes allumées corrigent leurs propres blocs
        for k, G in self.grappes.items():
            if k0 <= k < k0 + nk:
                Y[:, k - k0] += G @ cols[:, p + (k - k0)]
                flops += 2 * G.size
        self.flops += flops
        self.ops += nk
        self.temps += time.time() - t0
        return Y, flops

    def gflops(self):
        return self.flops / max(self.temps, 1e-12) / 1e9

    def blocs_par_seconde(self):
        return self.ops / max(self.temps, 1e-12)

    def inconnues_par_seconde(self):
        return self.blocs_par_seconde() * self.b

    # -------------------------------------------------------------- nœuds
    def repartir(self, nb_noeuds):
        """Répartit les classes et grappes à l'OCTET : parité = 1/N exactement."""
        flux = np.concatenate([np.ascontiguousarray(x).view(np.uint8).ravel()
                               for c in sorted(self.U)
                               for x in (self.U[c], self.V[c])] +
                              [np.ascontiguousarray(self.grappes[k]).view(np.uint8).ravel()
                               for k in sorted(self.grappes)])
        coupes = np.linspace(0, len(flux), nb_noeuds + 1).astype(int)
        noeuds = {k: flux[coupes[k]:coupes[k + 1]].copy() for k in range(nb_noeuds)}
        taille = max(len(v) for v in noeuds.values())
        parite = np.zeros(taille, np.uint8)
        for v in noeuds.values():
            parite[:len(v)] ^= v
        return noeuds, parite


def noyau_lisse(d, h=1e-3):
    return 1.0 / (1.0 + np.abs(d) * h)


def fournisseur(b):
    def f(k):
        i = np.arange(k * b, (k + 1) * b, dtype=np.float64)
        return np.sin(0.001 * i) + 0.5 * np.cos(0.0007 * i + 1.0)
    return f
