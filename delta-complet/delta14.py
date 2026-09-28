#!/usr/bin/env python3
"""Δ14 V1.4 — GRAPPES ET NŒUDS EN Δ-FLOPS : fond infini + grappes allumées.

Transposition exacte de Δ5 (chaîne infinie + grappes individuelles) au calcul :

  qubit          -> bloc de calcul (b x b)
  fond infini    -> opérateur INVARIANT PAR TRANSLATION : tous les blocs situés
                    à la même distance de la diagonale sont IDENTIQUES, donc
                    stockés UNE SEULE FOIS (un seul jeu U_c V_c par distance c)
  grappe allumée -> une particularité locale du problème (défaut, source,
                    obstacle) : seule celle-là est stockée individuellement
  X              -> rang de la classe de distance (à la demande, par epsilon)
  cône           -> on ne calcule que les blocs de sortie demandés
  ECC + nœuds    -> blocs répartis, empreintes, parité

Conséquence, exactement comme en Δ5 : la mémoire ne dépend PAS de la taille du
problème, et le coût d'une question locale est CONSTANT, que l'opérateur porte
sur mille inconnues ou sur mille milliards.

Le vecteur d'entrée lui-même n'est jamais matérialisé : il est fourni à la
demande, tranche par tranche (fournisseur procédural).

Dépendances : numpy. Compatible Termux.
"""
import math, time
import numpy as np


class FondInfini:
    """Opérateur invariant par translation, compressé par classe de distance."""

    def __init__(self, noyau, b=128, eps=1e-8, portee_max=64):
        self.noyau, self.b, self.eps = noyau, b, eps
        self.classes = {}          # distance en blocs -> (U, V) ou None si négligeable
        self.rangs = {}
        i = np.arange(b)
        ref = None
        for c in range(0, portee_max + 1):
            M = noyau(i[:, None] - (i[None, :] + c * b))
            nrm = np.linalg.norm(M)
            if ref is None:
                ref = nrm
            if nrm <= eps * ref:                 # au-delà, la contribution est nulle
                break
            u, s, vh = np.linalg.svd(M, full_matrices=False)
            tot = float((s ** 2).sum())
            cum = np.cumsum(s[::-1] ** 2)[::-1]
            r = len(s)
            for k in range(1, len(s) + 1):
                if (cum[k] if k < len(cum) else 0.0) / tot <= eps ** 2:
                    r = k
                    break
            self.classes[c] = (u[:, :r] * s[:r], vh[:r, :])
            self.rangs[c] = r
            if c > 0:                            # la classe -c est la transposée du noyau
                M2 = noyau(i[:, None] - (i[None, :] - c * b))
                u2, s2, vh2 = np.linalg.svd(M2, full_matrices=False)
                tot2 = float((s2 ** 2).sum()); cum2 = np.cumsum(s2[::-1] ** 2)[::-1]
                r2 = len(s2)
                for k in range(1, len(s2) + 1):
                    if (cum2[k] if k < len(cum2) else 0.0) / tot2 <= eps ** 2:
                        r2 = k
                        break
                self.classes[-c] = (u2[:, :r2] * s2[:r2], vh2[:r2, :])
                self.rangs[-c] = r2
        self.portee = max(abs(c) for c in self.classes)

    def octets(self):
        return sum(U.size + V.size for U, V in self.classes.values()) * 8

    def bloc_sortie(self, k, fournisseur):
        """Calcule le bloc de sortie n° k. Coût CONSTANT, indépendant de la
        taille du problème : seules les classes de distance comptent."""
        y = np.zeros(self.b)
        flops = 0
        for c, (U, V) in self.classes.items():
            v = fournisseur(k + c)               # tranche d'entrée, à la demande
            t = V @ v
            y += U @ t
            flops += 2 * (V.size + U.size)
        return y, flops


class Grappe:
    """Particularité locale : un bloc de correction allumé à une position."""
    def __init__(self, position, correction):
        self.position = position
        self.M = correction

    def octets(self):
        return self.M.size * 8


class DeltaFlops:
    """Fond infini + grappes allumées + répartition sur nœuds."""

    def __init__(self, fond):
        self.fond = fond
        self.grappes = {}
        self.flops = 0
        self.temps = 0.0
        self.ops = 0

    def allumer(self, position, correction):
        self.grappes[position] = Grappe(position, correction)
        return self.grappes[position].octets()

    def octets(self):
        return self.fond.octets() + sum(g.octets() for g in self.grappes.values())

    def bloc(self, k, fournisseur):
        t0 = time.time()
        y, f = self.fond.bloc_sortie(k, fournisseur)
        g = self.grappes.get(k)
        if g is not None:                        # la grappe corrige son bloc
            y = y + g.M @ fournisseur(k)
            f += 2 * g.M.size
        self.flops += f
        self.ops += 1
        self.temps += time.time() - t0
        return y, f

    def gflops(self):
        return self.flops / max(self.temps, 1e-12) / 1e9

    def ops_par_seconde(self):
        return self.ops / max(self.temps, 1e-12)

    def repartir(self, nb_noeuds):
        """Répartit les classes du fond et les grappes sur les nœuds + parité."""
        elements = [('classe', c) for c in sorted(self.fond.classes)] + \
                   [('grappe', p) for p in sorted(self.grappes)]
        noeuds = {k: [] for k in range(nb_noeuds)}
        for i, e in enumerate(elements):
            noeuds[i % nb_noeuds].append(e)
        flux = []
        for k, lst in noeuds.items():
            parts = []
            for typ, cle in lst:
                a = np.concatenate([x.ravel() for x in self.fond.classes[cle]]) if typ == 'classe' \
                    else self.grappes[cle].M.ravel()
                parts.append(np.ascontiguousarray(a).view(np.uint8))
            flux.append(np.concatenate(parts) if parts else np.zeros(0, np.uint8))
        taille = max(len(x) for x in flux)
        parite = np.zeros(taille, np.uint8)
        for x in flux:
            parite[:len(x)] ^= x
        return noeuds, parite


# ------------------------------------------------------------- utilitaires
def noyau_lisse(d, h=1e-3):
    return 1.0 / (1.0 + np.abs(d) * h)


def fournisseur_procedural(b, graine=0):
    """Le vecteur d'entrée n'existe pas en mémoire : il est calculé à la demande."""
    def f(k):
        i = np.arange(k * b, (k + 1) * b, dtype=np.float64)
        return np.sin(0.001 * i) + 0.5 * np.cos(0.0007 * i + 1.0)
    return f


def dense_reference(n, b, noyau, grappes=None):
    """Construit explicitement l'opérateur (petits n seulement) pour contrôle."""
    i = np.arange(n)
    M = noyau(i[:, None] - i[None, :])
    if grappes:
        for p, G in grappes.items():
            M[p * b:(p + 1) * b, p * b:(p + 1) * b] += G
    return M
