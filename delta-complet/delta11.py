#!/usr/bin/env python3
"""Δ11 V1.1 — TRAVAIL À LA DEMANDE : on ne réveille que ce qui sert.

Trois mécanismes, tous mesurés :

1) PORTES INERTES
   une porte à distance <= tol de l'identité ne change rien : Δ la saute.
   ni contraction, ni SVD, ni écriture mémoire.

2) QUBITS AU REPOS
   un site qu'aucune porte active n'a touché depuis la dernière observation
   n'est pas recalculé. Le registre n'est pas « en marche » par défaut.

3) ÉVALUATION PARESSEUSE PAR CÔNE DE LUMIÈRE
   pour répondre à une question portant sur quelques qubits, seules comptent
   les portes de leur PASSÉ : à d couches de la fin, l'influence ne peut venir
   que de d sites de distance. Δ ne calcule que ce cône, et laisse dormir le
   reste du registre, aussi grand soit-il.

ÉNERGIE : on compte les opérations réellement exécutées (flops estimés par les
SVD et contractions) et le temps. La conversion en joules est une ESTIMATION
déclarée, à partir d'une puissance typique (téléphone ~2 W, PC ~65 W) :
    énergie ≈ puissance x temps effectif.
Aucune mesure électrique réelle n'est possible depuis ici ; c'est écrit.

Dépendances : numpy. Compatible Termux.
"""
import math, time
import numpy as np

C = np.complex128
I4 = np.eye(4, dtype=C)


def inerte(U, tol=1e-12):
    """La porte est-elle l'identité, à une phase globale près ?"""
    ph = U[0, 0]
    if abs(abs(ph) - 1) > 1e-9:
        ph = 1.0
    return float(np.abs(U / ph - I4).max()) <= tol


class Delta11:
    def __init__(self, n, eps=1e-10, X_max=256, tol_inerte=1e-12):
        self.n, self.eps, self.X_max, self.tol = n, eps, X_max, tol_inerte
        self.G = [np.zeros((1, 2, 1), C) for _ in range(n)]
        for g in self.G:
            g[0, 0, 0] = 1
        self.L = [np.ones(1) for _ in range(n + 1)]
        self.logfid = 0.0
        self.X_pic = 1
        self.executees = self.sautees = 0
        self.flops = 0
        self.temps = 0.0

    def memoire(self):
        return sum(g.size for g in self.G) * 16 + sum(l.size for l in self.L) * 8

    def energie_J(self, watts=2.0):
        """ESTIMATION : puissance typique x temps de calcul effectif."""
        return watts * self.temps

    def porte2(self, i, U):
        if inerte(U, self.tol):
            self.sautees += 1
            return
        t0 = time.time()
        th = np.einsum('l,lsm,m,mtr,r->lstr', self.L[i], self.G[i], self.L[i + 1],
                       self.G[i + 1], self.L[i + 2])
        th = np.einsum('stuv,luvr->lstr', U.reshape(2, 2, 2, 2), th)
        M = th.reshape(th.shape[0] * 2, 2 * th.shape[3])
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        self.flops += 22 * M.shape[0] * M.shape[1] * min(M.shape)      # coût SVD ~ 22 m n k
        tot = float((s ** 2).sum())
        cum = np.cumsum(s[::-1] ** 2)[::-1]
        X = len(s)
        for k in range(1, len(s) + 1):
            if (cum[k] if k < len(cum) else 0.0) / tot <= self.eps:
                X = k
                break
        X = max(1, min(X, self.X_max))
        keep = float((s[:X] ** 2).sum())
        self.logfid += math.log(max(keep / tot, 1e-300))
        inv = lambda x: np.where(x > 1e-10, 1.0 / np.maximum(x, 1e-30), 0.0)
        self.G[i] = u[:, :X].reshape(-1, 2, X) * inv(self.L[i])[:, None, None]
        self.G[i + 1] = vh[:X, :].reshape(X, 2, -1) * inv(self.L[i + 2])[None, None, :]
        self.L[i + 1] = s[:X] / math.sqrt(keep)
        self.X_pic = max(self.X_pic, X)
        self.executees += 1
        self.temps += time.time() - t0

    def executer(self, circuit, sites=None):
        """circuit = [(parité, {lien: porte}), ...]
        `sites` : si fourni, Δ ne calcule que le CÔNE DE LUMIÈRE de ces qubits."""
        besoin = None
        if sites is not None:
            besoin = [set(sites)]
            for par, portes in reversed(circuit):        # remontée du cône
                s = set(besoin[-1])
                for i in portes:
                    if i in s or i + 1 in s:
                        s |= {i, i + 1}
                besoin.append(s)
            besoin = list(reversed(besoin))[:-1]
        for c, (par, portes) in enumerate(circuit):
            utile = besoin[c] if besoin is not None else None
            for i, U in sorted(portes.items()):
                if utile is not None and i not in utile and i + 1 not in utile:
                    self.sautees += 1
                    continue
                self.porte2(i, U)
        return self

    def z(self, q):
        w = np.einsum('l,lsr,r->s', self.L[q] ** 2, np.abs(self.G[q]) ** 2, self.L[q + 1] ** 2)
        return float((w * np.array([1.0, -1.0])).sum() / w.sum())

    @property
    def fid(self):
        return math.exp(self.logfid)


# --------------------------------------------------------------- circuits
def circuit_zones(n, depth, seed, zones, theta=0.9, inertes=0.0):
    """Un registre où SEULES certaines zones travaillent.
    `zones` : liste de (début, largeur). Ailleurs : portes identité (inertes).
    `inertes` : proportion supplémentaire de portes identité dans les zones."""
    rng = np.random.default_rng(seed)
    actifs = set()
    for d, w in zones:
        actifs |= set(range(d, d + w))
    circ = []
    for c in range(depth):
        par = c % 2
        portes = {}
        for i in range(par, n - 1, 2):
            if i in actifs and (i + 1) in actifs and rng.random() >= inertes:
                a = rng.uniform(-theta, theta)
                ph = np.exp(-0.5j * a)
                U = np.diag([ph, ph.conj(), ph.conj(), ph]).astype(C)
                h = rng.uniform(-theta, theta)
                R = np.array([[math.cos(h / 2), -math.sin(h / 2)],
                              [math.sin(h / 2), math.cos(h / 2)]], C)
                portes[i] = U @ np.kron(R, R)
            else:
                portes[i] = I4.copy()                 # porte inerte : qubit au repos
        circ.append((par, portes))
    return circ
