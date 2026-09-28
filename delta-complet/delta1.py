#!/usr/bin/env python3
"""Δ1 V0.1 — émulateur Δ-QI-V à intrication flottante et proportionnée.

Architecture (canon René / Δ) :
  - individualisme : chaque qubit vit seul tant qu'il n'est pas intriqué
  - îlots tensoriels : une porte à 2 qubits sur deux îlots distincts les FUSIONNE
  - intrication flottante : après chaque porte, Δ tente de REDÉCOUPER l'îlot
      * test de Schmidt sur les coupures que la porte a pu modifier
      * niveau 1 (exact)       : coupe si la valeur singulière résiduelle = 0
      * niveau 2 (flottant)    : coupe si le poids jeté <= ε_i
  - proportionné (facteur %) : ε_i = ε · (1 - P_i)
      P_i = 100 % -> qubit exact, jamais tronqué
      P_i =   4 % -> qubit peu important, tronqué au besoin
  - mesure : un qubit mesuré sort de son îlot (arme de libération)

Résultat principe : une porte sur (a,b) ne change la séparabilité QUE des
coupures qui séparent a de b. Le reste de l'état est intouché. Δ ne teste donc
que ces coupures : c'est exact, pas une heuristique.

Dépendances : numpy. Compatible Termux.
"""
import math, time
import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=np.complex128).reshape(2, 2, 2, 2)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128).reshape(2, 2, 2, 2)


def rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=np.complex128)


class Delta1:
    def __init__(self, n, eps=0.0, P=None, split=True, klimit=26, regle='strict'):
        self.n = n
        self.regle = regle   # 'strict' : le qubit le plus important de l'îlot décide
                             # 'detache' : seuls les qubits du bloc détaché décident
        self.klimit = klimit
        self.eps = float(eps)
        self.P = np.ones(n) if P is None else np.asarray(P, float)
        self.split = split
        # îlot = [liste de qubits, tenseur de forme (2,)*k]
        self.isl = {q: [[q], np.array([1, 0], dtype=np.complex128)] for q in range(n)}
        self.owner = list(range(n))           # qubit -> id d'îlot
        self.groups = {}                      # mémoire des fusions par îlot
        self.fid = 1.0                        # fidélité estimée (produit des poids gardés)
        self.kmax = 1
        self.mem_peak = 16 * 2 * n
        self.merges = self.splits = self.trunc = 0

    # ------------------------------------------------------------ mesures
    def memory(self):
        return sum(16 * t.size for _, t in self.isl.values())

    def _track(self):
        k = max(len(qs) for qs, _ in self.isl.values())
        self.kmax = max(self.kmax, k)
        if k > self.klimit:
            raise MemoryError(f'MUR 2^k atteint : k = {k}')
        self.mem_peak = max(self.mem_peak, self.memory())

    def eps_q(self, q):
        return self.eps * (1.0 - self.P[q])

    # ------------------------------------------------------------ portes
    def g1(self, U, q):
        i = self.owner[q]
        qs, t = self.isl[i]
        ax = qs.index(q)
        t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
        self.isl[i] = [qs, t]

    def _newid(self):
        nid = max(max(self.isl) + 1 if self.isl else 0, self.n)
        while nid in self.isl:
            nid += 1
        return nid

    def g2(self, U, a, b):
        ia, ib = self.owner[a], self.owner[b]
        if ia != ib:                                     # FUSION
            qa, ta = self.isl.pop(ia)
            qb, tb = self.isl.pop(ib)
            ga, gb = self.groups.pop(ia, []), self.groups.pop(ib, [])
            qs = qa + qb
            nid = min(ia, ib)
            self.isl[nid] = [qs, np.multiply.outer(ta, tb)]
            # MÉMOIRE DES FUSIONS : chaque ancien îlot reste une coupure candidate
            self.groups[nid] = ga + gb + [frozenset(qa), frozenset(qb)]
            for q in qs:
                self.owner[q] = nid
            self.merges += 1
            ia = nid
        qs, t = self.isl[ia]
        xa, xb = qs.index(a), qs.index(b)
        t = np.tensordot(U, t, axes=([2, 3], [xa, xb]))
        self.isl[ia] = [qs, np.moveaxis(t, [0, 1], [xa, xb])]
        self._track()
        if self.split:
            self._refloat(a, b)
        self._track()

    # ------------------------------------------------------ redécoupage
    def _refloat(self, a, b, budget=10):
        """Intrication flottante : teste les SEULES coupures que la porte (a,b)
        a pu modifier, c'est-à-dire celles qui séparent a de b. Les singletons
        {a}, {b} et les anciens îlots fusionnés sont les candidats."""
        tries = 0
        changed = True
        while changed and tries < budget:
            changed = False
            i = self.owner[a]
            if len(self.isl[i][0]) == 1 and self.owner[b] != i:
                break
            cands = [frozenset([a]), frozenset([b])]
            for G in self.groups.get(i, []):
                if (a in G) != (b in G):
                    cands.append(G)
            seen = set()
            for G in sorted(cands, key=len):
                if G in seen:
                    continue
                seen.add(G)
                j = self.owner[next(iter(G))]
                if not all(self.owner[q] == j for q in G):
                    continue
                if len(self.isl[j][0]) <= len(G):
                    continue
                tries += 1
                if self._cut(j, G):
                    changed = True
                    break
                if tries >= budget:
                    break

    def _cut(self, i, G):
        qs, t = self.isl[i]
        A = [q for q in qs if q in G]
        B = [q for q in qs if q not in G]
        perm = [qs.index(q) for q in A + B]
        M = np.transpose(t, perm).reshape(2 ** len(A), 2 ** len(B))
        rho = M @ M.conj().T if M.shape[0] <= M.shape[1] else M.conj().T @ M
        w = np.linalg.eigvalsh(rho)
        tot = float(w.sum().real)
        lost = max(tot - float(w[-1].real), 0.0) / max(tot, 1e-300)
        if self.regle == 'strict':
            epsc = min(self.eps_q(q) for q in qs)   # le qubit le plus important décide
        else:
            epsc = min(self.eps_q(q) for q in A)    # le bloc détaché décide
        if lost > max(epsc, 1e-12):
            return False
        U_, s, Vh = np.linalg.svd(M, full_matrices=False)
        ta, tb = U_[:, 0], Vh[0, :]
        grp = self.groups.pop(i, [])
        self.isl[i] = [A, ta.reshape((2,) * len(A))]
        self.groups[i] = [g for g in grp if g < G]
        nid = self._newid()
        self.isl[nid] = [B, tb.reshape((2,) * len(B))]
        rest = frozenset(B)
        self.groups[nid] = [g - G for g in grp if (g - G) and (g - G) < rest]
        for q in B:
            self.owner[q] = nid
        if lost > 1e-12:
            self.fid *= (1.0 - lost)
            self.trunc += 1
        self.splits += 1
        return True

    # ------------------------------------------------------------ mesure
    def measure(self, q, rng):
        i = self.owner[q]
        qs, t = self.isl[i]
        ax = qs.index(q)
        M = np.moveaxis(t, ax, 0).reshape(2, -1)
        p0 = float(np.vdot(M[0], M[0]).real)
        p1 = float(np.vdot(M[1], M[1]).real)
        o = 0 if rng.random() < p0 / (p0 + p1) else 1
        rest = M[o] / np.linalg.norm(M[o])
        others = [x for x in qs if x != q]
        u = np.array([1, 0] if o == 0 else [0, 1], dtype=np.complex128)
        if others:
            self.isl[i] = [others, rest.reshape((2,) * len(others))]
            self.groups[i] = [g - {q} for g in self.groups.get(i, []) if g - {q}]
            nid = self._newid()
            self.isl[nid] = [[q], u]
            self.owner[q] = nid
            # la projection peut libérer tout le reste de l'îlot (ex. GHZ) :
            # on tente d'éplucher chaque qubit restant, exactement (ε = 0)
            if self.split:
                for x in list(others):
                    j = self.owner[x]
                    if len(self.isl[j][0]) > 1:
                        self._cut(j, frozenset([x]))
        else:
            self.isl[i] = [[q], u]
        self.splits += 1
        self._track()
        return o

    # ------------------------------------------------- état dense (contrôle)
    def dense(self):
        """Reconstruit le vecteur d'état complet. Réservé aux petits n (contrôle)."""
        t = np.array(1.0, dtype=np.complex128)
        order = []
        for qs, ti in self.isl.values():
            t = np.multiply.outer(t, ti)
            order += qs
        perm = [order.index(q) for q in range(self.n)]
        return np.transpose(t, perm).reshape(-1)


# ================================================================ référence
class Dense:
    """Simulateur exact vecteur d'état 2^n : la vérité terrain."""
    def __init__(self, n):
        self.n = n
        self.t = np.zeros((2,) * n, dtype=np.complex128)
        self.t[(0,) * n] = 1

    def g1(self, U, q):
        self.t = np.moveaxis(np.tensordot(U, self.t, axes=([1], [q])), 0, q)

    def g2(self, U, a, b):
        t = np.tensordot(U, self.t, axes=([2, 3], [a, b]))
        self.t = np.moveaxis(t, [0, 1], [a, b])

    def vec(self):
        return self.t.reshape(-1)


# ================================================================ circuits
def c_produit(n, rng):
    ops = []
    for q in range(n):
        ops.append(('g1', ry(rng.uniform(0, math.pi)), q))
        ops.append(('g1', rz(rng.uniform(0, 2 * math.pi)), q))
    return ops


def c_ghz(n, rng):
    ops = [('g1', H, 0)]
    for q in range(n - 1):
        ops.append(('g2', CNOT, q, q + 1))
    return ops


def c_aller_retour(n, rng):
    """Intrique par paires puis défait : l'intrication flottante doit se dissoudre."""
    ops = []
    for q in range(0, n - 1, 2):
        ops.append(('g1', H, q))
        ops.append(('g2', CNOT, q, q + 1))
    for q in range(1, n - 1, 2):
        ops.append(('g2', CZ, q, q + 1))
    for q in range(1, n - 1, 2):
        ops.append(('g2', CZ, q, q + 1))
    for q in range(0, n - 1, 2):
        ops.append(('g2', CNOT, q, q + 1))
        ops.append(('g1', H, q))
    return ops


def c_vague(n, rng):
    """Intrication flottante pure : une vague de paires intriquées qui traverse
    le registre. Chaque paire s'intrique réellement (état de Bell), puis se libère."""
    ops = []
    for q in range(n - 1):
        ops.append(('g1', H, q))
        ops.append(('g2', CNOT, q, q + 1))      # paire de Bell : vraie intrication
        ops.append(('g1', rz(0.7), q + 1))
        ops.append(('g1', rz(-0.7), q + 1))
        ops.append(('g2', CNOT, q, q + 1))      # libération
        ops.append(('g1', H, q))
        ops.append(('g1', ry(0.3 + 0.01 * q), q))
    return ops


def c_brique(n, depth, rng):
    """Circuit aléatoire 1D en briques, voisins proches : le cas « réaliste »."""
    ops = []
    for d in range(depth):
        for q in range(n):
            ops.append(('g1', ry(rng.uniform(0, math.pi)), q))
            ops.append(('g1', rz(rng.uniform(0, 2 * math.pi)), q))
        for q in range(d % 2, n - 1, 2):
            ops.append(('g2', CZ, q, q + 1))
    return ops


def c_faible(n, depth, rng, theta=0.15):
    """Intrication faible : rotations proches de l'identité entre voisins."""
    ops = []
    for d in range(depth):
        for q in range(n):
            ops.append(('g1', rx(rng.uniform(-theta, theta)), q))
        for q in range(d % 2, n - 1, 2):
            ops.append(('g2', CNOT, q, q + 1))
            ops.append(('g1', rz(rng.uniform(-theta, theta)), q + 1))
            ops.append(('g2', CNOT, q, q + 1))
    return ops


def run(sim, ops):
    for op in ops:
        if op[0] == 'g1':
            sim.g1(op[1], op[2])
        else:
            sim.g2(op[1], op[2], op[3])
    return sim
