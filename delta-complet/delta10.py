#!/usr/bin/env python3
"""Δ10 V1.0 — SYMÉTRIE U(1) : tenseurs par blocs de charge.

Idée : si le circuit conserve une quantité (ici le nombre de qubits à 1), alors
un tenseur ne peut relier que des états de charges compatibles :

        charge à gauche  +  charge du site  =  charge à droite

Toutes les autres cases sont nulles PAR CONSTRUCTION, pas par approximation.
Δ10 ne les stocke donc plus du tout : chaque lien porte des SECTEURS étiquetés
par leur charge, et chaque tenseur est un dictionnaire de blocs.

Conséquences :
  - mémoire et temps divisés (secteurs indépendants, SVD par bloc, plus petite)
  - fidélité STRICTEMENT identique : aucune information n'est jetée
  - l'état reste physiquement valide : une troncature ne peut plus mélanger
    des secteurs interdits
  - une porte qui violerait la symétrie est REFUSÉE, jamais subie en silence

Dépendances : numpy. Compatible Termux.
"""
import math
import numpy as np

C = np.complex128


class SymetrieViolee(Exception):
    pass


def porte_hop(theta, phi=0.0):
    """Porte conservant le nombre d'excitations (type saut / XY + phase)."""
    c, s = math.cos(theta), math.sin(theta)
    U = np.zeros((4, 4), C)
    U[0, 0] = 1
    U[1, 1] = c; U[1, 2] = -1j * s
    U[2, 1] = -1j * s; U[2, 2] = c
    U[3, 3] = np.exp(1j * phi)
    return U


def charges_portees(U, tol=1e-12):
    """Vérifie que U conserve la charge : U[(s,t),(u,v)] != 0 => s+t == u+v."""
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for a, (s, t) in enumerate(idx):
        for b, (u, v) in enumerate(idx):
            if abs(U[a, b]) > tol and (s + t) != (u + v):
                raise SymetrieViolee(
                    f"la porte relie la charge {u+v} à la charge {s+t} : "
                    f"symétrie U(1) violée (élément {abs(U[a,b]):.3e})")
    return True


class Delta10:
    """MPS par blocs de charge, forme de Vidal (Gam, lam) sectorisée."""

    def __init__(self, n, occupation=None, eps=1e-10, X_max=512):
        self.n, self.eps, self.X_max = n, eps, X_max
        occ = np.zeros(n, int) if occupation is None else np.asarray(occupation, int)
        self.lam = [{0: np.ones(1)}]
        self.Gam = []
        q = 0
        for i in range(n):
            s = int(occ[i])
            self.Gam.append({(q, s): np.ones((1, 1), C)})
            q += s
            self.lam.append({q: np.ones(1)})
        self.logfid = 0.0
        self.X_pic = 1
        self.blocs_evites = 0
        self.flops = 0        # flops réellement exécutés
        self.flops_sans_blocs = 0   # ce qu'aurait coûté le même pas sans symétrie
        self.temps = 0.0

    # ------------------------------------------------------------- mesures
    def memoire(self):
        m = sum(b.size for G in self.Gam for b in G.values()) * 16
        m += sum(v.size for L in self.lam for v in L.values()) * 8
        return m

    def memoire_dense(self):
        """Ce que coûterait le MÊME état sans blocs (tenseurs pleins)."""
        d = [sum(v.size for v in L.values()) for L in self.lam]
        return sum(d[i] * 2 * d[i + 1] for i in range(self.n)) * 16

    def dim_lien(self, j):
        return sum(v.size for v in self.lam[j].values())

    # -------------------------------------------------------------- portes
    def porte2(self, i, U):
        import time as _t
        _t0 = _t.time()
        charges_portees(U)                      # refus si la symétrie est violée
        gauche, milieu, droite = self.lam[i], self.lam[i + 1], self.lam[i + 2]
        A, B = self.Gam[i], self.Gam[i + 1]
        # 1. thêta par (ql, s, t)
        th = {}
        for (ql, s), a in A.items():
            qm = ql + s
            for t in (0, 1):
                b = B.get((qm, t))
                if b is None or (qm + t) not in droite:
                    continue
                m = (a * milieu[qm][None, :]) @ b
                m = gauche[ql][:, None] * m * droite[qm + t][None, :]
                th[(ql, s, t)] = m
        if not th:
            return
        # 2. application de la porte (blocs de charge totale c = s+t)
        idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
        th2 = {}
        for (ql, s, t), m in th.items():
            for a_, (u, v) in enumerate(idx):
                if u + v != s + t:
                    continue
                coef = U[a_, idx.index((s, t))]
                if abs(coef) < 1e-15:
                    self.blocs_evites += 1
                    continue
                cle = (ql, u, v)
                th2[cle] = th2.get(cle, 0) + coef * m
        # 3. regroupement par charge intermédiaire a = ql + s (= secteur du lien)
        secteurs = {}
        for (ql, s, t), m in th2.items():
            secteurs.setdefault(ql + s, {})[(ql, s, t)] = m
        nouveaux_lam, nouveaux_A, nouveaux_B, svds = {}, {}, {}, {}
        for a, blocs in secteurs.items():
            lignes = sorted({(ql, s) for (ql, s, t) in blocs})
            colonnes = sorted({t for (ql, s, t) in blocs})
            dl = [gauche[ql].size for (ql, s) in lignes]
            dc = [droite[a + t].size for t in colonnes]
            M = np.zeros((sum(dl), sum(dc)), C)
            for r, (ql, s) in enumerate(lignes):
                for c_, t in enumerate(colonnes):
                    m = blocs.get((ql, s, t))
                    if m is not None:
                        M[sum(dl[:r]):sum(dl[:r + 1]), sum(dc[:c_]):sum(dc[:c_ + 1])] = m
            u, sv, vh = np.linalg.svd(M, full_matrices=False)
            self.flops += 22 * M.shape[0] * M.shape[1] * min(M.shape)
            svds[a] = (u, sv, vh, lignes, colonnes, dl, dc)
        # 4. troncature GLOBALE sur tous les secteurs réunis
        toutes = np.concatenate([svds[a][1] for a in svds])
        ordre = np.sort(toutes)[::-1]
        tot = float((ordre ** 2).sum())
        garde = len(ordre)
        cum = np.cumsum(ordre[::-1] ** 2)[::-1]
        for k in range(1, len(ordre) + 1):
            if (cum[k] if k < len(cum) else 0.0) / tot <= self.eps:
                garde = k
                break
        garde = min(garde, self.X_max)
        seuil = ordre[garde - 1]
        keep = 0.0
        for a, (u, sv, vh, lignes, colonnes, dl, dc) in svds.items():
            k = int((sv >= seuil).sum())
            if k == 0:
                continue
            keep += float((sv[:k] ** 2).sum())
            nouveaux_lam[a] = sv[:k]
            for r, (ql, s) in enumerate(lignes):
                nouveaux_A[(ql, s)] = u[sum(dl[:r]):sum(dl[:r + 1]), :k] / gauche[ql][:, None]
            for c_, t in enumerate(colonnes):
                nouveaux_B[(a, t)] = vh[:k, sum(dc[:c_]):sum(dc[:c_ + 1])] / droite[a + t][None, :]
        norme = math.sqrt(keep)
        self.logfid += math.log(max(keep / tot, 1e-300))
        self.lam[i + 1] = {a: v / norme for a, v in nouveaux_lam.items()}
        self.Gam[i] = nouveaux_A
        self.Gam[i + 1] = nouveaux_B
        self.X_pic = max(self.X_pic, self.dim_lien(i + 1))
        # ce que la MÊME étape aurait coûté sans blocs : une seule grosse SVD
        m_, n_ = 2 * sum(v.size for v in gauche.values()), 2 * sum(v.size for v in droite.values())
        self.flops_sans_blocs += 22 * m_ * n_ * min(m_, n_)
        self.temps += _t.time() - _t0

    def couche(self, parite, portes):
        for k, i in enumerate(range(parite, self.n - 1, 2)):
            self.porte2(i, portes[k])

    @property
    def fid(self):
        return math.exp(self.logfid)

    # ------------------------------------------------------ contrôle exact
    def dense(self):
        """Reconstruit le vecteur d'état complet (petits n uniquement)."""
        etats = {(0,): {0: np.ones(1, C)}}   # (bits) -> {charge: vecteur de lien}
        cour = {0: np.ones((1, 1), C)}       # charge -> matrice (configurations, lien)
        confs = {0: [()]}
        for i in range(self.n):
            suiv, cs = {}, {}
            for q, M in cour.items():
                for s in (0, 1):
                    A = self.Gam[i].get((q, s))
                    if A is None:
                        continue
                    lg = self.lam[i][q]
                    bloc = (M * 1.0) @ (lg[:, None] * A)
                    q2 = q + s
                    suiv[q2] = np.vstack([suiv[q2], bloc]) if q2 in suiv else bloc
                    cs.setdefault(q2, []).extend([c + (s,) for c in confs[q]])
            cour, confs = suiv, cs
        v = np.zeros(2 ** self.n, C)
        for q, M in cour.items():
            lg = self.lam[self.n].get(q)
            if lg is None:
                continue
            vals = (M * lg[None, :]).sum(1) if M.shape[1] == lg.size else M[:, 0]
            for c, x in zip(confs[q], vals):
                v[int(''.join(map(str, c)), 2)] = x
        return v


# ------------------------------------------------- référence dense classique
def dense_reference(n, circuit, occupation=None):
    t = np.zeros((2,) * n, C)
    occ = np.zeros(n, int) if occupation is None else np.asarray(occupation, int)
    t[tuple(occ)] = 1
    for parite, portes in circuit:
        for k, i in enumerate(range(parite, n - 1, 2)):
            U = portes[k].reshape(2, 2, 2, 2)
            t = np.tensordot(U, t, axes=([2, 3], [i, i + 1]))
            t = np.moveaxis(t, [0, 1], [i, i + 1])
    return t.reshape(-1)


def circuit_symetrique(n, depth, seed):
    rng = np.random.default_rng(seed)
    return [(d % 2, [porte_hop(rng.uniform(0, math.pi / 2), rng.uniform(0, 2 * math.pi))
                     for _ in range(d % 2, n - 1, 2)]) for d in range(depth)]
