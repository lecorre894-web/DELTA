#!/usr/bin/env python3
"""Δ8 V0.8 — X à la demande, facteur % (P), multiplicateur (mu), 3 arbitrages.

Canon Δ (décidé par René) :
  %  (P_i)  : ce qu'on a le DROIT DE PERDRE      -> eps_j = eps * (1 - P_j)
  mu (mu_j) : ce qu'on a le DROIT DE DÉPENSER    -> X_max_j = X_base * mu_j
  X  (X_j)  : ce que le calcul RÉCLAME            -> plus petit X tel que
                                                     perte <= eps_j, borné par X_max_j
  mu est VIVANT : un lien qui sature réclame plus à la couche suivante,
  un lien calme rend son budget (mu_j = f(P, D, V, k, t) du chapitre 14).

ARBITRAGE quand le budget mémoire global sature — les trois sont implémentés :
  'pourcent'      : les liens des qubits importants servis d'abord
  'proportionnel' : part proportionnelle à P * mu * demande réelle
  'merite'        : le X va là où il évite le plus de perte

SATURATION (choix de René) : ARRÊT ET ALERTE.
  si un lien ne peut pas tenir sa tolérance dans le budget accordé, Δ lève
  SaturationDelta en nommant le lien, la couche, la perte et le X manquant.
  Aucune erreur silencieuse, jamais.

Dépendances : numpy. Compatible Termux.
"""
import math
import numpy as np

C = np.complex128


class SaturationDelta(Exception):
    def __init__(self, couche, lien, perte, eps, X_accorde, X_requis):
        self.info = dict(couche=couche, lien=lien, perte=perte, tolerance=eps,
                         X_accorde=X_accorde, X_requis=X_requis)
        super().__init__(f"SATURATION couche {couche}, lien {lien} : perte {perte:.3e} > "
                         f"tolérance {eps:.3e} avec X={X_accorde} (il en faudrait {X_requis})")


def svd_alea(M, k, rng, sur=8):
    """SVD randomisée : on cherche les fils qui comptent avec des directions
    aléatoires. Le hasard sert à CHERCHER, jamais à DÉCIDER."""
    p = min(M.shape[1], k + sur)
    Y = M @ (rng.normal(size=(M.shape[1], p)) + 1j * rng.normal(size=(M.shape[1], p)))
    Q, _ = np.linalg.qr(Y)
    U2, s, Vh = np.linalg.svd(Q.conj().T @ M, full_matrices=False)
    return Q @ U2, s, Vh


class Delta8:
    def __init__(self, n, P=None, X_base=8, mu=None, eps=1e-6, budget_octets=None,
                 arbitrage='proportionnel', alea=False, mu_vivant=True, graine=0,
                 fixe=False, politique='arret'):
        self.n = n
        self.P = np.zeros(n) if P is None else np.asarray(P, float)
        self.X_base, self.eps = X_base, eps
        self.mu = np.ones(n + 1) if mu is None else np.asarray(mu, float)
        self.budget = budget_octets
        self.arbitrage, self.alea, self.mu_vivant = arbitrage, alea, mu_vivant
        self.fixe = fixe              # X fixe : X = X_base partout (mode de comparaison)
        self.politique = politique    # 'arret' : arrêt et alerte (règle de René)
                                      # 'audit' : continue et compte l'erreur (comparaison seulement)
        self.alertes = []
        self.rng = np.random.default_rng(graine)
        self.G = [np.zeros((1, 2, 1), C) for _ in range(n)]
        for g in self.G:
            g[0, 0, 0] = 1
        self.L = [np.ones(1) for _ in range(n + 1)]
        self.logfid = 0.0
        self.X_pic = 1
        self.journal = []            # (couche, lien, X accordé, X demandé, perte)

    # ---------------------------------------------------------------- outils
    def eps_j(self, j):
        p = max(self.P[j - 1], self.P[j]) if 0 < j < self.n else 0.0
        return self.eps * (1.0 - p)

    def X_max_j(self, j):
        return max(1, int(round(self.X_base * self.mu[j])))

    def memoire(self):
        return sum(g.size for g in self.G) * 16 + sum(l.size for l in self.L) * 8

    def X_demande(self, s, eps):
        """Plus petit X tel que le poids jeté <= eps."""
        w = (s ** 2)[::-1].cumsum()[::-1]          # poids jeté si on garde k
        tot = float((s ** 2).sum())
        for k in range(1, len(s) + 1):
            if (w[k] if k < len(w) else 0.0) / tot <= eps:
                return k
        return len(s)

    # ------------------------------------------------------------- arbitrage
    def _accorder(self, demandes, pertes_evitees, j_list):
        """Répartit le budget mémoire entre les liens. Renvoie {lien: X accordé}."""
        voulu = {j: d for j, d in zip(j_list, demandes)}
        cout = lambda a: sum(self.G[i].shape[0] * 2 * max(a.get(i + 1, self.L[i + 1].size), 1) * 16
                             for i in range(self.n))
        # mu est un BUDGET, pas un plafond arbitraire : tant que le budget global
        # le permet, mu s'étend jusqu'à la demande réelle du lien.
        if self.budget is None or cout(voulu) <= self.budget:
            for j, d in voulu.items():
                self.mu[j] = max(self.mu[j], d / self.X_base)
            return voulu
        acc = {j: min(d, self.X_max_j(j)) for j, d in zip(j_list, demandes)}
        if cout(acc) <= self.budget:
            return acc
        if self.arbitrage == 'pourcent':
            ordre = sorted(j_list, key=lambda j: -max(self.P[j - 1], self.P[j]))
        elif self.arbitrage == 'merite':
            ordre = sorted(j_list, key=lambda j: -pertes_evitees[j_list.index(j)])
        else:                                     # proportionnel
            poids = {j: max(self.P[j - 1], self.P[j], 1e-3) * self.mu[j] * acc[j] for j in j_list}
            tot = sum(poids.values())
            red = {j: max(1, int(acc[j] * (self.budget / max(cout(acc), 1)) ** 0.5 *
                                 (poids[j] * len(j_list) / max(tot, 1e-12)) ** 0.25)) for j in j_list}
            return {j: min(red[j], acc[j]) for j in j_list}
        petit = {j: 1 for j in j_list}
        for j in ordre:                           # on sert par ordre de priorité
            essai = dict(petit); essai[j] = acc[j]
            if cout(essai) <= self.budget:
                petit = essai
        return petit

    # ----------------------------------------------------------------- portes
    def couche(self, num, parite, portes):
        j_list, thetas, svds = [], [], []
        for k, i in enumerate(range(parite, self.n - 1, 2)):
            U = portes[k]
            Ga, Gb = self.G[i], self.G[i + 1]
            th = np.einsum('l,lsm,m,mtr,r->lstr', self.L[i], Ga, self.L[i + 1], Gb, self.L[i + 2])
            th = np.einsum('stuv,luvr->lstr', U.reshape(2, 2, 2, 2), th)
            M = th.reshape(th.shape[0] * 2, 2 * th.shape[3])
            if self.alea and min(M.shape) > 32:
                u, s, vh = svd_alea(M, min(self.X_max_j(i + 1), min(M.shape) - 1), self.rng)
            else:
                u, s, vh = np.linalg.svd(M, full_matrices=False)
            j_list.append(i + 1); thetas.append((i, th, M)); svds.append((u, s, vh))
        if self.fixe:
            dem = [min(self.X_base, len(s)) for (u, s, vh) in svds]
        else:
            dem = [self.X_demande(s, self.eps_j(j)) for (u, s, vh), j in zip(svds, j_list)]
        evit = [float((s[:d] ** 2).sum() / (s ** 2).sum()) for (u, s, vh), d in zip(svds, dem)]
        acc = self._accorder(dem, evit, j_list)
        for (i, th, M), (u, s, vh), j, d in zip(thetas, svds, j_list, dem):
            X = max(1, min(acc[j], len(s)))
            tot = float((s ** 2).sum()); keep = float((s[:X] ** 2).sum())
            perte = max(0.0, 1.0 - keep / tot)
            if perte > self.eps_j(j) + 1e-15 and not self.fixe:
                if self.politique == 'arret':
                    raise SaturationDelta(num, j, perte, self.eps_j(j), X, d)
                self.alertes.append((num, j, perte, X, d))
            self.logfid += math.log(max(keep / tot, 1e-300))
            sk = s[:X] / math.sqrt(keep)
            la, lb = self.L[i], self.L[i + 2]
            inv = lambda x: np.where(x > 1e-10, 1.0 / np.maximum(x, 1e-30), 0.0)
            self.G[i] = (u[:, :X].reshape(-1, 2, X)) * inv(la)[:, None, None]
            self.G[i + 1] = (vh[:X, :].reshape(X, 2, -1)) * inv(lb)[None, None, :]
            self.L[i + 1] = sk
            self.X_pic = max(self.X_pic, X)
            self.journal.append((num, j, X, d, perte))
            if self.mu_vivant:                   # mu_j = f(P, D, V, k, t)
                charge = X / max(self.X_max_j(j), 1)
                self.mu[j] = float(np.clip(self.mu[j] * (1.15 if charge > 0.8 else 0.93), 0.25, 64.0))

    def couche1(self, portes):
        self.G = [np.einsum('st,ltr->lsr', U, g) for U, g in zip(portes, self.G)]

    @property
    def fid(self):
        return math.exp(self.logfid)

    def dense(self):
        psi = np.ones((1, 1), C)
        for s in range(self.n):
            A = self.G[s] * self.L[s][:, None, None]
            psi = np.einsum('pl,lsr->psr', psi, A).reshape(-1, A.shape[2])
        return psi[:, 0]


def executer(sim, circuit):
    """circuit = liste de (parité, [portes 4x4 par lien], [portes 1 qubit ou None])"""
    for num, (par, g2, g1) in enumerate(circuit):
        if g1 is not None:
            sim.couche1(g1)
        sim.couche(num, par, g2)
    return sim
