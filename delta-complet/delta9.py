#!/usr/bin/env python3
"""Δ9 V0.9 — SVD randomisée gardée, ECC (CRC + parité XOR), reprise, distribué.

1) SVD RANDOMISÉE INTÉGRÉE, AVEC GARDE-FOU
   le tirage aléatoire cherche vite les fils qui comptent ; on VÉRIFIE ensuite
   le résidu ||M - U S V|| sur la part gardée. S'il dépasse la tolérance, Δ
   refait une SVD exacte. Le hasard cherche, il ne décide jamais seul.

2) ECC — la correction d'erreur ici n'est pas quantique (Δ n'a pas de bruit
   physique) mais de STOCKAGE et de TRANSPORT :
     - CRC32 par tenseur : détecte toute corruption d'un octet
     - parité XOR entre nœuds : un nœud perdu est reconstruit exactement
       (code d'effacement, tolérance à une panne)

3) REPRISE SUR SATURATION
   la règle de René reste l'arrêt et l'alerte. Avec politique='reprise', Δ
   REJOUE la couche fautive avec un X étendu au lieu d'abandonner, et l'écrit
   au journal. Rien n'est jamais tronqué en silence.

4) DISTRIBUÉ
   la chaîne est découpée en segments confiés à des nœuds ; chaque nœud tient
   ses tenseurs, son CRC, et le nœud de parité tient le XOR de tous les autres.

Dépendances : numpy. Compatible Termux.
"""
import math, zlib, copy
import numpy as np
import delta8 as D8


# ─────────────────────────────────────────────── 1. SVD randomisée gardée
def svd_gardee(M, k, rng, tol=1e-10, sur=10, stats=None):
    """SVD randomisée + vérification. Renvoie (U, s, Vh, mode)."""
    mn = min(M.shape)
    if mn <= 32 or k >= mn - 1:
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        if stats is not None:
            stats['exacte'] += 1
        return U, s, Vh, 'exacte'
    p = min(mn, k + sur)
    Y = M @ (rng.normal(size=(M.shape[1], p)) + 1j * rng.normal(size=(M.shape[1], p)))
    Q, _ = np.linalg.qr(Y)
    U2, s, Vh = np.linalg.svd(Q.conj().T @ M, full_matrices=False)
    U = Q @ U2
    # GARDE-FOU : l'énergie captée doit valoir celle de la matrice
    capte = float((s ** 2).sum())
    total = float((np.abs(M) ** 2).sum())
    if total - capte > tol * max(total, 1e-300):
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        if stats is not None:
            stats['rattrapee'] += 1
        return U, s, Vh, 'rattrapee'
    if stats is not None:
        stats['aleatoire'] += 1
    return U, s, Vh, 'aleatoire'


# ───────────────────────────────────────────────────────────── 2. ECC
def crc(a):
    return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).tobytes()) & 0xFFFFFFFF


def octets(a):
    return np.ascontiguousarray(a).view(np.uint8).ravel()


class ECC:
    """Parité XOR sur N blocs : un bloc perdu est reconstruit exactement."""
    def __init__(self):
        self.parite = None
        self.crcs = {}

    def ajouter(self, cle, a):
        b = octets(a)
        if self.parite is None:
            self.parite = b.copy()
        else:
            if len(b) > len(self.parite):
                self.parite = np.pad(self.parite, (0, len(b) - len(self.parite)))
            self.parite[:len(b)] ^= b
        self.crcs[cle] = (crc(a), a.shape, a.dtype.str, len(b))

    def verifier(self, cle, a):
        return crc(a) == self.crcs[cle][0]

    def reconstruire(self, cle, blocs_restants):
        """Reconstruit le bloc manquant : XOR de la parité et de tous les autres."""
        p = self.parite.copy()
        for c, a in blocs_restants.items():
            b = octets(a)
            p[:len(b)] ^= b
        c0, forme, dt, n = self.crcs[cle]
        a = p[:n].copy().view(np.dtype(dt)).reshape(forme)
        return a, crc(a) == c0


# ─────────────────────────────────── 3+4. moteur avec reprise et ECC
class Delta9(D8.Delta8):
    def __init__(self, *a, tol_alea=1e-10, **kw):
        super().__init__(*a, **kw)
        self.tol_alea = tol_alea
        self.stats = {'exacte': 0, 'aleatoire': 0, 'rattrapee': 0}
        self.reprises = []

    def couche(self, num, parite, portes):
        """Même couche que Δ8, mais SVD gardée et reprise sur saturation."""
        etat = None
        if self.politique == 'reprise':
            etat = (copy.deepcopy(self.G), copy.deepcopy(self.L), self.logfid,
                    self.mu.copy(), list(self.journal))
        try:
            self._couche(num, parite, portes)
        except D8.SaturationDelta as e:
            if self.politique != 'reprise':
                raise
            self.G, self.L, self.logfid, self.mu, self.journal = (
                etat[0], etat[1], etat[2], etat[3], etat[4])
            j = e.info['lien']
            facteur = max(2.0, e.info['X_requis'] / max(e.info['X_accorde'], 1))
            self.mu[j] *= facteur                       # on étend le budget du lien fautif
            self.reprises.append(dict(e.info, facteur=round(facteur, 2)))
            self._couche(num, parite, portes)           # on REJOUE la couche

    def _couche(self, num, parite, portes):
        j_list, ctx, svds = [], [], []
        for k, i in enumerate(range(parite, self.n - 1, 2)):
            th = np.einsum('l,lsm,m,mtr,r->lstr', self.L[i], self.G[i], self.L[i + 1],
                           self.G[i + 1], self.L[i + 2])
            th = np.einsum('stuv,luvr->lstr', portes[k].reshape(2, 2, 2, 2), th)
            M = th.reshape(th.shape[0] * 2, 2 * th.shape[3])
            if self.alea:
                # k PRÉDICTIF : la demande d'un lien change peu d'une couche à
                # l'autre. On part du X précédent, avec de la marge.
                k_est = min(2 * len(self.L[i + 1]) + 8, self.X_max_j(i + 1))
                u, s, vh, mode = svd_gardee(M, k_est, self.rng,
                                            self.tol_alea, stats=self.stats)
            else:
                u, s, vh = np.linalg.svd(M, full_matrices=False)
                self.stats['exacte'] += 1
            j_list.append(i + 1); ctx.append(i); svds.append((u, s, vh))
        if self.fixe:
            dem = [min(self.X_base, len(s)) for (u, s, vh) in svds]
        else:
            dem = [self.X_demande(s, self.eps_j(j)) for (u, s, vh), j in zip(svds, j_list)]
        evit = [float((s[:d] ** 2).sum() / (s ** 2).sum()) for (u, s, vh), d in zip(svds, dem)]
        acc = self._accorder(dem, evit, j_list)
        for i, (u, s, vh), j, d in zip(ctx, svds, j_list, dem):
            X = max(1, min(acc[j], len(s)))
            tot = float((s ** 2).sum()); keep = float((s[:X] ** 2).sum())
            perte = max(0.0, 1.0 - keep / tot)
            if perte > self.eps_j(j) + 1e-15 and not self.fixe:
                if self.politique in ('arret', 'reprise'):
                    raise D8.SaturationDelta(num, j, perte, self.eps_j(j), X, d)
                self.alertes.append((num, j, perte, X, d))
            self.logfid += math.log(max(keep / tot, 1e-300))
            inv = lambda x: np.where(x > 1e-10, 1.0 / np.maximum(x, 1e-30), 0.0)
            self.G[i] = u[:, :X].reshape(-1, 2, X) * inv(self.L[i])[:, None, None]
            self.G[i + 1] = vh[:X, :].reshape(X, 2, -1) * inv(self.L[i + 2])[None, None, :]
            self.L[i + 1] = s[:X] / math.sqrt(keep)
            self.X_pic = max(self.X_pic, X)
            self.journal.append((num, j, X, d, perte))
            if self.mu_vivant:
                charge = X / max(self.X_max_j(j), 1)
                self.mu[j] = float(np.clip(self.mu[j] * (1.15 if charge > 0.8 else 0.93), 0.25, 1024.0))

    # ------------------------------------------------------------ distribué
    def repartir(self, nb_noeuds):
        """Découpe l'état en N parts d'OCTETS égales (un nœud peut détenir une
        fraction de tenseur) + parité XOR. La parité coûte alors exactement 1/N,
        au lieu de la taille du plus gros tenseur."""
        formes = [g.shape for g in self.G]
        flux = np.concatenate([g.ravel() for g in self.G])
        b = octets(flux)
        coupes = np.linspace(0, len(b), nb_noeuds + 1).astype(int)
        noeuds, ecc = {}, ECC()
        for k in range(nb_noeuds):
            bloc = b[coupes[k]:coupes[k + 1]].copy()
            noeuds[k] = dict(octets=(int(coupes[k]), int(coupes[k + 1])), bloc=bloc)
            ecc.ajouter(k, bloc)
        ecc.formes = formes
        ecc.total = len(b)
        return noeuds, ecc

    def remonter(self, noeuds, ecc):
        """Recolle les parts et restaure les tenseurs."""
        b = np.concatenate([noeuds[k]['bloc'] for k in sorted(noeuds)])
        flux = b.view(np.complex128)
        G, pos = [], 0
        for f in ecc.formes:
            m = int(np.prod(f))
            G.append(flux[pos:pos + m].reshape(f).copy()); pos += m
        self.G = G
        return self
