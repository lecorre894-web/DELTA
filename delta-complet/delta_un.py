#!/usr/bin/env python3
"""Δ-UN V2.0 — LA FUSION : un seul objet, toutes les briques mesurées.

           ┌──────────────────────── Δ-UN ────────────────────────┐
           │  CALCUL   fond infini partagé + grappes allumées     │  Δ5, Δ14, Δ15
           │           X à la demande, lots, symétrie, cône       │  Δ8, Δ10, Δ11
           │  CACHE    hiérarchie X3D, répertoire global,         │  Δ17, Δ19
           │           prédiction, éviction = dégradation         │
           │  MÉMOIRE  pages dédupliquées, condensateurs,         │  Δ13, Δ16, Δ18
           │           tampon en anneau                           │
           │  SCEAU    clé, arbre de Merkle, chiffrement,         │  Δ9, Δ18
           │           parité sur le chiffré                      │
           └──────────────────────────────────────────────────────┘

TROIS RÉGLAGES, ET RIEN D'AUTRE (le canon de René) :
    eps   ce qu'on a le droit de PERDRE
    P (%) ce qui COMPTE : précision, priorité de cache, protection
    mu    ce qu'on a le droit de DÉPENSER

RÈGLE D'OR, vérifiée dans chaque brique : le gain vient de la STRUCTURE.
Sur des données sans structure, Δ-UN ne promet rien et ne gagne rien.

Dépendances : numpy + bibliothèque standard. Termux OK.
"""
import hashlib, time
import numpy as np
import delta15 as D15      # calcul : fond infini, grappes, lots
import delta16 as D16      # mémoire : pages, anneau
import delta18 as D18      # condensateurs, sceau, Merkle, coffre
import delta19 as D19      # cache X3D à N étages


class DeltaUn:
    def __init__(self, noyau, b=128, eps=1e-8, portee=40, P=None, mu=1.0,
                 etages=256, capacite_L1=512 * 1024, cle=b'Delta-Rene'):
        self.eps, self.P, self.mu = eps, P or (lambda a: 0.0), mu
        self.cle = hashlib.blake2b(cle, digest_size=32).digest()
        t0 = time.time()
        self.calcul = D15.Omni(noyau, b=b, eps=eps, portee_max=portee)
        self.t_init = time.time() - t0
        self.b = b
        self.memoire = D16.MemoireDelta(eps=eps)
        self.cache = D19.X3D(self._calculer_bloc, niveaux=etages,
                             capacite_L1=capacite_L1, P=self.P)
        self.fournisseur = D15.fournisseur(b)
        self.demandes = 0

    # ---------------------------------------------------------- le calcul
    def _calculer_bloc(self, k):
        Y, _ = self.calcul.lot(k, 1, self.fournisseur)
        return Y[:, 0]

    def allumer(self, position, correction):
        """Une grappe individuelle : l'individualité ne coûte que là où elle est."""
        return self.calcul.allumer(position, correction)

    def bloc(self, k):
        """Répond pour le bloc k : cache d'abord, calcul ensuite."""
        self.demandes += 1
        return self.cache.lire(k)

    def plage(self, k0, nk, cacher=False):
        """Un lot : c'est la voie rapide (deux grands produits par classe).

        `cacher=True` : le lot NOURRIT le cache. C'est le vrai point de fusion —
        le calcul en gros, une seule fois, remplit la hiérarchie ; les questions
        détaillées qui suivent ne recalculent plus rien.
        """
        Y, f = self.calcul.lot(k0, nk, self.fournisseur)
        if cacher:
            for m in range(nk):
                self.cache._ranger(k0 + m, Y[:, m].copy(), 0)
        return Y, f

    # ------------------------------------------- la référence, sans indulgence
    def exact(self, k0, nk):
        """Référence dense : le même opérateur, sans aucune compression.
        Δ doit toujours pouvoir être confronté à ce qu'il prétend approcher."""
        b, p = self.b, self.calcul.portee
        i = np.arange(b)
        cols = np.stack([self.fournisseur(k) for k in range(k0 - p, k0 + nk + p)], 1)
        Y = np.zeros((b, nk))
        for c in range(-p, p + 1):
            M = self.calcul.noyau(i[:, None] - (i[None, :] + c * b))
            Y += M @ cols[:, p + c: p + c + nk]
        for k, G in self.calcul.grappes.items():
            if k0 <= k < k0 + nk:
                Y[:, k - k0] += G @ cols[:, p + (k - k0)]
        return Y

    def erreur(self, k0, nk):
        """L'erreur relative réellement commise, mesurée et déclarée."""
        A = self.plage(k0, nk)[0]
        B = self.exact(k0, nk)
        return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))

    # -------------------------------------------------------- la mémoire
    def ranger(self, idx, M, P=None):
        return self.memoire.ecrire(idx, M, P=self.P(idx) if P is None else P)

    def latences_us(self):
        c = self.cache
        return (c.t_succes / max(c.succes, 1) * 1e6, c.t_echecs / max(c.echecs, 1) * 1e6)

    # ----------------------------------------------------------- le sceau
    def sceller(self, blocs):
        """Scelle, chiffre et prépare la parité : l'état porte sa propre preuve."""
        self.coffre = D18.CoffreDelta(self.cle, blocs)
        self.parite = self.coffre.parite()
        return self.coffre.merkle.racine

    def verifier(self):
        return self.coffre.verifier()

    def reconstruire(self, i):
        return self.coffre.reconstruire(i, self.parite)

    # ---------------------------------------------------------- le bilan
    def bilan(self):
        c = self.calcul
        return {
            'calcul': {
                'classes': len(c.U), 'classes_vides': c.ignorees,
                'grappes_allumees': len(c.grappes),
                'octets': c.octets(), 'gflops': round(c.gflops(), 2),
                'inconnues_par_seconde': int(c.inconnues_par_seconde()),
            },
            'cache': {
                'etages': len(self.cache.etages),
                'capacite_totale': self.cache.capacite_totale(),
                'occupe': self.cache.octets(),
                'taux_succes': round(self.cache.taux(), 4),
                'latence_us': round(self.cache.latence_us(), 2),
                'degradations': self.cache.degradations,
            },
            'memoire': {
                'pages_logiques': len(self.memoire.table),
                'pages_stockees': len(self.memoire.pages),
                'deduplication': round(self.memoire.dedup(), 2) if self.memoire.table else None,
                'octets': self.memoire.octets(),
                'octets_bruts': self.memoire.octets_bruts,
            },
            'reglages': {'eps': self.eps, 'mu': self.mu},
        }


def demonstration(n_logique=10 ** 18, blocs=1024, verbose=True):
    """Démonstration de bout en bout sur un problème de taille 10^18."""
    def dire(*a):
        if verbose:
            print(*a, flush=True)
    D = DeltaUn(D15.noyau_lisse, b=128, eps=1e-8, portee=40, etages=256,
                P=lambda k: 1.0 if k % 500 == 0 else 0.0)
    k0 = n_logique // 128 // 2
    rng = np.random.default_rng(1)
    for j in range(8):
        D.allumer(k0 + j * 97, rng.standard_normal((128, 128)) * 0.2)
    dire(f'opérateur : {len(D.calcul.U)} classes, {len(D.calcul.grappes)} grappes, '
         f'{D.calcul.octets() / 1024:.1f} Kio pour 10^18 inconnues (init {D.t_init * 1000:.0f} ms)')
    dire(f'erreur relative contre l\'opérateur dense : {D.erreur(k0, 16):.2e}')
    t0 = time.time()
    Y, f = D.plage(k0, blocs, cacher=True)
    t_lot = time.time() - t0
    dire(f'lot de {blocs} blocs ({blocs * 128:,} inconnues) : {t_lot:.2f} s, '
         f'{D.calcul.gflops():.2f} Gflops, {D.calcul.inconnues_par_seconde():,.0f} inconnues/s')
    t0 = time.time()
    for j in range(300):
        D.bloc(k0 + (j % 60))
    t_cache = time.time() - t0
    dire(f'300 questions avec cache : {t_cache * 1000:.0f} ms, '
         f'{100 * D.cache.taux():.1f} % de succès, {D.cache.latence_us():.1f} µs/accès')
    parts = [Y[:, i] for i in range(0, min(blocs, 64))]
    t0 = time.time()
    racine = D.sceller(parts)
    t_sceau = time.time() - t0
    dire(f'état scellé : racine {racine.hex()[:24]}... en {t_sceau * 1000:.0f} ms, '
         f'sceaux valides : {D.verifier()}')
    perdu = D.coffre.chiffres[3].copy()
    rec = D.reconstruire(3)
    dire(f'nœud 3 perdu puis reconstruit sans déchiffrer les autres : '
         f'{np.array_equal(rec, perdu)}')
    return D
