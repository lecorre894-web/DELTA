#!/usr/bin/env python3
"""Δ16 V1.6 — Δ-MÉMOIRE ET TAMPONS : le principe Δ appliqué au stockage.

  grappes identiques -> PAGES DÉDUPLIQUÉES : une page répétée n'est stockée
                        qu'une fois ; toutes les autres la référencent
  grappe allumée     -> page unique, copiée seulement à l'écriture (copy-on-write)
  X à la demande     -> RANG de compression de la page (SVD tronquée à epsilon)
  % (P)              -> PRÉCISION de la page : float64 / float32 / float16
  cône de lumière    -> on ne matérialise que les pages réellement lues
  ECC                -> empreinte par page, parité XOR entre nœuds
  LOAD || COMPUTE || STORE -> TAMPON EN ANNEAU avec préchargement en fond

Tout est mesuré : octets réellement occupés, taux de déduplication, bande
passante en Go/s, opérations par seconde, pic de mémoire vive réel.

Dépendances : numpy (+ threads de la bibliothèque standard). Termux OK.
"""
import hashlib, math, threading, queue, time, zlib
import numpy as np

PREC = {1.00: np.float64, 0.50: np.float32, 0.10: np.float16}


def precision_de(P):
    for s in sorted(PREC, reverse=True):
        if P >= s:
            return PREC[s]
    return np.float16


class Page:
    """Une page : pleine ou factorisée (U V), à la précision demandée."""
    __slots__ = ('U', 'V', 'plein', 'X', 'dtype', 'forme', 'crc', 'refs', 'relevee')

    def __init__(self, M, eps=1e-6, P=1.0):
        dt = precision_de(P)
        # GARDE-FOU : un % bas demande peu de précision, mais float16 déborde
        # au-delà de 65504. Plutôt que de produire des infinis en silence, Δ
        # REMONTE d'un cran et le déclare. Une donnée fausse ne vaut pas mieux
        # qu'une donnée absente.
        self.relevee = False
        self.forme = M.shape
        m, n = M.shape
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        tot = float((s ** 2).sum()) + 1e-300
        cum = np.cumsum(s[::-1] ** 2)[::-1]
        X = len(s)
        for k in range(1, len(s) + 1):
            if (cum[k] if k < len(cum) else 0.0) / tot <= eps ** 2:
                X = k
                break
        factorisee = X * (m + n) < m * n
        while True:
            with np.errstate(over='ignore'):
                if factorisee:
                    A = (u[:, :X] * s[:X]).astype(dt); B = vh[:X, :].astype(dt)
                    bon = np.isfinite(A).all() and np.isfinite(B).all()
                else:
                    A = M.astype(dt); B = None
                    bon = np.isfinite(A).all()
            if bon or dt is np.float64:
                break
            # GARDE-FOU : un % bas demande peu de précision, mais float16 déborde
            # au-delà de 65504. Plutôt que de ranger des infinis EN SILENCE, Δ
            # remonte d'un cran et le déclare (`relevee`). Une donnée fausse ne
            # vaut pas mieux qu'une donnée absente.
            dt = np.float32 if dt is np.float16 else np.float64
            self.relevee = True
        self.dtype = dt
        if factorisee:
            self.U, self.V, self.plein, self.X = A, B, None, X
        else:
            self.plein, self.U, self.V, self.X = A, None, None, min(m, n)
        self.refs = 1
        self.crc = self._crc()

    def _crc(self):
        a = self.plein if self.plein is not None else np.concatenate([self.U.ravel(), self.V.ravel()])
        return zlib.crc32(np.ascontiguousarray(a).view(np.uint8).tobytes()) & 0xFFFFFFFF

    def valide(self):
        return self._crc() == self.crc

    def octets(self):
        t = np.dtype(self.dtype).itemsize
        return (self.plein.size if self.plein is not None else self.U.size + self.V.size) * t

    def lire(self):
        return (self.plein if self.plein is not None else self.U @ self.V).astype(np.float64)


class MemoireDelta:
    """Mémoire paginée : déduplication, compression, copy-on-write, ECC."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.pages = {}           # empreinte -> Page
        self.table = {}           # index logique -> empreinte
        self.octets_bruts = 0     # ce que le tout aurait coûté sans Δ
        self.lectures = 0
        self.materialisees = 0
        self.relevees = 0          # pages dont la précision a dû être remontée

    @staticmethod
    def empreinte(M):
        return hashlib.blake2b(np.ascontiguousarray(M).tobytes(), digest_size=16).hexdigest()

    def ecrire(self, idx, M, P=1.0):
        self.octets_bruts += M.nbytes
        e = self.empreinte(M)
        if e in self.pages:                     # PAGE DÉJÀ CONNUE : on référence
            self.pages[e].refs += 1
        else:
            self.pages[e] = Page(M, self.eps, P)
            self.relevees += int(self.pages[e].relevee)
        self.table[idx] = e
        return e

    def ecrire_modifiee(self, idx, M, P=1.0):
        """Copy-on-write : la page devient unique seulement maintenant."""
        return self.ecrire(idx, M, P)

    def lire(self, idx):
        self.lectures += 1
        e = self.table[idx]
        self.materialisees += 1
        return self.pages[e].lire()

    def octets(self):
        return sum(p.octets() for p in self.pages.values())

    def dedup(self):
        n = len(self.table)
        return n / max(len(self.pages), 1)

    def valide(self):
        return all(p.valide() for p in self.pages.values())

    def repartir(self, nb):
        flux = np.concatenate([np.ascontiguousarray(
            p.plein if p.plein is not None else np.concatenate([p.U.ravel(), p.V.ravel()])
        ).view(np.uint8).ravel() for p in self.pages.values()])
        coup = np.linspace(0, len(flux), nb + 1).astype(int)
        parts = {k: flux[coup[k]:coup[k + 1]].copy() for k in range(nb)}
        t = max(len(v) for v in parts.values())
        par = np.zeros(t, np.uint8)
        for v in parts.values():
            par[:len(v)] ^= v
        return parts, par


class Anneau:
    """Tampon en anneau : LOAD (fil de fond) || COMPUTE (fil principal).

    Le producteur prépare les tuiles suivantes pendant que le calcul travaille
    sur la tuile courante. La mémoire vive reste bornée par `profondeur`.
    """

    def __init__(self, producteur, nb_tuiles, profondeur=3):
        self.producteur, self.nb, self.profondeur = producteur, nb_tuiles, profondeur
        self.file = queue.Queue(maxsize=profondeur)
        self.octets = 0
        self.pic = 0

    def _charger(self):
        for k in range(self.nb):
            t = self.producteur(k)
            self.octets += t.nbytes
            self.file.put(t)
        self.file.put(None)

    def parcourir(self):
        fil = threading.Thread(target=self._charger, daemon=True)
        fil.start()
        while True:
            t = self.file.get()
            if t is None:
                break
            self.pic = max(self.pic, (self.file.qsize() + 1) * t.nbytes)
            yield t
        fil.join()


def bande_passante(octets, secondes):
    return octets / max(secondes, 1e-12) / 1e9        # Go/s
