#!/usr/bin/env python3
"""Δ18 V1.8 — CONDENSATEURS ET CRYPTOGRAPHIE : ce qu'un QPU ne peut pas faire.

Un état quantique ne peut pas être signé tant qu'il est cohérent : le lire pour
le sceller le détruit, et on ne peut pas le copier (théorème de non-clonage).
Δ, lui, est classique de bout en bout : son état PEUT être scellé, chiffré,
vérifié et reconstruit. C'est un avantage propre à l'émulation, pas un défaut.

CONDENSATEUR (plusieurs chiffres dans une cellule)
  chaque valeur est quantifiée sur k bits, puis PLUSIEURS valeurs sont
  empaquetées dans un même mot de 64 bits. Une cellule retient donc plusieurs
  chiffres. Le % (P) choisit k : 16 bits pour le cœur, 8 ou 4 bits au loin.

CRYPTOGRAPHIE
  sceau à clé (BLAKE2b avec clé) : détecte la corruption ET la falsification,
    là où un CRC se refabrique en une seconde
  arbre de Merkle : une seule racine authentifie tout l'état, et en cas
    d'altération l'arbre DÉSIGNE le bloc fautif en log2(n) comparaisons
  chiffrement au repos : flux pseudo-aléatoire tiré de BLAKE2b en mode
    compteur, appliqué par XOR
  PROPRIÉTÉ UTILE : la parité XOR des blocs CHIFFRÉS reste valide. Un nœud
  perdu se reconstruit sans jamais déchiffrer les autres.

AVERTISSEMENT : ceci est une démonstration pédagogique bâtie sur la
bibliothèque standard. Pour un usage réel, utiliser une bibliothèque
cryptographique éprouvée (AES-GCM, ChaCha20-Poly1305).

Dépendances : numpy + hashlib. Termux OK.
"""
import hashlib, hmac, math, time
import numpy as np


# ───────────────────────────────────────────────── condensateur multi-chiffres
class Condensateur:
    """Empaquette plusieurs valeurs quantifiées dans des mots de 64 bits."""

    def __init__(self, x, bits=16):
        assert 64 % bits == 0, 'bits doit diviser 64'
        self.bits, self.n = bits, x.size
        self.lo = float(x.min())
        self.hi = float(x.max())
        ech = (self.hi - self.lo) or 1.0
        niveaux = (1 << bits) - 1
        q = np.rint((x.astype(np.float64) - self.lo) / ech * niveaux).astype(np.uint64)
        par_mot = 64 // bits
        pad = (-q.size) % par_mot
        q = np.concatenate([q, np.zeros(pad, np.uint64)])
        q = q.reshape(-1, par_mot)
        mot = np.zeros(q.shape[0], np.uint64)
        for j in range(par_mot):
            mot |= q[:, j] << np.uint64(j * bits)
        self.mots = mot
        self.par_mot = par_mot

    def octets(self):
        return self.mots.nbytes

    def lire(self):
        masque = np.uint64((1 << self.bits) - 1)
        vals = np.empty((self.mots.size, self.par_mot), np.uint64)
        for j in range(self.par_mot):
            vals[:, j] = (self.mots >> np.uint64(j * self.bits)) & masque
        q = vals.reshape(-1)[:self.n].astype(np.float64)
        ech = (self.hi - self.lo) or 1.0
        return self.lo + q / ((1 << self.bits) - 1) * ech


def bits_pour(P):
    """% -> nombre de chiffres par cellule. 100 % = 16 bits, 50 % = 8, 10 % = 4."""
    return 16 if P >= 1.0 else (8 if P >= 0.5 else 4)


# ─────────────────────────────────────────────────────────── cryptographie
def sceau(cle, donnees):
    """Sceau à clé : détecte la corruption ET la falsification."""
    return hashlib.blake2b(np.ascontiguousarray(donnees).view(np.uint8).tobytes(),
                           key=cle, digest_size=16).digest()


def flux(cle, compteur, n):
    """Flux pseudo-aléatoire (BLAKE2b en mode compteur) pour le chiffrement."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.blake2b(compteur.to_bytes(8, 'little') + i.to_bytes(8, 'little'),
                               key=cle, digest_size=64).digest()
        i += 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8)


def chiffrer(cle, compteur, octets):
    return octets ^ flux(cle, compteur, len(octets))


class Merkle:
    """Arbre de Merkle : une racine pour tout l'état, et le bloc fautif désigné."""

    def __init__(self, blocs, cle=b'', strict=False):
        """`strict=True` : refuse un arbre MUTÉ.

        Correctif hérité du test de conformité Bitcoin (CVE-2012-2459) : en
        appariant le dernier nœud d'un niveau impair avec lui-même, on rend
        deux listes DIFFÉRENTES capables de produire la MÊME racine. Un
        attaquant s'en sert pour faire passer un faux état pour le vrai.
        La parade, celle de Bitcoin Core : si deux nœuds APPARIÉS et tous
        deux réellement présents sont identiques, l'arbre est rejeté.
        Défaut à False pour ne pas casser les usages existants ; à mettre à
        True dès qu'un tiers peut choisir le contenu des blocs."""
        self.cle = cle
        self.strict = strict
        self.feuilles = [hashlib.blake2b(b.tobytes(), key=cle, digest_size=16).digest()
                         for b in blocs]
        self.niveaux = [list(self.feuilles)]
        cour = self.feuilles
        while len(cour) > 1:
            suiv = []
            for i in range(0, len(cour), 2):
                if strict and i + 1 < len(cour) and cour[i] == cour[i + 1]:
                    raise ValueError('deux nœuds appariés identiques : arbre muté '
                                     '(CVE-2012-2459)')
                d = cour[i] + (cour[i + 1] if i + 1 < len(cour) else cour[i])
                suiv.append(hashlib.blake2b(d, key=cle, digest_size=16).digest())
            self.niveaux.append(suiv)
            cour = suiv
        self.racine = cour[0]

    def verifier(self, blocs):
        """Renvoie (tout_bon, indices des blocs fautifs) en O(n) de hachage."""
        f = [hashlib.blake2b(b.tobytes(), key=self.cle, digest_size=16).digest()
             for b in blocs]
        mauvais = [i for i, (a, b) in enumerate(zip(f, self.feuilles)) if a != b]
        return (not mauvais), mauvais

    def preuve(self, i):
        """Chemin d'authentification : log2(n) empreintes, chacune avec son côté."""
        p, idx = [], i
        for niv in self.niveaux[:-1]:
            j = idx ^ 1
            voisin = niv[j] if j < len(niv) else niv[idx]
            p.append((voisin, idx % 2 == 1))       # True = le voisin est à GAUCHE
            idx //= 2
        return p

    @staticmethod
    def verifier_preuve(bloc, preuve, racine, cle=b''):
        h = hashlib.blake2b(bloc.tobytes(), key=cle, digest_size=16).digest()
        for voisin, a_gauche in preuve:
            d = (voisin + h) if a_gauche else (h + voisin)
            h = hashlib.blake2b(d, key=cle, digest_size=16).digest()
        return h == racine


class CoffreDelta:
    """État Δ scellé, chiffré, réparti, avec parité sur le CHIFFRÉ."""

    def __init__(self, cle, blocs):
        self.cle = cle
        self.clairs = blocs
        self.octets = [np.ascontiguousarray(b).view(np.uint8).ravel() for b in blocs]
        self.chiffres = [chiffrer(cle, i, o) for i, o in enumerate(self.octets)]
        self.sceaux = [sceau(cle, c) for c in self.chiffres]
        self.merkle = Merkle(self.chiffres, cle)

    def taille(self):
        return sum(c.nbytes for c in self.chiffres)

    def verifier(self):
        return all(hmac.compare_digest(sceau(self.cle, c), s)
                   for c, s in zip(self.chiffres, self.sceaux))

    def dechiffrer(self, i):
        return chiffrer(self.cle, i, self.chiffres[i])

    def parite(self):
        t = max(c.size for c in self.chiffres)
        p = np.zeros(t, np.uint8)
        for c in self.chiffres:
            p[:c.size] ^= c
        return p

    def reconstruire(self, i, parite):
        """Reconstruit le bloc i depuis la parité, SANS déchiffrer les autres."""
        p = parite.copy()
        for j, c in enumerate(self.chiffres):
            if j != i:
                p[:c.size] ^= c
        return p[:self.chiffres[i].size]
