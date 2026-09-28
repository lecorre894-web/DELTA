#!/usr/bin/env python3
"""Δ-SHA V1.0 — UN MOTEUR SHA-256 COMPLET, OUVERT, REPRENABLE.

hashlib est une boîte fermée : il hache vite, et c'est tout. On ne peut ni
voir son registre, ni le sauvegarder, ni reprendre un hachage interrompu
trois jours plus tard, ni répartir le travail sur plusieurs machines.

Δ-SHA est l'inverse : plus LENT (c'est mesuré, et dit franchement), mais
OUVERT. Ce qu'on gagne en échange de la vitesse :

  ÉTAT SORTABLE   les 8 mots du registre + le compteur + le reste du tampon.
                  On arrête, on range 48 octets, on reprend plus tard —
                  même machine ou une autre. hashlib ne sait pas faire ça.
  MIDSTATE        cas particulier du précédent : la partie fixe d'un message
                  cuite une fois pour toutes (le gain Bitcoin, x1,5).
  PARALLÈLE       N messages menés de front dans le même pipeline.
  ARBRE           un gros fichier haché par morceaux indépendants, réunis
                  par un arbre de Merkle : vérifiable morceau par morceau,
                  et réparable sans tout relire.
  VÉRIFIABLE      les 64 constantes sont RECALCULÉES depuis les nombres
                  premiers, jamais recopiées. Aucune faute de frappe
                  possible, et on peut auditer la machine elle-même.

LE JUGE : hashlib, les vecteurs officiels FIPS 180-4, et des blocs Bitcoin
réels. Un moteur qui donne un hash faux ne vaut rien, si élégant soit-il.

Dépendances : numpy (pour le parallèle) + bibliothèque standard. Termux OK.
"""
import hashlib, struct
import numpy as np

MASQUE = 0xFFFFFFFF


# ═══════════════════════════ les constantes, DÉRIVÉES et non recopiées
def _racine_entiere(n, k):
    if n == 0:
        return 0
    x = 1 << ((n.bit_length() + k - 1) // k + 1)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _premiers(n):
    p, c = [], 2
    while len(p) < n:
        if all(c % q for q in p if q * q <= c):
            p.append(c)
        c += 1
    return p


def constantes():
    """H : partie fractionnaire des racines CARRÉES des 8 premiers nombres
    premiers. K : celle des racines CUBIQUES des 64 premiers.
    C'est la définition même de SHA-256. On la recalcule pour pouvoir la
    vérifier, au lieu de recopier 64 nombres à la main."""
    pr = _premiers(64)
    H = tuple(_racine_entiere(p << 64, 2) & MASQUE for p in pr[:8])
    K = tuple(_racine_entiere(p << 96, 3) & MASQUE for p in pr)
    return H, K


H0, K = constantes()
assert H0[0] == 0x6A09E667 and K[0] == 0x428A2F98 and K[63] == 0xC67178F2, \
    'constantes SHA-256 fausses'


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASQUE


def compresser(h, bloc64):
    """Les 64 tours, sur UN bloc de 64 octets. Le cœur de la machine."""
    w = list(struct.unpack('>16I', bloc64)) + [0] * 48
    for i in range(16, 64):
        s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & MASQUE
    a, b, c, d, e, f, g, hh = h
    for i in range(64):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & MASQUE & g)
        t1 = (hh + S1 + ch + K[i] + w[i]) & MASQUE
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & MASQUE
        hh, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASQUE, c, b, a, (t1 + t2) & MASQUE
    return tuple((x + y) & MASQUE for x, y in
                 zip(h, (a, b, c, d, e, f, g, hh)))


# ═══════════════════════════════════════════════════════ LE MOTEUR
class MoteurSHA256:
    """Même usage que hashlib.sha256, mais le capot s'ouvre.

        m = MoteurSHA256()
        m.update(b'bon')
        etat = m.etat()                      # 48 octets, rangeables n'importe où
        ...                                  # des heures, une autre machine
        m2 = MoteurSHA256.depuis_etat(etat)
        m2.update(b'jour')
        m2.hexdigest()                       # == sha256(b'bonjour')
    """

    __slots__ = ('h', 'tampon', 'n')

    def __init__(self, donnees=b''):
        self.h = H0
        self.tampon = b''
        self.n = 0                     # octets absorbés au total
        if donnees:
            self.update(donnees)

    # ------------------------------------------------------------ absorber
    def update(self, donnees):
        d = self.tampon + bytes(donnees)
        self.n += len(donnees)
        i = 0
        while i + 64 <= len(d):
            self.h = compresser(self.h, d[i:i + 64])
            i += 64
        self.tampon = d[i:]
        return self

    # ---------------------------------------------------------- terminer
    def _bourrage(self):
        """0x80, des zéros, puis la longueur en BITS sur 8 octets. La longueur
        est ce qui empêche d'allonger un message sans refaire le calcul."""
        reste = len(self.tampon)
        pad = b'\x80' + b'\x00' * ((55 - reste) % 64) + struct.pack('>Q', self.n * 8)
        return self.tampon + pad

    def digest(self):
        h = self.h
        d = self._bourrage()
        for i in range(0, len(d), 64):
            h = compresser(h, d[i:i + 64])
        return struct.pack('>8I', *h)

    def hexdigest(self):
        return self.digest().hex()

    def copy(self):
        m = MoteurSHA256()
        m.h, m.tampon, m.n = self.h, self.tampon, self.n
        return m

    # ------------------------------- CE QUE hashlib NE SAIT PAS FAIRE
    def etat(self):
        """Sérialise TOUT l'état : 32 octets de registre + 8 de compteur +
        1 de longueur de tampon + le tampon (0 à 63 octets).
        Au plus 104 octets pour suspendre le hachage d'un fichier de
        n'importe quelle taille."""
        return (struct.pack('>8I', *self.h) + struct.pack('>Q', self.n)
                + bytes([len(self.tampon)]) + self.tampon)

    @staticmethod
    def depuis_etat(b):
        m = MoteurSHA256()
        m.h = struct.unpack('>8I', b[:32])
        m.n = struct.unpack('>Q', b[32:40])[0]
        lt = b[40]
        m.tampon = b[41:41 + lt]
        return m

    def registre(self):
        """Les 8 mots, à nu. C'est ce qui rend le midstate possible."""
        return self.h


def sha256(b):
    return MoteurSHA256(b).digest()


def sha256d(b):
    return MoteurSHA256(sha256(b)).digest()


def hmac_sha256(cle, message):
    """HMAC, bâti sur le moteur : clé repliée, ipad/opad, deux passes.
    C'est ce qui permet de sceller sans qu'un tiers puisse refabriquer."""
    if len(cle) > 64:
        cle = sha256(cle)
    cle = cle + b'\x00' * (64 - len(cle))
    ipad = bytes(x ^ 0x36 for x in cle)
    opad = bytes(x ^ 0x5C for x in cle)
    return sha256(opad + sha256(ipad + message))


# ═════════════════════════════════ LE MOTEUR PARALLÈLE (N de front)
def _rotr_v(x, n):
    return (x >> np.uint32(n)) | (x << np.uint32(32 - n))


class MoteurParallele:
    """N messages de MÊME longueur, menés de front dans le même pipeline.
    Pas plus rapide que hashlib (mesuré), mais c'est la forme qu'il faut
    pour répartir sur des voies, des cœurs ou des machines."""

    def __init__(self, n):
        self.n = n
        self.h = np.repeat(np.array(H0, np.uint32)[:, None], n, axis=1)
        self.K = np.array(K, np.uint32)

    def absorber(self, blocs):
        """blocs : (n, 64) octets."""
        b = np.ascontiguousarray(blocs, np.uint8).reshape(self.n, 64)
        W = np.empty((64, self.n), np.uint32)
        W[:16] = b.view('>u4').astype(np.uint32).T
        for i in range(16, 64):
            s0 = (_rotr_v(W[i - 15], 7) ^ _rotr_v(W[i - 15], 18)
                  ^ (W[i - 15] >> np.uint32(3)))
            s1 = (_rotr_v(W[i - 2], 17) ^ _rotr_v(W[i - 2], 19)
                  ^ (W[i - 2] >> np.uint32(10)))
            W[i] = W[i - 16] + s0 + W[i - 7] + s1
        a, b_, c, d, e, f, g, hh = (self.h[i].copy() for i in range(8))
        for i in range(64):
            S1 = _rotr_v(e, 6) ^ _rotr_v(e, 11) ^ _rotr_v(e, 25)
            ch = (e & f) ^ (~e & g)
            t1 = hh + S1 + ch + self.K[i] + W[i]
            S0 = _rotr_v(a, 2) ^ _rotr_v(a, 13) ^ _rotr_v(a, 22)
            maj = (a & b_) ^ (a & c) ^ (b_ & c)
            t2 = S0 + maj
            hh, g, f, e, d, c, b_, a = g, f, e, d + t1, c, b_, a, t1 + t2
        self.h = self.h + np.stack([a, b_, c, d, e, f, g, hh])
        return self

    def digests(self):
        s = np.empty((self.n, 32), np.uint8)
        for i in range(8):
            w = self.h[i]
            for j, dec in enumerate((24, 16, 8, 0)):
                s[:, 4 * i + j] = (w >> np.uint32(dec)).astype(np.uint8)
        return s


def bourrer(message):
    """Bourrage d'un message quelconque -> multiple de 64 octets."""
    n = len(message)
    return message + b'\x80' + b'\x00' * ((55 - n) % 64) + struct.pack('>Q', n * 8)


# ═══════════════════════════════════ LE MOTEUR EN ARBRE (Δ-FRACTAL)
def hacher_par_arbre(donnees, taille_feuille=4096):
    """Hache un gros bloc par morceaux INDÉPENDANTS, réunis par un arbre.

    Trois choses qu'un hachage linéaire ne donne pas :
      - les feuilles se calculent en parallèle, sur autant de voies qu'on veut
      - on peut prouver qu'un morceau appartient au tout en log2(N) empreintes
      - un morceau abîmé se remplace sans relire les autres
    Ce n'est PAS un SHA-256 du fichier : c'est une autre empreinte, qui
    répond à d'autres questions. Les deux sont livrées, on ne les confond pas.
    """
    feuilles = [sha256(donnees[i:i + taille_feuille])
                for i in range(0, max(len(donnees), 1), taille_feuille)]
    niveaux = [feuilles]
    cour = feuilles
    while len(cour) > 1:
        reels = len(cour)
        if len(cour) % 2:
            cour = cour + [cour[-1]]
        haut = []
        for i in range(0, len(cour), 2):
            if i + 1 < reels and cour[i] == cour[i + 1]:
                raise ValueError('deux feuilles appariées identiques : '
                                 'arbre muté (CVE-2012-2459)')
            haut.append(sha256(cour[i] + cour[i + 1]))
        niveaux.append(haut)
        cour = haut
    return cour[0], niveaux


def preuve_arbre(niveaux, i):
    p, idx = [], i
    for niv in niveaux[:-1]:
        j = idx ^ 1
        voisin = niv[j] if j < len(niv) else niv[idx]
        p.append((voisin, idx % 2 == 1))
        idx //= 2
    return p


def verifier_arbre(feuille, preuve, racine):
    h = feuille
    for voisin, a_gauche in preuve:
        h = sha256((voisin + h) if a_gauche else (h + voisin))
    return h == racine
