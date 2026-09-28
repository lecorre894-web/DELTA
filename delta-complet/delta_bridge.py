#!/usr/bin/env python3
"""Δ-BRIDGE V1.0 — LE PONT : on n'envoie plus le calcul tout droit, on
l'extrapole, on le met en parallèle, et on regarde ce qui sort.

L'ASTUCE DE RENÉ, PRISE AU SÉRIEUX
  Au lieu d'appeler la fonction de hachage bloc après bloc comme une boîte
  noire, on l'OUVRE : on expose son registre d'état, son codec d'expansion,
  et on fait passer N messages DE FRONT dans le même pipeline. Puis on
  extrapole : la partie de l'en-tête qui ne bouge jamais est calculée UNE
  fois pour toutes et gardée en registre.

  Le juge est impitoyable et c'est voulu : tout chemin construit ici doit
  redonner, au bit près, les hashs officiels de blocs Bitcoin réels. Un
  chemin élégant qui donne un hash faux ne vaut rien.

CE QUE LE PONT CONTIENT
  RegistreSHA          les 8 mots d'état a..h, en N exemplaires de front
  codec_expansion      W[0..63] : le codec qui étire 16 mots en 64
  compresser           les 64 tours, vectorisés sur N voies
  midstate             L'EXTRAPOLATION : 64 des 80 octets de l'en-tête ne
                       dépendent pas du nonce -> on les cuit une seule fois
  CondensateurExact    condensateur à LONG CHIFFRE, SANS AUCUNE PERTE
  sous_cible           comparaison de N hashs à la cible, d'un seul geste

LA RÈGLE DU PONT, ET ELLE EST ABSOLUE
  Les condensateurs de Δ18 sont À PERTE : 16, 8, 4 ou 2 bits par chiffre.
  On ne fait JAMAIS passer une empreinte par là. Un seul bit faux et le
  hash n'est plus le hash. Le pont vers la cryptographie est donc EXACT ou
  il n'existe pas. C'est une contrainte de conception, pas un détail.

Dépendances : numpy + hashlib. Termux OK.
"""
import hashlib, struct, time
import numpy as np

M32 = np.uint32(0xFFFFFFFF)


# ═══════════════════════════════════ les constantes, DÉRIVÉES et non recopiées
def _racine_entiere(n, k):
    """Racine k-ième entière par Newton : aucune recopie, donc aucune faute
    de frappe possible sur les 64 constantes."""
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


def _constantes():
    """H = partie fractionnaire des racines CARRÉES des 8 premiers nombres
    premiers ; K = celle des racines CUBIQUES des 64 premiers. C'est la
    définition de SHA-256, et on la recalcule au lieu de la copier."""
    pr = _premiers(64)
    # racine_entiere(p * 2^64, 2) = floor(sqrt(p) * 2^32) : la partie
    # fractionnaire est ce qui reste une fois la partie entière ôtée,
    # c'est-à-dire simplement les 32 bits de poids faible.
    H = [_racine_entiere(p << 64, 2) & 0xFFFFFFFF for p in pr[:8]]
    K = [_racine_entiere(p << 96, 3) & 0xFFFFFFFF for p in pr]
    return np.array(H, np.uint32), np.array(K, np.uint32)


H0, K = _constantes()
assert int(H0[0]) == 0x6A09E667 and int(K[0]) == 0x428A2F98, 'constantes SHA-256 fausses'


# ═════════════════════════════════════════════════════ registre et codecs
def rotr(x, n):
    return (x >> np.uint32(n)) | (x << np.uint32(32 - n))


def codec_expansion(bloc):
    """Le CODEC du message : 16 mots en entrée, 64 en sortie.
    bloc : (16, N) uint32  ->  W : (64, N) uint32"""
    n = bloc.shape[1]
    W = np.empty((64, n), np.uint32)
    W[:16] = bloc
    for i in range(16, 64):
        s0 = rotr(W[i - 15], 7) ^ rotr(W[i - 15], 18) ^ (W[i - 15] >> np.uint32(3))
        s1 = rotr(W[i - 2], 17) ^ rotr(W[i - 2], 19) ^ (W[i - 2] >> np.uint32(10))
        W[i] = W[i - 16] + s0 + W[i - 7] + s1
    return W


class RegistreSHA:
    """Les 8 mots d'état, tenus en N exemplaires de front.
    C'est le composant qui manquait : un ÉTAT qu'on peut sortir, garder,
    recharger. Une boîte noire ne le permet pas ; un registre, si."""

    __slots__ = ('h',)

    def __init__(self, h):
        self.h = np.array(h, np.uint32)          # (8, N)

    @staticmethod
    def depart(n):
        return RegistreSHA(np.repeat(H0[:, None], n, axis=1))

    def copie(self):
        return RegistreSHA(self.h.copy())

    def octets(self, i=0):
        return b''.join(int(x).to_bytes(4, 'big') for x in self.h[:, i])

    def nbytes(self):
        return self.h.nbytes


def compresser(reg, W):
    """Les 64 tours de SHA-256, sur N voies à la fois. Aucune boucle sur N."""
    a, b, c, d, e, f, g, h = (reg.h[i].copy() for i in range(8))
    for i in range(64):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + K[i] + W[i]
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        h, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + t2
    return RegistreSHA(np.stack([reg.h[0] + a, reg.h[1] + b, reg.h[2] + c,
                                 reg.h[3] + d, reg.h[4] + e, reg.h[5] + f,
                                 reg.h[6] + g, reg.h[7] + h]))


def blocs_depuis_octets(tampon, n):
    """(n, 64) octets -> (16, n) uint32 gros-boutiste. Le codec d'entrée."""
    a = np.frombuffer(tampon, np.uint8).reshape(n, 64)
    return a.view('>u4').astype(np.uint32).T.copy()


# ════════════════════════════════════════════ L'EXTRAPOLATION : le midstate
def midstate(entete80):
    """Les 64 premiers octets de l'en-tête ne contiennent NI le nonce NI les
    4 derniers octets de l'horodatage : ils ne bougent pas d'un essai à
    l'autre. On les compresse UNE fois et on garde le registre.

    C'est du pur Δ : le gain ne vient pas d'un calcul plus malin, il vient
    de la STRUCTURE du message. Trois compressions deviennent deux."""
    b1 = blocs_depuis_octets(entete80[:64], 1)
    return compresser(RegistreSHA.depart(1), codec_expansion(b1))


def queues(entete80, nonces):
    """Le deuxième bloc, en N exemplaires, un par nonce : 16 octets utiles,
    le bourrage 0x80, des zéros, puis la longueur 640 bits."""
    n = len(nonces)
    q = np.zeros((n, 64), np.uint8)
    q[:, :12] = np.frombuffer(entete80[64:76], np.uint8)
    q[:, 12:16] = np.frombuffer(np.asarray(nonces, '<u4').tobytes(),
                                np.uint8).reshape(n, 4)
    q[:, 16] = 0x80
    q[:, 62] = (640 >> 8) & 0xFF
    q[:, 63] = 640 & 0xFF
    return q


def sha256d_pont(entete80, nonces, mid=None):
    """Le chemin complet du pont : N nonces traités de front, en repartant
    d'un midstate déjà cuit. Renvoie (n, 32) octets, ordre interne."""
    n = len(nonces)
    mid = mid if mid is not None else midstate(entete80)
    reg = RegistreSHA(np.repeat(mid.h, n, axis=1))
    r1 = compresser(reg, codec_expansion(blocs_depuis_octets(queues(entete80, nonces), n)))
    # deuxième SHA : le condensé de 32 octets + bourrage + longueur 256 bits
    b2 = np.zeros((16, n), np.uint32)
    b2[:8] = r1.h
    b2[8] = np.uint32(0x80000000)
    b2[15] = np.uint32(256)
    r2 = compresser(RegistreSHA.depart(n), codec_expansion(b2))
    sortie = np.empty((n, 32), np.uint8)
    for i in range(8):
        w = r2.h[i]
        sortie[:, 4 * i + 0] = (w >> np.uint32(24)).astype(np.uint8)
        sortie[:, 4 * i + 1] = (w >> np.uint32(16)).astype(np.uint8)
        sortie[:, 4 * i + 2] = (w >> np.uint32(8)).astype(np.uint8)
        sortie[:, 4 * i + 3] = w.astype(np.uint8)
    return sortie


# ══════════════════════════════ CONDENSATEUR EXACT À LONG CHIFFRE
class CondensateurExact:
    """Condensateur à LONG CHIFFRE, sans aucune perte.

    Celui de Δ18 range des nombres FLOTTANTS en 16, 8, 4 ou 2 bits : il perd,
    et c'est assumé là-bas. Ici on range des chiffres de 128, 256 ou 512 bits
    — des empreintes, des clés, des cibles — et on n'a PAS LE DROIT de perdre
    le moindre bit. Le chiffre est découpé en membres de 32 bits rangés en
    colonnes, ce qui permet de comparer, trier ou combiner N chiffres longs
    d'un seul geste, sans jamais repasser par un entier Python.
    """

    def __init__(self, donnees, bits=256):
        assert bits % 32 == 0
        self.bits, self.membres = bits, bits // 32
        a = np.frombuffer(donnees, np.uint8) if isinstance(donnees, (bytes, bytearray)) \
            else np.ascontiguousarray(donnees, np.uint8)
        a = a.reshape(-1, bits // 8)
        self.n = a.shape[0]
        self.mots = a.view('>u4').astype(np.uint32)     # (n, membres), poids fort d'abord

    def octets(self):
        return self.mots.nbytes

    def lire(self, i=None):
        """Restitution EXACTE. Aucune tolérance, aucun epsilon : c'est le
        même objet ou ce n'est pas lui."""
        m = self.mots if i is None else self.mots[i:i + 1]
        return m.astype('>u4').tobytes()

    def entiers(self):
        """Vers des entiers Python — lent, réservé au contrôle."""
        return [int.from_bytes(self.lire(i), 'big') for i in range(self.n)]

    def sous(self, cible_octets):
        """Les N chiffres sont-ils sous la cible ? Comparaison lexicographique
        des membres, du plus fort au plus faible, sans conversion."""
        c = np.frombuffer(cible_octets, np.uint8).view('>u4').astype(np.uint32)
        inf = np.zeros(self.n, bool)
        egal = np.ones(self.n, bool)
        for k in range(self.membres):
            inf |= egal & (self.mots[:, k] < c[k])
            egal &= (self.mots[:, k] == c[k])
        return inf

    def parite(self):
        """XOR de tous les chiffres : une signature de lot, exacte."""
        p = np.zeros(self.membres, np.uint32)
        for k in range(self.membres):
            p[k] = np.bitwise_xor.reduce(self.mots[:, k])
        return p.astype('>u4').tobytes()


def sous_cible(hashs, cible_int):
    """N empreintes (n,32) en ordre interne -> lesquelles sont sous la cible."""
    gros = hashs[:, ::-1].copy()                       # ordre d'affichage
    c = cible_int.to_bytes(32, 'big')
    return CondensateurExact(gros.tobytes(), 256).sous(c)


# ══════════════════════════ CODEC VARINT + TXID : couper le dernier fil
class Ruban:
    """Un ruban d'octets qu'on déroule. C'est le composant qui manquait pour
    lire une transaction BRUTE au lieu de croire un explorateur sur parole."""

    def __init__(self, b):
        self.b, self.i = b, 0

    def prendre(self, n):
        x = self.b[self.i:self.i + n]
        self.i += n
        return x

    def u8(self):
        return self.prendre(1)[0]

    def u32(self):
        return int.from_bytes(self.prendre(4), 'little')

    def u64(self):
        return int.from_bytes(self.prendre(8), 'little')

    def varint(self):
        """CompactSize : 1 octet, ou un préfixe qui annonce 2, 4 ou 8 octets.
        Le codec le plus simple de Bitcoin, et pourtant indispensable : sans
        lui, impossible de savoir où finit une entrée et où commence la
        suivante."""
        p = self.u8()
        if p < 0xFD:
            return p
        if p == 0xFD:
            return int.from_bytes(self.prendre(2), 'little')
        if p == 0xFE:
            return int.from_bytes(self.prendre(4), 'little')
        return int.from_bytes(self.prendre(8), 'little')


def txid_depuis_brut(brut):
    """Recalcule le txid depuis les octets de la transaction.

    Subtilité qui compte : depuis SegWit (2017), une transaction peut porter
    un marqueur 0x00 0x01 et des TÉMOINS. Le txid, lui, se calcule sur la
    transaction DÉPOUILLÉE de ses témoins. Hacher les octets bruts tels
    quels donnerait le wtxid, pas le txid. C'est exactement le genre de
    détail qui fait qu'un pont a l'air de marcher et ne marche pas.
    Renvoie (txid affiché, segwit ?)."""
    r = Ruban(brut)
    debut = r.i
    version = r.prendre(4)
    segwit = len(brut) > 5 and brut[4] == 0x00 and brut[5] == 0x01
    if segwit:
        r.prendre(2)
    corps = bytearray()
    n_in = r.varint()
    d = r.i
    for _ in range(n_in):
        r.prendre(36)                       # point de dépense
        r.prendre(r.varint())               # script de déverrouillage
        r.prendre(4)                        # numéro de séquence
    n_out = r.varint()
    for _ in range(n_out):
        r.u64()                             # montant
        r.prendre(r.varint())               # script de verrouillage
    fin_corps = r.i
    corps += encoder_varint(n_in) + brut[d:fin_corps]
    if segwit:
        for _ in range(n_in):
            for _ in range(r.varint()):
                r.prendre(r.varint())
    verrou = r.prendre(4)
    depouillee = version + bytes(corps) + verrou
    return hashlib.sha256(hashlib.sha256(depouillee).digest()).digest()[::-1].hex(), segwit


def encoder_varint(n):
    if n < 0xFD:
        return bytes([n])
    if n <= 0xFFFF:
        return b'\xfd' + n.to_bytes(2, 'little')
    if n <= 0xFFFFFFFF:
        return b'\xfe' + n.to_bytes(4, 'little')
    return b'\xff' + n.to_bytes(8, 'little')


# ═════════════════════════════════ ce qui manque encore, et à quoi ça sert
COMPOSANTS = [
    ('registre d\'état SHA-256', 'FAIT',
     'les 8 mots a..h sortis de la boîte noire : c\'est lui qui rend le '
     'midstate possible'),
    ('codec d\'expansion W[0..63]', 'FAIT',
     'étire 16 mots en 64 ; vectorisé, il traite N messages de front'),
    ('codec d\'en-tête 80 octets', 'FAIT (delta_btc)',
     'sérialisation petit-boutiste ; l\'inversion des txid est la faute n°1'),
    ('midstate / extrapolation', 'FAIT',
     'trois compressions deviennent deux : gain de structure, pas de magie'),
    ('condensateur EXACT à long chiffre', 'FAIT',
     '128 à 512 bits par chiffre, zéro perte ; le pont vers la crypto ne '
     'tolère aucun epsilon'),
    ('comparateur de cible vectoriel', 'FAIT',
     'N empreintes jugées d\'un geste, sans entier Python'),
    ('codec varint (CompactSize) + txid', 'FAIT',
     'lit une transaction BRUTE et recalcule son txid soi-même, témoins '
     'SegWit retirés : le dernier fil de confiance envers l\'explorateur '
     'est coupé'),
    ('codec Base58Check', 'MANQUANT',
     'adresses historiques 1... ; somme de contrôle sur 4 octets'),
    ('codec Bech32 / Bech32m', 'MANQUANT',
     'adresses bc1... ; code correcteur BCH qui DÉSIGNE le caractère fautif '
     '— exactement l\'esprit de l\'ECC de Δ9'),
    ('registre secp256k1 (ECDSA / Schnorr)', 'MANQUANT, et le plus lourd',
     'vérifier une SIGNATURE, pas seulement une empreinte. C\'est là que '
     'l\'arithmétique modulaire à long chiffre servirait vraiment, et c\'est '
     'là aussi qu\'une bibliothèque éprouvée doit prendre le relais'),
    ('arbre de Merkle témoin (SegWit)', 'MANQUANT',
     'racine des témoins, rangée dans la coinbase depuis 2017'),
]
