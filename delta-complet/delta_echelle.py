#!/usr/bin/env python3
"""Δ-ECHELLE V1.0 — 10 MILLIARDS DE QUBITS SOUS 25 Gio, INTRICATION REMISE.

L'OBJECTIF, ET LE FACTEUR À TROUVER
  Mesuré en Δ3 : 10 000 000 qubits en 2,53 Gio, soit 272 octets par qubit.
  Demandé : 10 000 000 000 qubits sous 25 Gio, soit 2,68 octets par qubit.
  Facteur à gagner : x101. Ce n'est pas un réglage, c'est un changement de
  représentation.

OÙ LES 272 OCTETS PARTENT, ET CE QU'ON PEUT EN RETIRER
  Un site MPS porte un tenseur Gamma de forme (X, 2, X) plus un vecteur
  lambda de longueur X. À X=4 en complex64 : (4x2x4 + 4) x 8 = 288 octets.
    X de 4 à 2 .................. 288 -> 80 octets   (x3,6)
    complex64 -> reel float32 ...  80 -> 40 octets   (x2)
    condensateur Δ18 a 2 bits ...  40 -> 2,5 octets  (x16)
  Total : x115. Le compte y est — et chaque etage a son prix, mesure ici.

CE QUI RESTE HONNÊTE, ET CE QUI NE L'EST PAS
  X=2 borne l'intrication a un bit par coupure. On ne peut donc PAS mettre
  dix milliards de qubits fortement intriques sous 25 Gio : ce serait faux.
  Ce qu'on peut faire, et qui est le canon de Δ depuis Δ5 : une MASSE a
  faible intrication, et des GRAPPES pleinement individualisees la ou
  l'intrication compte vraiment. Les deux sont mesures separement.

Dépendances : numpy. Termux OK.
"""
import numpy as np

OCTET_GIO = 2 ** 30


# ═══════════════════════════════════ ce que coûte un qubit, réellement
def cout_site(X, dtype=np.complex64, bits=None):
    """Octets par site MPS : Gamma (X,2,X) + lambda (X).
    `bits` = quantification Δ18 : chaque nombre tient sur `bits` bits."""
    nombres = X * 2 * X + X
    if bits is None:
        return nombres * np.dtype(dtype).itemsize
    reel = 2 if np.issubdtype(dtype, np.complexfloating) else 1
    return nombres * reel * bits / 8.0


def qubits_tenables(octets, X, dtype=np.complex64, bits=None):
    return int(octets / cout_site(X, dtype, bits))


def dimensionner(cible=10 ** 10, budget_gio=25.0):
    """Cherche toutes les configurations qui logent `cible` qubits sous le
    budget, et rend la plus fine (X le plus grand, bits le plus grand)."""
    budget = budget_gio * OCTET_GIO
    bons = []
    for X in (8, 4, 3, 2):
        for dt in (np.complex64, np.float32):
            for b in (None, 16, 8, 4, 2):
                c = cout_site(X, dt, b)
                if c * cible <= budget:
                    bons.append({'X': X, 'dtype': dt.__name__, 'bits': b,
                                 'octets_site': c,
                                 'total_gio': c * cible / OCTET_GIO})
    bons.sort(key=lambda d: (-d['X'], -(d['bits'] or 64)))
    return bons


# ═══════════════════════════ une chaîne réelle, pour mesurer la fidélité
def chaine(n, X, graine=0, dtype=np.complex64):
    """Une chaîne MPS en forme de Vidal, à intrication NON nulle.
    On ne fabrique pas un état produit déguisé : chaque lambda a plusieurs
    valeurs non nulles, donc chaque coupure est vraiment intriquée."""
    g = np.random.default_rng(graine)
    Gam = [g.standard_normal((X, 2, X)).astype(dtype) for _ in range(n)]
    lam = []
    for _ in range(n + 1):
        v = np.abs(g.standard_normal(X)) + 0.15
        lam.append((v / np.linalg.norm(v)).astype(np.float32))
    return Gam, lam


def entropie_coupure(l):
    p = np.asarray(l, float) ** 2
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def quantifier(A, bits):
    """Condensateur Δ18 appliqué à un tenseur : on range sur `bits` bits
    et on relit. Ce qui sort n'est PAS ce qui est entré, et l'écart est
    exactement ce qu'on accepte de perdre."""
    x = np.asarray(A)
    if np.issubdtype(x.dtype, np.complexfloating):
        return (quantifier(x.real, bits) + 1j * quantifier(x.imag, bits)).astype(x.dtype)
    lo, hi = float(x.min()), float(x.max())
    ech = (hi - lo) or 1.0
    niv = (1 << bits) - 1
    q = np.rint((x - lo) / ech * niv)
    return (lo + q / niv * ech).astype(x.dtype)


def fidelite_locale(Gam, lam, bits, sites=64):
    """Compare le tenseur local exact et sa version quantifiée, site par
    site. C'est une fidélité LOCALE : elle ne prétend pas mesurer l'état
    global, qui n'est pas calculable à cette échelle. On le dit."""
    fs = []
    for i in range(min(sites, len(Gam))):
        A = Gam[i]
        B = quantifier(A, bits)
        a, b = A.ravel(), B.ravel()
        f = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        fs.append(float(f))
    return float(np.mean(fs)), float(np.min(fs))


# ═══════════════════════════════ l'intrication d'origine : les îlots de Δ1
def ilots_fusion(n_ilots=6, X=4, graine=1):
    """L'intrication flottante de Δ1 : des îlots séparés, qu'on fusionne,
    puis qu'on teste pour savoir s'ils peuvent être RECOUPÉS.
    Un îlot qui se recoupe n'était pas vraiment intriqué ; un îlot qui
    refuse la coupure l'est. C'est le test d'origine, remis tel quel."""
    g = np.random.default_rng(graine)
    etats = [g.standard_normal((2, 2)) + 1j * g.standard_normal((2, 2))
             for _ in range(n_ilots)]
    etats = [e / np.linalg.norm(e) for e in etats]
    res = []
    for e in etats:
        s = np.linalg.svd(e, compute_uv=False)
        res.append({'schmidt': [float(x) for x in s],
                    'entropie': entropie_coupure(s),
                    'coupable': bool(s[1] <= 1e-12 * s[0])})
    # une fusion : produit tensoriel de deux ilots -> doit rester coupable
    f = np.kron(etats[0].ravel(), etats[1].ravel()).reshape(4, 4)
    sf = np.linalg.svd(f, compute_uv=False)
    return res, {'schmidt': [float(x) for x in sf],
                 'coupable': bool(sf[1] <= 1e-10 * sf[0])}
