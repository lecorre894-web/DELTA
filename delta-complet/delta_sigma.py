#!/usr/bin/env python3
"""Δ-SIGMA V1.0 — LE FACTEUR Σ ET ∫, UN SEUL BLOC TEXTE, UN SEUL HASH.

    Σ  la somme      — ce que valent TOUS les grains réunis
    ∫  l'intégrale   — ce que vaut le fond CONTINU, entre deux bornes

Jusqu'ici Δ répondait grain par grain. Ici il répond d'un seul nombre, puis
d'un seul texte, puis d'une seule empreinte. Et cette empreinte, on la
confronte à Δ complet : si Δ est vraiment déterministe, relancer tout depuis
zéro doit redonner le MÊME hash, bit pour bit. Sinon, Δ se contredit.

DEUX PIÈGES, ET ILS SONT RÉELS
  1. Σ naïve MENT. Additionner des millions de flottants l'un après l'autre
     accumule l'erreur d'arrondi. La somme compensée (Neumaier) rattrape le
     morceau perdu à chaque addition. L'écart est mesuré ici, pas supposé.
  2. Un bloc texte qui contient une DURÉE n'est pas reproductible. On sépare
     donc net : ce qui est déterministe entre dans le texte et dans le hash ;
     ce qui dépend de la machine et de l'humeur du moment reste dehors, et on
     le dit.

Dépendances : numpy + hashlib. Termux OK.
"""
import hashlib
import numpy as np


# ══════════════════════════════════════════════════════════ Σ — LA SOMME
def somme_naive(x):
    s = 0.0
    for v in x:
        s += float(v)
    return s


def somme_compensee(x):
    """Neumaier : à chaque addition, on récupère le morceau que l'arrondi
    allait jeter, et on le garde de côté. À la fin, on le rend."""
    s = 0.0
    c = 0.0
    for v in x:
        v = float(v)
        t = s + v
        if abs(s) >= abs(v):
            c += (s - t) + v
        else:
            c += (v - t) + s
        s = t
    return s + c


class Sigma:
    """Σ sur les grains : la somme de tout, exacte autant qu'on peut l'être,
    et qui déclare de combien la somme naïve se serait trompée."""

    def __init__(self):
        self.s = 0.0
        self.c = 0.0
        self.naive = 0.0
        self.n = 0

    def ajouter(self, bloc):
        b = np.asarray(bloc, np.float64)
        self.n += b.size
        self.naive += float(b.sum())
        for v in b:
            v = float(v)
            t = self.s + v
            if abs(self.s) >= abs(v):
                self.c += (self.s - t) + v
            else:
                self.c += (v - t) + self.s
            self.s = t
        return self

    def valeur(self):
        return self.s + self.c

    def ecart_naif(self):
        return abs(self.naive - self.valeur())


# ═════════════════════════════════════════════════════ ∫ — L'INTÉGRALE
def integrale_simpson(f, a, b, n):
    """Simpson composite. n doit être pair. Erreur en (b-a)^5/n^4 : on peut
    donc dire d'avance de combien on se trompe, et le vérifier après."""
    if n % 2:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())


def noyau(d, h=1e-3):
    """Le fond de Δ, celui de Δ14/Δ15."""
    return 1.0 / (1.0 + np.abs(d) * h)


def integrale_exacte(a, b, h=1e-3):
    """∫ 1/(1+h x) dx = ln(1+h b)/h - ln(1+h a)/h, pour 0 <= a <= b.
    On a donc une VÉRITÉ à opposer au calcul numérique."""
    return (np.log(1.0 + h * b) - np.log(1.0 + h * a)) / h


# ═══════════════════════════════ LE BLOC UNIQUE, ET SON EMPREINTE
def ligne(cle, valeur):
    """Une ligne canonique. Format figé : sans cela, deux exécutions
    identiques produiraient deux textes différents, donc deux hashs."""
    if isinstance(valeur, float):
        v = f'{valeur:.17g}'
    elif isinstance(valeur, (bytes, bytearray)):
        v = valeur.hex()
    else:
        v = str(valeur)
    return f'{cle}={v}'


def bloc_texte(lignes):
    """UN SEUL bloc. Trié, normalisé, terminé par un retour à la ligne."""
    return '\n'.join(sorted(lignes)) + '\n'


def sha256d(b):
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def signature(texte):
    """L'empreinte de tout Δ, calculée avec le double SHA-256 de Bitcoin —
    le seul que nous ayons vérifié conforme sur des blocs réels."""
    return sha256d(texte.encode('utf-8'))[::-1].hex()
