#!/usr/bin/env python3
"""Δ19 V1.9 — Δ-X3D : hiérarchie de cache à 256 niveaux, l'éviction devient
dégradation.

DEUX IDÉES, ET ELLES TIENNENT ENSEMBLE

1) RÉPERTOIRE GLOBAL -> le nombre de niveaux ne coûte RIEN en latence.
   Un cache matériel classique s'arrête à 3 ou 4 niveaux parce qu'il les
   interroge l'un après l'autre : chaque étage ajoute sa latence. Δ tient un
   répertoire unique (adresse -> niveau) : UNE seule recherche, quel que soit
   le nombre d'étages. 256 niveaux coûtent donc autant qu'un seul.

2) L'ÉVICTION DEVIENT UNE DÉGRADATION -> plus rien n'est jamais perdu.
   Quand un étage déborde, sa ligne la plus froide NE DISPARAÎT PAS : elle
   descend d'un étage et y est stockée plus comprimée (moins de bits par
   chiffre, rang plus bas). Une donnée très ancienne finit en 2 bits par
   chiffre, floue mais présente. La capacité apparente devient immense, et
   l'erreur croît doucement au lieu de tomber d'un coup à « absent ».

  grappes    : les lignes identiques ne sont stockées qu'une fois, à tous les étages
  % (P)      : une ligne à 100 % ne descend jamais
  condensateur (Δ18) : c'est lui qui porte la dégradation, bits par bits
  ECC        : sceau par ligne à chaque étage

Dépendances : numpy + bibliothèque standard. Termux OK.
"""
import time, zlib
from collections import OrderedDict
import numpy as np
import delta18 as D18


class Etage:
    """Un niveau de la hiérarchie : capacité, finesse (bits), contenu."""

    def __init__(self, niveau, capacite, bits):
        self.niveau, self.capacite, self.bits = niveau, capacite, bits
        self.lignes = OrderedDict()          # adresse -> Condensateur (ou tableau brut)
        self.octets = 0

    def poser(self, a, cond, taille):
        self.lignes[a] = cond
        self.octets += taille

    def retirer(self, a, taille):
        self.lignes.pop(a, None)
        self.octets -= taille

    def plein(self):
        return self.octets > self.capacite


class X3D:
    """Hiérarchie Δ : N étages, répertoire global, dégradation en descendant."""

    def __init__(self, source, niveaux=256, capacite_L1=256 * 1024, facteur=1.35,
                 bits_max=16, bits_min=2, P=None):
        self.source = source
        self.P = P or (lambda a: 0.0)
        self.etages = []
        cap = capacite_L1
        for k in range(niveaux):
            # la finesse décroît avec la profondeur : 16 bits en haut, 2 en bas
            # la finesse chute par paliers : plus on descend, plus c'est flou
            b = 16 if k < 4 else (8 if k < 16 else (4 if k < 64 else 2))
            b = max(bits_min, min(bits_max, b))
            self.etages.append(Etage(k, int(cap), b))
            cap *= facteur
        self.repertoire = {}                 # adresse -> niveau  (UNE recherche)
        self.succes = self.echecs = 0
        self.t_succes = self.t_echecs = 0.0
        self.degradations = 0
        self.oubliees = 0
        self.premier_libre = 0
        self.finesse = {}          # adresse -> la finesse la plus BASSE qu'elle ait connue
        self.par_niveau = np.zeros(niveaux, int)

    # ------------------------------------------------------------- interne
    def _ranger(self, a, x, niveau):
        e = self.etages[niveau]
        c = D18.Condensateur(x, bits=e.bits)
        e.poser(a, c, c.octets())
        self.repertoire[a] = niveau
        self.finesse[a] = min(self.finesse.get(a, 64), c.bits)
        self._equilibrer(niveau)

    def _deplacer(self, a, c, de, vers):
        """Déplace une ligne d'un étage à l'autre.

        TROIS RÈGLES, et chacune a été payée par une mesure ratée :
        - en DESCENDANT, on recomprime seulement si l'étage d'arrivée est plus
          grossier que la ligne ne l'est déjà ;
        - en MONTANT, on ne recomprime JAMAIS : remettre en 16 bits une ligne
          tombée à 2 bits n'invente aucune information, cela ne ferait que
          gaspiller huit fois la place en faisant croire à une précision
          disparue. La ligne monte telle quelle : elle gagne en VITESSE, pas
          en finesse. La finesse perdue est perdue, et Δ le dit ;
        - à finesse égale, le condensateur est déplacé TEL QUEL : aucun coût,
          et surtout aucune perte de génération à force de relire-réécrire.
        """
        e, e2 = self.etages[de], self.etages[vers]
        e.retirer(a, c.octets())
        if vers > de and e2.bits < c.bits:
            c2 = D18.Condensateur(c.lire(), bits=e2.bits)
        else:
            c2 = c
        e2.poser(a, c2, c2.octets())
        self.repertoire[a] = vers
        self.finesse[a] = min(self.finesse.get(a, 64), c2.bits)
        if vers < self.premier_libre:
            self.premier_libre = vers
        return c2

    def _premier_libre_sous(self, k):
        """Le premier étage SOUS k qui a encore de la place.

        C'est la correction qui change tout : la ligne froide ne descend plus
        d'un cran en poussant son voisin, qui pousse le sien, sur cent étages.
        Elle va DIRECTEMENT là où il reste de la place. Un seul saut au lieu
        d'une cascade — et le nombre d'étages redevient gratuit."""
        j = max(k + 1, self.premier_libre)
        n = len(self.etages)
        while j < n and self.etages[j].plein():
            j += 1
        return j

    def _equilibrer(self, niveau):
        """Tant que l'étage déborde, sa ligne la plus froide descend, d'un seul
        saut, vers le premier étage qui a de la place."""
        e = self.etages[niveau]
        n = len(self.etages)
        while e.plein():
            bouge = False
            for a in list(e.lignes.keys()):       # de la plus froide à la plus chaude
                if self.P(a) >= 1.0:              # ligne à 100 % : ne descend jamais
                    continue
                c = e.lignes[a]
                j = self._premier_libre_sous(niveau)
                if j < n:
                    self._deplacer(a, c, niveau, j)
                    self.degradations += 1
                    self.premier_libre = j if not self.etages[j].plein() else j + 1
                else:
                    e.retirer(a, c.octets())
                    self.repertoire.pop(a, None)  # plus une place nulle part : on oublie
                    self.oubliees += 1
                bouge = True
                if not e.plein():
                    break
            if not bouge:                          # que des lignes protégées : on s'arrête
                break

    # -------------------------------------------------------------- accès
    def lire(self, a):
        t0 = time.time()
        niv = self.repertoire.get(a)          # UNE SEULE recherche, 256 étages ou 3
        if niv is not None:
            c = self.etages[niv].lignes[a]
            x = c.lire()
            self.succes += 1
            self.par_niveau[niv] += 1
            self.etages[niv].lignes.move_to_end(a)
            if niv > 0:                        # promotion vers le haut, sans ré-encodage
                self._deplacer(a, c, niv, 0)
                self._equilibrer(0)
            self.t_succes += time.time() - t0
            return x
        x = self.source(a)
        self._ranger(a, x, 0)
        self.echecs += 1
        self.t_echecs += time.time() - t0
        return x

    # ------------------------------------------------------------ mesures
    def octets(self):
        return sum(e.octets for e in self.etages)

    def capacite_totale(self):
        return sum(e.capacite for e in self.etages)

    def taux(self):
        return self.succes / max(self.succes + self.echecs, 1)

    def latence_us(self):
        n = self.succes + self.echecs
        return (self.t_succes + self.t_echecs) / max(n, 1) * 1e6

    def occupation(self):
        return [(e.niveau, e.bits, len(e.lignes), e.octets) for e in self.etages if e.lignes]


class CacheClassique:
    """Référence : 3 niveaux, interrogés l'un après l'autre, éviction = perte."""

    def __init__(self, source, caps=(256 * 1024, 2 * 1024 * 1024, 32 * 1024 * 1024)):
        self.source = source
        self.niv = [OrderedDict() for _ in caps]
        self.caps = caps
        self.succes = self.echecs = self.perdues = 0
        self.t = 0.0

    def octets(self, k):
        return sum(v.nbytes for v in self.niv[k].values())

    def lire(self, a):
        t0 = time.time()
        for k, d in enumerate(self.niv):       # recherche étage par étage
            if a in d:
                x = d.pop(a)
                self.niv[0][a] = x
                self.succes += 1
                self.t += time.time() - t0
                return x
        x = self.source(a)
        self.niv[0][a] = x
        for k in range(len(self.niv)):
            while self.octets(k) > self.caps[k] and self.niv[k]:
                b, v = self.niv[k].popitem(last=False)
                if k + 1 < len(self.niv):
                    self.niv[k + 1][b] = v
                else:
                    self.perdues += 1          # au dernier étage : PERDU
        self.echecs += 1
        self.t += time.time() - t0
        return x

    def taux(self):
        return self.succes / max(self.succes + self.echecs, 1)

    def latence_us(self):
        return self.t / max(self.succes + self.echecs, 1) * 1e6
