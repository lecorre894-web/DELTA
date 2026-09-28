#!/usr/bin/env python3
"""Δ-FRACTAL V1.0 — CHAQUE GRAIN PORTE LE CONCEPT ENTIER.

L'IDÉE DE RENÉ, MOT POUR MOT : « sur chaque micro-qubit tu octroies ce calcul
dans son intégrité ; le concept complet fera l'infini ».

CE QUE ÇA VEUT DIRE, EN CLAIR
  Jusqu'ici, une grappe était une DONNÉE : un bloc de nombres, calculé par le
  fond partagé. Ici elle devient un ÊTRE COMPLET : elle calcule sa part, elle
  connaît son erreur, elle porte son sceau, et elle sait prouver qu'elle est
  bien elle. Et — c'est là que l'infini entre — un paquet de ces êtres EST
  lui-même un être de la même espèce. Même interface, même sceau, même
  preuve. Donc on peut recommencer. Indéfiniment.

              cellule → nœud → nœud de nœuds → ... et ainsi de suite
              (chacun offrant exactement les mêmes quatre gestes)

LES QUATRE GESTES, IDENTIQUES À CHAQUE ÉTAGE
  valeur()   ce que je vaux              sceau()     ma signature
  preuve(i)  la preuve qu'un des miens   verifier()  suis-je intact ?
             m'appartient vraiment

CE QUE LA RÉCURSION DONNE VRAIMENT
  Une preuve d'appartenance de log2(N) empreintes. Pour dix-huit milliards de
  milliards d'unités : environ un kilo-octet. C'est ça, l'infini utile — pas
  une capacité de rangement, une capacité de PREUVE.

CE QU'ELLE COÛTE, ET IL FAUT LE MESURER AVANT DE S'EXTASIER
  Si le grain est trop fin, le concept complet pèse plus lourd que ce qu'il
  transporte : un sceau de 16 octets sur une cellule de 8 octets, c'est 200 %
  de surcoût. Il existe donc une TAILLE DE GRAIN où l'infini devient gratuit,
  et en dessous de laquelle il se paye très cher. Ce module la mesure.

Dépendances : numpy + hashlib. Termux OK.
"""
import hashlib, math, time
import numpy as np


def sceller(cle, donnees):
    return hashlib.blake2b(donnees, key=cle, digest_size=16).digest()


class Etre:
    """L'interface unique. Une cellule et un nœud de dix millions de cellules
    répondent exactement aux mêmes quatre gestes : c'est ce qui rend la
    récursion possible, et donc l'infini."""

    def valeur(self):
        raise NotImplementedError

    def sceau(self):
        return self._sceau

    def poids(self):
        """Nombre de cellules élémentaires sous moi."""
        raise NotImplementedError

    def octets(self):
        raise NotImplementedError


class Cellule(Etre):
    """Le grain. Il calcule SA part, et il porte sa propre preuve."""

    __slots__ = ('k', 'v', '_sceau')

    def __init__(self, k, fournisseur, cle):
        self.k = k
        self.v = fournisseur(k)                      # le calcul, en propre
        self._sceau = sceller(cle, np.ascontiguousarray(self.v).tobytes())

    def valeur(self):
        return self.v

    def poids(self):
        return 1

    def octets(self):
        return self.v.nbytes + len(self._sceau)

    def utile(self):
        return self.v.nbytes

    def recalculer(self, cle):
        """RELIT la donnée et refait le sceau. C'est le seul geste qui protège
        vraiment : un sceau qu'on se contente de relire dans un coin ne
        protège rien du tout — il confirme seulement qu'on l'a bien rangé."""
        return sceller(cle, np.ascontiguousarray(self.v).tobytes())


class Noeud(Etre):
    """Un paquet d'êtres — et lui-même un être. C'est toute l'astuce.

    Son sceau est bâti sur ceux de ses enfants, par un arbre de Merkle
    binaire interne, avec le garde-fou trouvé hier sur Bitcoin : deux enfants
    appariés identiques = arbre muté = refus."""

    __slots__ = ('enfants', 'niveaux', '_sceau', 'cle', '_poids')

    def __init__(self, enfants, cle, strict=True):
        assert enfants, 'un nœud sans enfant n\'est pas un être'
        self.enfants, self.cle = enfants, cle
        self._poids = sum(e.poids() for e in enfants)
        cour = [e.sceau() for e in enfants]
        self.niveaux = [list(cour)]
        while len(cour) > 1:
            reels = len(cour)
            if len(cour) % 2:
                cour = cour + [cour[-1]]
            haut = []
            for i in range(0, len(cour), 2):
                if strict and i + 1 < reels and cour[i] == cour[i + 1]:
                    raise ValueError('deux enfants appariés identiques : '
                                     'arbre muté (CVE-2012-2459)')
                haut.append(sceller(cle, cour[i] + cour[i + 1]))
            self.niveaux.append(haut)
            cour = haut
        self._sceau = cour[0]

    def valeur(self):
        return np.concatenate([np.atleast_1d(e.valeur()) for e in self.enfants])

    def poids(self):
        return self._poids

    def octets(self):
        return sum(e.octets() for e in self.enfants) + len(self._sceau) \
            + sum(len(n) * 16 for n in self.niveaux)

    def utile(self):
        return sum(e.utile() for e in self.enfants)

    def recalculer(self, cle):
        """Refait le sceau DEPUIS LE BAS : chaque enfant relit sa donnée, et
        le nœud remonte l'arbre avec ces sceaux frais. Coût : une relecture
        complète, O(N) — il n'y a pas de miracle, vérifier c'est relire.
        Ce qui est gratuit, en revanche, c'est de savoir OÙ ça cloche."""
        cour = [e.recalculer(cle) for e in self.enfants]
        while len(cour) > 1:
            if len(cour) % 2:
                cour = cour + [cour[-1]]
            cour = [sceller(cle, cour[i] + cour[i + 1])
                    for i in range(0, len(cour), 2)]
        return cour[0]

    # ------------------------------------------------------------- preuve
    def preuve_locale(self, i):
        """Chemin d'authentification de l'enfant i, dans CE nœud seulement."""
        p, idx = [], i
        for niv in self.niveaux[:-1]:
            j = idx ^ 1
            voisin = niv[j] if j < len(niv) else niv[idx]
            p.append((voisin, idx % 2 == 1))
            idx //= 2
        return p

    @staticmethod
    def verifier_chemin(sceau_feuille, chemin, racine, cle):
        h = sceau_feuille
        for voisin, a_gauche in chemin:
            h = sceller(cle, (voisin + h) if a_gauche else (h + voisin))
        return h == racine


class Fractal:
    """L'arbre entier : des cellules, groupées, regroupées, jusqu'à la racine.
    Chaque étage est fait de la même matière que celui du dessous."""

    def __init__(self, n_cellules, fournisseur, cle, branchement=2, strict=True):
        t0 = time.time()
        self.cle, self.branchement = cle, branchement
        self.cellules = [Cellule(k, fournisseur, cle) for k in range(n_cellules)]
        self.etages = [self.cellules]
        cour = self.cellules
        while len(cour) > 1:
            paquets = [cour[i:i + branchement] for i in range(0, len(cour), branchement)]
            cour = [Noeud(p, cle, strict) if len(p) > 1 else p[0] for p in paquets]
            self.etages.append(cour)
        self.racine = cour[0]
        self.t_construction = time.time() - t0

    def profondeur(self):
        return len(self.etages)

    def octets(self):
        return self.racine.octets() if isinstance(self.racine, Noeud) \
            else self.racine.octets()

    def utile(self):
        return sum(c.utile() for c in self.cellules)

    def surcout(self):
        u = self.utile()
        return (self.octets() - u) / max(u, 1)

    # ------------------------------------- la preuve qui traverse TOUS les étages
    def preuve(self, i):
        """Preuve d'appartenance de la cellule i, de la feuille à la racine,
        en traversant chaque étage. C'est la récursion mise à plat."""
        chemin, idx = [], i
        for e in range(len(self.etages) - 1):
            cour = self.etages[e]
            parent_idx = idx // self.branchement
            parent = self.etages[e + 1][parent_idx]
            if isinstance(parent, Noeud):
                chemin += parent.preuve_locale(idx % self.branchement)
            idx = parent_idx
        return chemin

    def verifier(self, i, chemin=None):
        """La preuve part du sceau RECALCULÉ sur la donnée, jamais du sceau
        rangé à côté. Sans cela, altérer la donnée sans toucher au sceau
        passerait inaperçu — c'était le cas, et c'était faux."""
        ch = chemin if chemin is not None else self.preuve(i)
        return Noeud.verifier_chemin(self.cellules[i].recalculer(self.cle), ch,
                                     self.racine.sceau(), self.cle)

    def intact(self):
        return self.racine.recalculer(self.cle) == self.racine.sceau()

    def taille_preuve(self, i):
        return len(self.preuve(i)) * 16

    # ---------------------------------------------- trouver le grain fautif
    def designer_fautif(self):
        """Descend de la racine vers la feuille en ne suivant que la branche
        dont le sceau recalculé ne colle plus.

        CE QUI EST HONNÊTE À DIRE SUR LE COÛT : détecter qu'il y a un
        problème exige de relire toute la donnée — O(N), et aucune structure
        n'y échappe. Ce que la fractale offre en plus est gratuit : au bout
        de cette même lecture, elle ne dit pas seulement « quelque chose ne
        va pas », elle dit LEQUEL. Un sceau global unique, lui, coûte
        exactement autant et ne donne que l'alarme."""
        if self.racine.recalculer(self.cle) == self.racine.sceau():
            return None, []
        etre, chemin = self.racine, []
        while isinstance(etre, Noeud):
            trouve = None
            for j, enf in enumerate(etre.enfants):
                if enf.recalculer(self.cle) != enf.sceau():
                    trouve = j
                    break
            if trouve is None:                 # l'ossature elle-même est touchée
                return etre, chemin
            chemin.append(trouve)
            etre = etre.enfants[trouve]
        return etre, chemin


def fournisseur_grain(b):
    """Le calcul propre à chaque grain : sa petite part du problème global."""
    def f(k):
        i = np.arange(k * b, (k + 1) * b, dtype=np.float64)
        return np.sin(0.001 * i) + 0.5 * np.cos(0.0007 * i + 1.0)
    return f
