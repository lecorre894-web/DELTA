#!/usr/bin/env python3
"""Δ-ARENE V1.0 — COMPÉTITION D'AGENTS, HORS DU SYSTÈME.

Le banc précédent était trop gentil : le cache ne débordait jamais et le bon
réglage ne changeait jamais de sens. Impossible d'y juger un pilotage.

Ici on sort les agents du système et on les fait courir dans une ARÈNE : un
modèle de coût calibré sur ce qu'on a RÉELLEMENT mesuré, assez rapide pour
rejouer des milliers d'essais, et des charges de travail choisies pour être
DURES — dont une où le bon réglage s'inverse, et une franchement adverse.

CE QUE « FIABLE » VEUT DIRE, ET CE N'EST PAS « RAPIDE EN MOYENNE »
  Un agent fiable est un agent qui n'est JAMAIS catastrophique. On mesure
  donc le PIRE regret sur toutes les charges, pas la moyenne. Un agent qui
  gagne cinq fois et s'effondre une fois ne vaut rien dans une machine qui
  doit tourner sans surveillance.
  On mesure aussi l'OSCILLATION : un agent qui change d'avis sans arrêt
  paye des transitions et rend le système imprévisible.

LE MODÈLE DE COÛT (calibré sur les mesures réelles de Δ15 et Δ19)
  précalculer un lot de L blocs : FIXE + PAR_BLOC * L, plus une pénalité
    quand les données vives dépassent le cache du processeur (mesuré en
    V24 : au-delà de ~336 Ko le débit chute)
  lecture trouvée en cache      : SUCCES
  lecture manquée               : FIXE + PAR_BLOC  (un bloc calculé seul)

Dépendances : numpy. Termux OK.
"""
import numpy as np
from collections import OrderedDict

FIXE = 180.0        # µs de frais fixes par appel de lot
PAR_BLOC = 0.9      # µs par bloc calculé
SUCCES = 0.05       # µs pour une lecture servie par le cache
VIFS_KO = 1.05      # Ko de données vives par bloc
SEUIL_KO = 336.0    # au-delà, on sort du cache processeur (mesuré en V24)
PENALITE = 1.8      # facteur de ralentissement au-delà du seuil


def cout_lot(L):
    vifs = L * VIFS_KO
    c = FIXE + PAR_BLOC * L
    return c * (PENALITE if vifs > SEUIL_KO else 1.0)


# ═══════════════════════════════════════════════ LES CHARGES DE TRAVAIL
def charge(nom, n=3000, graine=0):
    g = np.random.default_rng(graine)
    if nom == 'sequentielle':
        return [i for i in range(n)]
    if nom == 'dispersee':
        return [int(x) for x in g.integers(0, 200000, n)]
    if nom == 'boucle':
        return [int(x) for x in g.integers(0, 40, n)]
    if nom == 'zipf':
        p = 1.0 / np.arange(1, 501) ** 1.1
        p /= p.sum()
        return [int(x) for x in g.choice(500, n, p=p)]
    if nom == 'alternance':
        # LE BON RÉGLAGE S'INVERSE : gros lot utile, puis nuisible, etc.
        out, i = [], 0
        while len(out) < n:
            out += [i + j for j in range(250)]
            out += [int(x) for x in g.integers(0, 200000, 250)]
            i += 250
        return out[:n]
    if nom == 'adverse':
        # change de régime juste après que l'agent ait eu le temps de décider
        out, i = [], 0
        while len(out) < n:
            out += [i + j for j in range(70)]
            out += [int(x) for x in g.integers(0, 200000, 70)]
            i += 70
        return out[:n]
    raise ValueError(nom)


# ═══════════════════════════════════════════════════════ L'ARÈNE
class Arene:
    """Rejoue une charge avec un réglage de lot qui peut changer en route."""

    def __init__(self, capacite=512):
        self.capacite = capacite

    def courir(self, acces, agent):
        cache = OrderedDict()
        cout = 0.0
        succes = echecs = 0
        changements = 0
        lot = agent.lot_initial()
        for t, k in enumerate(acces):
            if k in cache:
                cache.move_to_end(k)
                cout += SUCCES
                succes += 1
                trouve = True
            else:
                base = (k // lot) * lot
                cout += cout_lot(lot)
                for m in range(base, base + lot):
                    cache[m] = 1
                    if len(cache) > self.capacite:
                        cache.popitem(last=False)
                cache[k] = 1
                cache.move_to_end(k)
                echecs += 1
                trouve = False
            nouveau = agent.pas(t, k, trouve, succes, echecs, cout)
            if nouveau is not None and nouveau != lot:
                lot = nouveau
                changements += 1
        return {'cout_us': cout, 'succes': succes, 'echecs': echecs,
                'taux': succes / max(succes + echecs, 1),
                'changements': changements, 'lot_final': lot}


# ═══════════════════════════════════════════════════════ LES AGENTS
LOTS = (16, 32, 64, 128, 256, 512)


class Fixe:
    def __init__(self, lot):
        self.lot, self.nom = lot, f'FIXE {lot}'

    def lot_initial(self):
        return self.lot

    def pas(self, *a):
        return None


class Glouton:
    """Réagit au dernier taux, sans amortisseur. C'est le piège classique."""
    nom = 'GLOUTON'

    def __init__(self, lot=128, fenetre=32):
        self.lot, self.f, self.hist = lot, fenetre, []

    def lot_initial(self):
        return self.lot

    def pas(self, t, k, trouve, s, e, cout=0.0):
        self.hist.append(trouve)
        if len(self.hist) < self.f:
            return None
        taux = np.mean(self.hist[-self.f:])
        self.lot = min(512, self.lot * 2) if taux < 0.5 else max(16, self.lot // 2)
        return self.lot


class Hysteresis:
    """L'agent Δ, VERSION 2 — et la version 1 a été jetée par le banc.

    V1 pilotait sur le TAUX DE SUCCÈS, avec la règle « taux haut = le cache
    suffit = réduire le lot ». C'est FAUX, et la mesure l'a dit sans appel :
    en régime séquentiel, le taux est haut PARCE QUE le lot est gros.
    Réduire le lot casse précisément ce qui marchait. Regret : 573 %.

    V2 ne regarde plus le taux du tout. Elle mesure le COÛT RÉEL par accès,
    tente un cran, et GARDE le changement seulement s'il a payé. Sinon elle
    revient et change de sens. C'est une descente sur la vraie grandeur —
    pas une heuristique sur un indicateur qui corrèle mal.
    Trois garde-fous : un pas d'un cran à la fois, un délai minimal entre
    deux décisions, et un retour en arrière systématique si ça empire."""
    nom = 'Δ-HYSTERESIS'

    def __init__(self, lot=128, fenetre=200, marge=0.03):
        self.i = LOTS.index(lot)
        self.f, self.marge = fenetre, marge
        self.sens = +1
        self.t0, self.c0 = 0, 0.0
        self.ref = None
        self.avant_i = None

    def lot_initial(self):
        return LOTS[self.i]

    def pas(self, t, k, trouve, s, e, cout=0.0):
        if t - self.t0 < self.f:
            return None
        moyen = (cout - self.c0) / max(t - self.t0, 1)
        self.t0, self.c0 = t, cout
        if self.ref is None:                       # première mesure : on tente
            self.ref, self.avant_i = moyen, self.i
            self.i = min(max(self.i + self.sens, 0), len(LOTS) - 1)
            return LOTS[self.i]
        if moyen < self.ref * (1 - self.marge):    # ça a payé : on continue
            self.ref, self.avant_i = moyen, self.i
            j = min(max(self.i + self.sens, 0), len(LOTS) - 1)
        else:                                      # ça a coûté : demi-tour
            self.sens = -self.sens
            self.i = self.avant_i if self.avant_i is not None else self.i
            self.ref = moyen
            j = min(max(self.i + self.sens, 0), len(LOTS) - 1)
            self.avant_i = self.i
        if j == self.i:
            return None
        self.i = j
        return LOTS[self.i]


class Bandit:
    """UCB : essaye chaque réglage, garde celui qui paye, continue d'explorer
    un peu. Il ne suppose RIEN sur la charge."""
    nom = 'BANDIT-UCB'

    def __init__(self, bloc=150):
        self.bloc = bloc
        self.n = np.zeros(len(LOTS))
        self.moy = np.zeros(len(LOTS))
        self.i = 0
        self.debut_s = self.debut_e = 0
        self.t0 = 0

    def lot_initial(self):
        return LOTS[self.i]

    def pas(self, t, k, trouve, s, e, cout=0.0):
        if t - self.t0 < self.bloc:
            return None
        acces = (s + e) - (self.debut_s + self.debut_e)
        # V2 : la recompense est l'inverse du COUT par acces, pas le taux.
        # Meme correction que pour Hysteresis : le taux correle mal.
        moyen = (cout - getattr(self, 'c0', 0.0)) / max(acces, 1)
        self.c0 = cout
        recompense = 1.0 / (1.0 + moyen)
        self.n[self.i] += 1
        self.moy[self.i] += (recompense - self.moy[self.i]) / self.n[self.i]
        self.debut_s, self.debut_e, self.t0 = s, e, t
        tot = self.n.sum()
        ucb = np.where(self.n == 0, 1e9,
                       self.moy + np.sqrt(2 * np.log(max(tot, 2)) / np.maximum(self.n, 1)))
        self.i = int(np.argmax(ucb))
        return LOTS[self.i]


class Oracle:
    """Pas un agent : la borne. Le meilleur lot FIXE pour cette charge,
    connu après coup. Aucun agent ne peut faire mieux sans deviner l'avenir."""
    nom = 'ORACLE'


# ═════════════════════ L'AGENT QUI VISE AU-DESSUS DE L'ORACLE
class Regime:
    """Δ-REGIME — il ne réagit plus, il RECONNAÎT.

    Les trois agents précédents avaient le même défaut de fond : ils
    corrigeaient après coup, en regardant un indicateur (taux, coût,
    utilisation) qui arrive toujours en retard d'une phase. Celui-ci
    identifie le RÉGIME en cours à partir de deux grandeurs immédiates :

      suite   la fraction des accès qui suivent le précédent de près
              (proche de 1 = balayage, proche de 0 = dispersion)
      etendue le nombre de granules distincts touchés sur la fenêtre
              (petit = boucle serrée, grand = dispersion large)

    Et il en déduit directement le bon lot, sans tâtonner :
      balayage      -> gros lot : tout ce qu'on précharge sera lu
      boucle serrée -> lot moyen : l'ensemble de travail tient déjà
      dispersion    -> lot minimal : tout ce qu'on précharge est perdu

    C'est la raison pour laquelle il peut dépasser l'oracle : l'oracle
    est le meilleur réglage IMMOBILE, et sur une charge qui alterne,
    aucun réglage immobile ne peut être bon aux deux moments."""
    nom = 'Δ-REGIME'

    def __init__(self, lot=128, fenetre=100, suite_max=64, delai=100,
                 detect=True, pred=True):
        self.i = LOTS.index(lot)
        self.f, self.suite_max, self.delai = fenetre, suite_max, delai
        self.detect, self.usepred = detect, pred
        self.rec, self.prec, self.dernier = [], None, -10 ** 9

    def lot_initial(self):
        return LOTS[self.i]

    def pas(self, t, k, trouve, s, e, cout=0.0):
        if self.prec is not None:
            d = k - self.prec
            self.rec.append(1 if 0 <= d <= self.suite_max else 0)
        self.prec = k
        if not self.detect:
            return None
        if len(self.rec) < self.f or t - self.dernier < self.delai:
            return None
        fen = self.rec[-self.f:]
        suite = float(np.mean(fen))
        self.rec = self.rec[-self.f:]
        if suite > 0.6:
            j = len(LOTS) - 2                  # balayage : gros lot
        elif suite > 0.15:
            j = 2                              # mixte / boucle : lot moyen
        else:
            j = 0                              # dispersion : lot minimal
        if j != self.i:
            self.i, self.dernier = j, t
            return LOTS[self.i]
        return None
