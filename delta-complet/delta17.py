#!/usr/bin/env python3
"""Δ17 V1.7 — Δ-CACHE : cache prédictif, priorité par %, réutilisation, ECC.

Le principe Δ appliqué au cache :

  grappes identiques -> lignes DÉDUPLIQUÉES (même contenu = une seule copie)
  grappe allumée     -> ligne unique, gardée seule
  X à la demande     -> la ligne est stockée compressée à son rang utile
  % (P)              -> PRIORITÉ de la ligne : une ligne à 100 % ne s'évince pas
  cône de lumière    -> PRÉDICTION : on précharge le futur probable, pas tout
  ECC                -> empreinte par ligne, corruption détectée
  réutilisation      -> mesurée (taux de succès, économie de latence réelle)

PRÉDICTEURS (les deux, fusionnés) :
  - PAS CONSTANT (stride) : si les deux derniers écarts sont égaux, le suivant
    l'est probablement aussi. C'est le cas des parcours réguliers.
  - SUCCESSEUR (Markov d'ordre 1) : on retient qui a suivi qui. C'est le cas
    des parcours irréguliers MAIS répétitifs (graphes, boucles imbriquées).
  Le pas prime quand il est confirmé, sinon le successeur prend la main.

Le préchargement travaille dans un FIL DE FOND : il occupe le temps pendant
lequel le calcul principal, lui, travaille (LOAD || COMPUTE de Δ16).

Dépendances : numpy + bibliothèque standard. Termux OK.
"""
import threading, queue, time, zlib
import numpy as np


class Ligne:
    __slots__ = ('data', 'P', 'crc', 'usages', 'dernier')

    def __init__(self, data, P=0.0):
        self.data = data
        self.P = P
        self.usages = 1
        self.dernier = time.time()
        self.crc = zlib.crc32(np.ascontiguousarray(data).view(np.uint8).tobytes()) & 0xFFFFFFFF

    def valide(self):
        return (zlib.crc32(np.ascontiguousarray(self.data).view(np.uint8).tobytes())
                & 0xFFFFFFFF) == self.crc

    def octets(self):
        return self.data.nbytes


class Predicteur:
    """Pas constant + successeur. Renvoie les prochaines adresses probables."""

    def __init__(self, profondeur=4):
        self.prof = profondeur
        self.dernier = None
        self.pas = None
        self.pas_confirme = 0
        self.successeur = {}
        self.justes = self.tentees = 0

    def observer(self, a):
        if self.dernier is not None:
            d = a - self.dernier
            if self.pas is not None and d == self.pas:
                self.pas_confirme += 1
            else:
                self.pas, self.pas_confirme = d, 0
            self.successeur[self.dernier] = a
        self.dernier = a

    def prevoir(self):
        if self.dernier is None:
            return []
        if self.pas is not None and self.pas_confirme >= 1 and self.pas != 0:
            return [self.dernier + self.pas * (k + 1) for k in range(self.prof)]
        out, cour = [], self.dernier
        for _ in range(self.prof):                   # chaîne de successeurs
            s = self.successeur.get(cour)
            if s is None or s in out:
                break
            out.append(s); cour = s
        return out


class DeltaCache:
    def __init__(self, source, capacite_octets, P=None, prefetch=True, profondeur=4, fils=4):
        self.source = source                  # adresse -> tableau (mémoire lente)
        self.cap = capacite_octets
        self.P = P or (lambda a: 0.0)         # priorité (%) d'une adresse
        self.lignes = {}
        self.contenu = {}                     # empreinte -> adresse maîtresse (dédup)
        from collections import OrderedDict
        self.ordre = OrderedDict()            # adresses évinçables, en ordre d'usage
        self.proteges = set()                 # adresses à 100 % : jamais évincées
        self._octets = 0                      # comptage INCRÉMENTAL (plus de O(n) par accès)
        self._refs = {}                       # id(ligne) -> nombre d'adresses qui la pointent
        self.pred = Predicteur(profondeur)
        self.prefetch = prefetch
        self.file = queue.Queue(maxsize=64)
        self.stop = False
        self.succes = self.echecs = self.precharges = self.precharges_utiles = 0
        self.t_succes = self.t_echecs = 0.0
        self.octets_source = 0
        self.en_vol = {}                      # adresse -> événement (préchargement en cours)
        self.verrou = threading.Lock()
        self.attendus = 0                     # accès servis par un préchargement EN COURS
        if prefetch:
            # PLUSIEURS FILS : une attente d'entrée-sortie ne consomme pas de
            # processeur, on peut donc en avoir plusieurs en vol et prendre
            # réellement de l'avance.
            self.fils = [threading.Thread(target=self._boucle, daemon=True) for _ in range(fils)]
            for f in self.fils:
                f.start()

    # ------------------------------------------------------------ interne
    def _boucle(self):
        while not self.stop:
            try:
                a = self.file.get(timeout=0.05)
            except queue.Empty:
                continue
            if a in self.lignes:
                continue
            with self.verrou:
                if a in self.en_vol:
                    continue
                ev = threading.Event()
                self.en_vol[a] = ev
            try:
                d = self.source(a)
                self._poser(a, d, precharge=True)
            except Exception:
                pass
            finally:
                with self.verrou:
                    self.en_vol.pop(a, None)
                ev.set()

    def _poser(self, a, d, precharge=False):
        e = zlib.crc32(np.ascontiguousarray(d).view(np.uint8).tobytes()) & 0xFFFFFFFF
        maitre = self.contenu.get(e)
        if maitre is not None and maitre in self.lignes:   # DÉDUPLICATION
            l = self.lignes[maitre]
        else:
            l = Ligne(d, self.P(a))
            self.contenu[e] = a
        self.lignes[a] = l
        k = id(l)
        if k not in self._refs:
            self._refs[k] = 0
            self._octets += l.octets()
        self._refs[k] += 1
        if l.P >= 1.0:
            self.proteges.add(a)
        else:
            self.ordre[a] = True
        if precharge:
            self.precharges += 1
        self._evincer()

    def _evincer(self):
        """Éviction en O(1) : la plus ancienne des lignes NON protégées part.
        Les lignes à 100 % ne sont jamais évincées (facteur % de René)."""
        while self._octets > self.cap and self.ordre:
            a, _ = self.ordre.popitem(last=False)
            l = self.lignes.pop(a, None)
            if l is None:
                continue
            k = id(l)
            self._refs[k] -= 1
            if self._refs[k] <= 0:
                self._octets -= l.octets()
                del self._refs[k]

    def octets(self):
        return self._octets

    # -------------------------------------------------------------- accès
    def lire(self, a):
        t0 = time.time()
        l = self.lignes.get(a)
        if l is not None:
            l.usages += 1; l.dernier = time.time()
            if a in self.ordre:
                self.ordre.move_to_end(a)
            self.succes += 1
            self.t_succes += time.time() - t0
            d = l.data
        else:
            with self.verrou:
                ev = self.en_vol.get(a)
            if ev is not None:                # DÉJÀ EN VOL : on attend, on ne refait pas
                ev.wait(timeout=5.0)
                l = self.lignes.get(a)
                if l is not None:
                    self.attendus += 1
                    self.succes += 1
                    self.t_succes += time.time() - t0
                    self.pred.observer(a)
                    return l.data
            self.echecs += 1
            d = self.source(a)
            self.octets_source += d.nbytes
            self._poser(a, d)
            self.t_echecs += time.time() - t0
        self.pred.observer(a)
        if self.prefetch:
            for x in self.pred.prevoir():
                if x not in self.lignes and not self.file.full():
                    self.file.put(x)
        return d

    def fermer(self):
        self.stop = True

    # ------------------------------------------------------------ mesures
    def taux(self):
        n = self.succes + self.echecs
        return self.succes / max(n, 1)

    def latence_succes_us(self):
        return self.t_succes / max(self.succes, 1) * 1e6

    def latence_echec_us(self):
        return self.t_echecs / max(self.echecs, 1) * 1e6

    def latence_moyenne_us(self):
        n = self.succes + self.echecs
        return (self.t_succes + self.t_echecs) / max(n, 1) * 1e6

    def valide(self):
        return all(l.valide() for l in self.lignes.values())


class CacheLRU:
    """Référence : cache classique, sans prédiction ni priorité."""
    def __init__(self, source, capacite_octets):
        self.source, self.cap = source, capacite_octets
        self.lignes = {}
        self.succes = self.echecs = 0
        self.t_succes = self.t_echecs = 0.0

    def octets(self):
        return sum(d.nbytes for d in self.lignes.values())

    def lire(self, a):
        t0 = time.time()
        if a in self.lignes:
            d = self.lignes.pop(a); self.lignes[a] = d
            self.succes += 1; self.t_succes += time.time() - t0
            return d
        d = self.source(a)
        self.lignes[a] = d
        while self.octets() > self.cap and self.lignes:
            self.lignes.pop(next(iter(self.lignes)))
        self.echecs += 1; self.t_echecs += time.time() - t0
        return d

    def taux(self):
        return self.succes / max(self.succes + self.echecs, 1)

    def latence_moyenne_us(self):
        n = self.succes + self.echecs
        return (self.t_succes + self.t_echecs) / max(n, 1) * 1e6
