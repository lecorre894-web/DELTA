#!/usr/bin/env python3
"""Δ-SYSTEME V1.0 — LE SYSTÈME ASSEMBLÉ, ET UN AGENT DEDANS.

Δ-UN avait fondu les briques de CALCUL. Ici on monte le système complet et
on y place un AGENT qui le pilote — avec la règle qu'on s'est donnée à 17 h :
un agent qui ne rapporte pas plus qu'il ne coûte doit dégager, et on le
mesure au lieu de l'espérer.

LE SYSTÈME
    CALCUL   fond infini + grappes + lots          (Δ5, Δ14, Δ15)
    CACHE    hiérarchie X3D, éviction = dégradation (Δ17, Δ19)
    MÉMOIRE  pages, condensateurs                   (Δ16, Δ18)
    SCEAU    clé, Merkle, parité sur le chiffré     (Δ9, Δ18)
    MESURE   diversité de Hill, sans seuil arbitraire (Δ-DIVERSITE)

L'AGENT, ET SES QUATRE ÉTAGES
  Étage 0  des compteurs et des seuils. Coût quasi nul, toujours actif.
  Étage 1  le prédicteur de Δ17 : pas constant + successeur.
  Étage 2  la diversité de Leinster : combien de contenus DISTINCTS
           circulent vraiment ? C'est elle qui fixe la finesse, au lieu
           d'un epsilon choisi à la main.
  Étage 3  la décision lente : taille de lot, capacité, priorités.
           Réveillée rarement, sur événement.

  Chaque étage ne se réveille que si celui du dessous n'a pas tranché.
  C'est le cône de lumière de Δ11 appliqué à la DÉCISION.

CE QU'ON COMPARE, ET C'EST LE SEUL JUGE QUI COMPTE
    FIXE    un réglage par défaut, immobile
    AGENT   le pilotage adaptatif
    ORACLE  le meilleur réglage connu APRÈS COUP, borne supérieure
  Si AGENT n'est pas entre les deux, il ne sert à rien. Mesuré, pas supposé.

Dépendances : numpy. Termux OK.
"""
import hashlib, time
import numpy as np
import delta15 as D15
import delta19 as D19
import delta18 as D18
import delta_diversite as DIV


# ═══════════════════════════════════════════════════════ LE SYSTÈME
class Systeme:
    def __init__(self, noyau=None, b=128, eps=1e-8, portee=40,
                 lot=8, bits=16, capacite_L1=256 * 1024, cle=b'Delta-Rene'):
        self.b, self.eps = b, eps
        self.cle = hashlib.blake2b(cle, digest_size=32).digest()
        self.calcul = D15.Omni(noyau or D15.noyau_lisse, b=b, eps=eps,
                               portee_max=portee)
        self.fournisseur = D15.fournisseur(b)
        self.cache = D19.X3D(self._bloc_neuf, niveaux=64,
                             capacite_L1=capacite_L1, facteur=1.25)
        self.lot, self.bits = lot, bits
        self.journal = []                 # (instant, evenement, valeur)
        self.n_calculs = self.n_lectures = 0
        self.t_calcul = self.t_lecture = 0.0

    def _bloc_neuf(self, k):
        t0 = time.perf_counter()
        Y, _ = self.calcul.lot(k, 1, self.fournisseur)
        self.t_calcul += time.perf_counter() - t0
        self.n_calculs += 1
        return Y[:, 0]

    def precalculer(self, k0, nk):
        """Un gros lot, rangé dans le cache : c'est le point de fusion."""
        t0 = time.perf_counter()
        Y, _ = self.calcul.lot(k0, nk, self.fournisseur)
        for m in range(nk):
            self.cache._ranger(k0 + m, Y[:, m].copy(), 0)
        dt = time.perf_counter() - t0
        self.t_calcul += dt
        return Y, dt

    def lire(self, k):
        t0 = time.perf_counter()
        v = self.cache.lire(k)
        self.t_lecture += time.perf_counter() - t0
        self.n_lectures += 1
        return v

    def exact(self, k0, nk):
        """La référence dense, sans compression. Δ doit toujours pouvoir
        être confronté à ce qu'il prétend approcher."""
        p, i = self.calcul.portee, np.arange(self.b)
        cols = np.stack([self.fournisseur(k)
                         for k in range(k0 - p, k0 + nk + p)], 1)
        Y = np.zeros((self.b, nk))
        for c in range(-p, p + 1):
            M = self.calcul.noyau(i[:, None] - (i[None, :] + c * self.b))
            Y += M @ cols[:, p + c: p + c + nk]
        return Y

    def etat(self):
        c = self.cache
        return {'taux': c.taux(), 'succes': c.succes, 'echecs': c.echecs,
                'octets_cache': c.octets(), 'degradations': c.degradations,
                'oubliees': c.oubliees, 'calculs': self.n_calculs,
                'lectures': self.n_lectures, 'lot': self.lot, 'bits': self.bits}


# ═══════════════════════════════════════════════════════ L'AGENT
class Agent:
    """Quatre étages, dormant par défaut, et un budget qu'il doit respecter.

    Il ne tourne PAS en permanence : il se réveille tous les `periode`
    accès, regarde ce qui s'est passé, et ne descend à l'étage suivant que
    si celui du dessous n'a pas su trancher."""

    def __init__(self, sys, periode=64, actif=False):
        """actif=False PAR DÉFAUT, et c'est une décision prise sur mesure.
        Le balayage complet (bench_reglage) montre que la granularité 8
        domine ce système : 0,84 s contre 5,03 s pour G=64 et 7,75 s pour
        G=128 sur la charge mixte. L'agent, lui, finit à 5,45 s.
        Tant que le grain fin gagne partout où ça COÛTE quelque chose, un
        pilotage n'a rien à apporter — il ne peut que s'éloigner du bon
        réglage. On l'allume quand le système change : cache qui déborde
        vraiment, coût de calcul variable, plusieurs machines."""
        self.sys, self.periode, self.actif = sys, periode, actif
        self.pred = __import__('delta17').Predicteur(4)
        self.n = 0
        self.reveils = self.decisions = 0
        self.t_agent = 0.0
        self.trace = []
        self.recents = []
        self.dernier_taux = None
        self.precalcules, self.lus = 0, set()   # travail fait / travail utile
        self.derniere, self.delai = -10 ** 9, 192

    # -------------------------------------------------- étage 0 : compteurs
    @property
    def lus_distincts(self):
        return len(self.lus)

    def observer(self, k, valeur=None, precalcules=0):
        self.n += 1
        self.precalcules += precalcules
        self.lus.add(k)
        self.pred.observer(k)
        if valeur is not None and len(self.recents) < 48:
            self.recents.append(valeur)
        if self.actif and self.n % self.periode == 0:
            self._reveiller()

    # ------------------------------------------- étages 1 à 3 : la décision
    def _reveiller(self):
        t0 = time.perf_counter()
        self.reveils += 1
        s, c = self.sys, self.sys.cache

        # étage 1 — préchargement : le prédicteur a-t-il une idée ?
        for x in self.pred.prevoir()[:2]:
            if x not in c.repertoire and x > 0:
                s.lire(x)

        taux = c.taux()
        # étage 2 — la diversité décide de la finesse, sans epsilon arbitraire
        if len(self.recents) >= 16:
            V = np.array(self.recents[-32:])
            Z = DIV.similarite_cos(V, raideur=8.0)
            d1 = DIV.hill_Z(np.ones(len(V)), Z, 1.0)
            frac = d1 / len(V)
            bits = 16 if frac > 0.6 else (8 if frac > 0.25 else 4)
            if bits != s.bits:
                self.trace.append((self.n, 'bits', s.bits, bits,
                                   f'diversité {d1:.1f}/{len(V)}'))
                s.bits = bits
                self.decisions += 1
            self.recents = self.recents[-8:]

        # étage 3 — décision lente : le signal est l'UTILISATION, et il a
        # fallu deux échecs mesurés pour arriver là.
        #   V1 suivait le TAUX DE SUCCÈS -> démoli par l'arène (regret 573 %) :
        #     en séquentiel le taux est haut PARCE QUE le lot est gros.
        #   V2 suivait le COÛT PAR ACCÈS -> démoli par le système (5,78 s
        #     contre 2,97 s) : quand la charge change de régime, le coût
        #     explose et l'agent attribue l'explosion à SA dernière décision.
        #     Il fait demi-tour à tort, et oscille.
        #   V3 mesure ce qu'on cherche vraiment à éviter : le travail fait
        #     POUR RIEN. Combien de blocs a-t-on précalculés, combien en
        #     a-t-on réellement lu ? Ce rapport ne dépend presque pas du
        #     régime — il dépend du réglage, ce qui est exactement la
        #     condition pour qu'une boucle de décision soit honnête.
        util = self.lus_distincts / max(self.precalcules, 1)
        if self.n - self.derniere < self.delai:      # amortisseur : on laisse
            self.precalcules, self.lus = 0, set()    # au réglage le temps de
            self.t_agent += time.perf_counter() - t0 # produire son effet
            return
        # PALIERS tous multiples de 8 : l'alignement reste commun, donc
        # changer d'avis n'invalide jamais le travail deja fait.
        PALIERS = (8, 16, 32, 64, 128)
        i = min(range(len(PALIERS)), key=lambda j: abs(PALIERS[j] - s.lot))
        j = i
        if util < 0.30 and i > 0:
            j = i - 1
        elif util > 0.90 and i < len(PALIERS) - 1:
            j = i + 1
        if PALIERS[j] != s.lot:
            self.trace.append((self.n, 'lot', s.lot, PALIERS[j],
                               f'utilisation {100*util:.0f} %'))
            s.lot = PALIERS[j]
            self.decisions += 1
            self.derniere = self.n
        self.precalcules, self.lus = 0, set()
        self.dernier_taux = taux
        self.t_agent += time.perf_counter() - t0

    def bilan(self):
        return {'reveils': self.reveils, 'decisions': self.decisions,
                'temps_agent_s': self.t_agent, 'trace': self.trace}


# ═══════════════════════════════════════════════════ LA CHARGE DE TRAVAIL
def charge(k0, n=1200, graine=0):
    """Trois régimes qui se succèdent. AUCUN réglage fixe ne peut être bon
    sur les trois — c'est là que se joue l'utilité d'un agent.

      phase 1  balayage serré        : forte localité
      phase 2  accès dispersés       : le cache ne sert presque à rien
      phase 3  boucle sur peu de blocs : le cache sert énormément
    """
    rng = np.random.default_rng(graine)
    a = [k0 + i for i in range(n // 3)]
    b = [k0 + int(x) for x in rng.integers(0, 40000, n // 3)]
    c = [k0 + 5000 + int(x) for x in rng.integers(0, 40, n - 2 * (n // 3))]
    return a + b + c, (len(a), len(b), len(c))
