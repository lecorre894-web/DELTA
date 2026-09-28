#!/usr/bin/env python3
"""Δ-CLUSTER V0.6 — coordinateur : répartit les grappes, route les questions.

Le coordinateur ne calcule rien. Il tient une carte « intervalle de sites ->
nœud » et envoie chaque question au seul nœud concerné (liaison minimale).
Un site hors de toute grappe est servi par n'importe quel nœud : tous portent
le même fond infini, reconstruit localement.

Répartition : chaque nouvelle grappe va au nœud le moins chargé en octets.
Le facteur % (P) est transmis et enregistré par le nœud ; il ne pilote pas
encore le placement (piste V0.7 : placer d'abord les grappes à P élevé).

Usage (Termux, lignes séparées) :
  python delta_noeud.py 8801 &
  python delta_noeud.py 8802 &
  python delta_cluster.py 8801 8802
"""
import sys, json, urllib.request


def appel(url, corps=None):
    data = None if corps is None else json.dumps(corps).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


class Cluster:
    def __init__(self, ports):
        self.noeuds = [f'http://127.0.0.1:{p}' for p in ports]
        self.carte = []                        # (lo, hi, url)
        for a in self.noeuds:                  # V : chaque nœud connaît les autres
            for b in self.noeuds:
                if a != b:
                    appel(a + '/relier', {'url': b})

    def charge(self, url):
        return appel(url + '/etat')['M']['octets_grappes']

    def allumer(self, debut, largeur, portes, P=1.0):
        cible = min(self.noeuds, key=self.charge)
        r = appel(cible + '/allumer', {'debut': debut, 'largeur': largeur, 'portes': portes, 'P': P})
        self.carte.append((debut, debut + largeur, cible))
        return cible, r

    def z(self, site, marge):
        for lo, hi, url in self.carte:
            if lo - marge <= site < hi + marge:
                return url, appel(f'{url}/z?site={site}')['z']
        url = self.noeuds[site % len(self.noeuds)]
        return url, appel(f'{url}/z?site={site}')['z']

    def etat(self):
        return [appel(u + '/etat') for u in self.noeuds]

    def arret(self):
        for u in self.noeuds:
            try:
                appel(u + '/arret')
            except Exception:
                pass


if __name__ == '__main__':
    import numpy as np
    ports = [int(p) for p in sys.argv[1:]] or [8801, 8802]
    c = Cluster(ports)
    rng = np.random.default_rng(42)
    d = 10
    for p, P in [(0, 1.0), (10**6, 0.71), (10**15, 0.04)]:
        portes = [[k, p + s, float(rng.uniform(-1.5, 1.5))] for k in range(d) for s in range(64)]
        url, r = c.allumer(p, 64, portes, P)
        print(f'grappe {p:>20,} (P={P:.0%}) -> {url}  {r["octets"]//1024} Kio  {r["temps_s"]} s')
    for s in [32, 10**6 + 32, 10**15 + 32, 10**18]:
        url, z = c.z(s, 2 * d)
        print(f'<Z>({s:>22,}) = {z:+.6f}   servi par {url}')
