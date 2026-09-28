#!/usr/bin/env python3
"""Δ-NŒUD V0.6 — un nœud autonome du Δ-QI-V CLUSTER.

Chaque nœud Δ_i = (QI, V, D, P, M) :
  QI : ses qubits individuels (les grappes qu'il héberge)
  V  : ses relations (les autres nœuds qu'il connaît)
  D  : ses dépendances (marge du cône de lumière : 2d sites de part et d'autre)
  P  : son facteur % (priorité déclarée de ses grappes)
  M  : sa mémoire locale (fond infini partagé + segments de ses grappes)

Principe : CALCUL LOCAL — LIAISON MINIMALE.
  Le fond infini est reconstruit localement par chaque nœud à partir de
  4 nombres (χ, θ, profondeur, graine) : il ne circule JAMAIS sur le réseau.
  Une grappe voyage sous forme d'angles ; une réponse, sous forme d'un nombre.

Lancer un nœud (Termux, une ligne) :
  python delta_noeud.py 8801

Interface HTTP / JSON :
  GET  /etat                       -> Δ_i = (QI, V, D, P, M)
  POST /allumer   {debut, largeur, portes:[[couche, site, angle], ...], P}
  GET  /z?site=N                   -> <Z> exact au site N (grappe locale ou fond)
  POST /relier    {url}            -> ajoute un voisin (V)
  GET  /arret                      -> arrête le nœud

Dépendances : numpy + bibliothèque standard. Compatible Termux.
"""
import sys, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import delta4 as D4
import delta5 as D5

CANON = dict(chi=16, theta=0.6, profondeur=10, graine=5)   # le fond, en 4 nombres


class Noeud:
    def __init__(self, nom, canon=CANON):
        self.nom, self.canon = nom, canon
        t0 = time.time()
        layers = D4.circuit_uniforme(canon['profondeur'], canon['graine'], canon['theta'])
        self.h = D5.Hybride(canon['chi'], layers)
        self.t_fond = time.time() - t0
        self.voisins = []
        self.grappes = []                 # métadonnées (debut, largeur, P)
        self.requetes = 0
        self.verrou = threading.Lock()

    def etat(self):
        d = self.h.d
        return {
            'noeud': self.nom,
            'QI': {'grappes': self.grappes,
                   'qubits_individuels': sum(g['largeur'] for g in self.grappes),
                   'qubits_representes': 'infini (fond partagé)'},
            'V': self.voisins,
            'D': {'profondeur': d, 'marge_cone_de_lumiere': 2 * d,
                  'segment_par_grappe': 'largeur + 4d'},
            'P': {str(g['debut']): g['P'] for g in self.grappes},
            'M': {'octets_total': self.h.memory(),
                  'octets_fond': self.h.fond.memory(),
                  'octets_grappes': self.h.memory() - self.h.fond.memory()},
            'canon': self.canon,
            'requetes_servies': self.requetes,
        }

    def allumer(self, debut, largeur, portes, P=1.0):
        pi = {(int(c), int(s)): D5.ry(float(a)) for c, s, a in portes}
        with self.verrou:
            # garde-fou : deux grappes d'un même nœud doivent rester indépendantes
            d = self.h.d
            for g in self.grappes:
                if not (debut + largeur + 2 * d <= g['debut'] or g['debut'] + g['largeur'] + 2 * d <= debut):
                    raise ValueError(f"grappe trop proche de {g['debut']} : cônes de lumière qui se croisent")
            t0 = time.time()
            m = self.h.allumer(int(debut), int(largeur), pi)
            self.grappes.append({'debut': int(debut), 'largeur': int(largeur), 'P': float(P),
                                 'octets': m, 'temps_s': round(time.time() - t0, 4)})
        return self.grappes[-1]

    def z(self, site):
        self.requetes += 1
        return self.h.z(int(site))


def serveur(port, nom=None):
    noeud = Noeud(nom or f'Δ@{port}')

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _rep(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == '/etat':
                    self._rep(200, noeud.etat())
                elif u.path == '/z':
                    site = int(parse_qs(u.query)['site'][0])
                    self._rep(200, {'site': site, 'z': noeud.z(site)})
                elif u.path == '/arret':
                    self._rep(200, {'arret': noeud.nom})
                    threading.Thread(target=httpd.shutdown, daemon=True).start()
                else:
                    self._rep(404, {'erreur': 'inconnu'})
            except Exception as e:
                self._rep(400, {'erreur': str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                corps = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if u.path == '/allumer':
                    self._rep(200, noeud.allumer(corps['debut'], corps['largeur'],
                                                 corps.get('portes', []), corps.get('P', 1.0)))
                elif u.path == '/relier':
                    noeud.voisins.append(corps['url'])
                    self._rep(200, {'V': noeud.voisins})
                else:
                    self._rep(404, {'erreur': 'inconnu'})
            except Exception as e:
                self._rep(400, {'erreur': str(e)})

    httpd = ThreadingHTTPServer(('127.0.0.1', port), H)
    print(f'{noeud.nom} prêt sur http://127.0.0.1:{port}  (fond infini construit en {noeud.t_fond*1000:.1f} ms)', flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    serveur(int(sys.argv[1]) if len(sys.argv) > 1 else 8801)
