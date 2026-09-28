"""Test du cluster : 3 nœuds réels sur le réseau local, comparés à un calcul local unique."""
import subprocess, time, sys, json, numpy as np
import delta4 as D4, delta5 as D5, delta_noeud as DN
from delta_cluster import Cluster, appel
ports = [8801, 8802, 8803]
procs = [subprocess.Popen([sys.executable, 'delta_noeud.py', str(p)], stdout=subprocess.PIPE, text=True) for p in ports]
for p in procs: print(p.stdout.readline().strip())
try:
    c = Cluster(ports)
    ref = D5.Hybride(DN.CANON['chi'], D4.circuit_uniforme(DN.CANON['profondeur'], DN.CANON['graine'], DN.CANON['theta']))
    rng = np.random.default_rng(42); d = DN.CANON['profondeur']
    pos = [(0, 1.0), (10**6, 0.71), (10**15, 0.04), (2 * 10**15, 0.92), (7 * 10**20, 1.0), (10**24, 0.5)]
    print('\n--- allumage des grappes (réparties sur le nœud le moins chargé) ---')
    for p, P in pos:
        portes = [[k, p + s, float(rng.uniform(-1.5, 1.5))] for k in range(d) for s in range(64)]
        url, r = c.allumer(p, 64, portes, P)
        ref.allumer(p, 64, {(k, s): D5.ry(a) for k, s, a in portes})
        taille_msg = len(json.dumps({'portes': portes}))
        print(f'grappe {p:>26,} P={P:4.0%} -> {url[-4:]}  {r["octets"]//1024} Kio  message envoyé {taille_msg//1024} Kio')
    print('\n--- questions routées ---')
    ecart = 0.0
    sites = [p + s for p, _ in pos for s in (-25, -5, 0, 17, 40, 63, 70, 90)] + [5 * 10**14, 10**30]
    t0 = time.time()
    for s in sites:
        url, z = c.z(s, 2 * d)
        ecart = max(ecart, abs(z - ref.z(s)))
    dt = time.time() - t0
    print(f'{len(sites)} questions en {dt:.2f} s, écart max cluster / calcul local = {ecart:.1e}')
    rep = json.dumps({'site': 10**24 + 17, 'z': c.z(10**24 + 17, 20)[1]})
    print(f'taille d une réponse sur le réseau : {len(rep)} octets')
    print('\n--- état des nœuds Δ_i = (QI, V, D, P, M) ---')
    for e in c.etat():
        print(f"{e['noeud']}: QI={e['QI']['qubits_individuels']} qubits individuels + {e['QI']['qubits_representes']} | "
              f"V={len(e['V'])} voisins | D=marge {e['D']['marge_cone_de_lumiere']} | P={e['P']} | M={e['M']['octets_total']//1024} Kio")
finally:
    c.arret(); time.sleep(0.5)
    for p in procs: p.terminate()
