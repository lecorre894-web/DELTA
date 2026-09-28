#!/usr/bin/env python3
"""Banc d'essai Δ1 : ce que les chiffres disent, contre la vérité terrain."""
import sys, time, json, math
import numpy as np
import delta1 as D

R = {}
def human(b):
    for u in ['o', 'Kio', 'Mio', 'Gio', 'Tio', 'Pio', 'Eio']:
        if b < 1024:
            return f'{b:.0f} {u}'
        b /= 1024
    return f'{b:.1e} Zio+'
def dense_mem(n):
    return 16 * 2 ** n            # entier Python exact, jamais de débordement

def human_big(b):
    if b < 1024 ** 7:
        return human(float(b))
    return f'10^{math.log10(16) + (b.bit_length() - 5) * math.log10(2):.0f} o'

def go(name, n, ops, **kw):
    t0 = time.time()
    sim = D.Delta1(n, **kw)
    try:
        D.run(sim, ops)
        mur = None
    except MemoryError as e:
        mur = str(e)
    dt = time.time() - t0
    return sim, dt, mur

rows = []
def line(circ, n, sim, dt, mur, extra=''):
    r = dict(circuit=circ, n=n, kmax=sim.kmax, mem_delta=sim.mem_peak,
             mem_dense_log10=math.log10(16) + n * math.log10(2), temps_s=round(dt, 3), mur=mur,
             fid_estimee=sim.fid, extra=extra)
    rows.append(r)
    lg = math.log10(16) + n * math.log10(2) - math.log10(max(sim.mem_peak, 1))
    print(f'{circ:24s} n={n:6d} k_max={sim.kmax:4d}  Δ={human(sim.mem_peak):>9s}  '
          f'dense={human_big(dense_mem(n)):>11s}  gain=x10^{lg:.1f}  t={dt:6.2f}s  '
          f'{"<<< " + mur if mur else ""} {extra}')

rng = np.random.default_rng(2026)
print('\n=== A. VALIDATION : Δ1 exact (ε=0) contre simulateur dense 2^n ===')
val = []
for circ, f in [('produit', D.c_produit), ('GHZ', D.c_ghz), ('aller-retour', D.c_aller_retour),
                ('brique d=6', lambda n, r: D.c_brique(n, 6, r)),
                ('faible d=8', lambda n, r: D.c_faible(n, 8, r))]:
    n = 14
    ops = f(n, np.random.default_rng(11))
    d = D.run(D.Delta1(n), ops)
    ref = D.run(D.Dense(n), ops)
    F = abs(np.vdot(ref.vec(), d.dense())) ** 2
    val.append((circ, F))
    print(f'{circ:14s} fidélité exacte = {F:.12f}')
R['validation'] = val

print('\n=== B. PASSAGE À L\'ÉCHELLE ===')
for n in [10, 100, 1000, 10000]:
    ops = D.c_produit(n, rng); sim, dt, mur = go('produit', n, ops); line('produit (0 intrication)', n, sim, dt, mur)
for n in [10, 100, 1000, 10000]:
    ops = D.c_vague(n, rng); sim, dt, mur = go('v', n, ops, klimit=22); line('vague flottante', n, sim, dt, mur)
for n in [10, 20, 30]:
    ops = D.c_aller_retour(n, rng); sim, dt, mur = go('ar', n, ops, klimit=22)
    kf = max(len(x[0]) for x in sim.isl.values()) if not mur else -1
    line('aller-retour chaîne', n, sim, dt, mur, f'k_final={kf}')
for n in [10, 16, 20, 22, 30]:
    ops = D.c_ghz(n, rng); sim, dt, mur = go('ghz', n, ops, klimit=22); line('GHZ (intrication totale)', n, sim, dt, mur)
# GHZ + mesure
n = 22
sim = D.Delta1(n); D.run(sim, D.c_ghz(n, rng)); k_av = max(len(x[0]) for x in sim.isl.values())
sim.measure(0, rng); k_ap = max(len(x[0]) for x in sim.isl.values())
print(f'GHZ n=22 + 1 mesure : k avant = {k_av}, k après = {k_ap}  (la mesure libère tout)')
R['ghz_mesure'] = [k_av, k_ap]
for d in [1, 2, 4]:
    n = 100
    ops = D.c_brique(n, d, rng); sim, dt, mur = go('b', n, ops, klimit=22); line(f'brique aléatoire d={d}', n, sim, dt, mur)

print('\n=== C. DOUBLE INTRICATION + PROPORTIONNÉ (circuit à intrication faible) ===')
print('    vérité terrain : n=18, comparaison à la simulation dense exacte')
n, depth = 18, 10
ops = D.c_faible(n, depth, np.random.default_rng(5), theta=0.15)
ref = D.run(D.Dense(n), ops).vec()
P_prop = np.full(n, 0.04); P_prop[[0, 6, 12]] = 1.0; P_prop[[1, 7, 13]] = 0.71; P_prop[[2, 8, 14]] = 0.92
tabC = []
for label, eps, P in [('exact ε=0', 0.0, None),
                      ('flottant ε=1e-4', 1e-4, np.zeros(n)),
                      ('flottant ε=1e-3', 1e-3, np.zeros(n)),
                      ('flottant ε=1e-2', 1e-2, np.zeros(n)),
                      ('proportionné ε=1e-2', 1e-2, P_prop)]:
    t0 = time.time(); sim = D.Delta1(n, eps=eps, P=P); D.run(sim, ops); dt = time.time() - t0
    Ft = abs(np.vdot(ref, sim.dense())) ** 2
    tabC.append(dict(mode=label, kmax=sim.kmax, mem=sim.mem_peak, fid_est=sim.fid, fid_vraie=Ft, coupes=sim.trunc, t=dt))
    print(f'{label:22s} k_max={sim.kmax:3d}  mem={human(sim.mem_peak):>9s}  '
          f'fid estimée={sim.fid:.6f}  fid VRAIE={Ft:.6f}  troncatures={sim.trunc:4d}  t={dt:.2f}s')
R['C'] = tabC

print('\n    même circuit, grande taille (plus de vérité terrain possible au-delà de ~28 qubits)')
for n in [40, 100, 300]:
    ops = D.c_faible(n, depth, np.random.default_rng(5), theta=0.15)
    for label, eps in [('exact ε=0', 0.0), ('flottant ε=1e-3', 1e-3)]:
        sim, dt, mur = go('f', n, ops, eps=eps, P=np.zeros(n), klimit=22)
        line(f'faible d=10 {label}', n, sim, dt, mur, f'fid_est={sim.fid:.4f}')

R['rows'] = rows
json.dump(R, open('resultats_delta1.json', 'w'), indent=1, default=float)
print('\nrésultats -> resultats_delta1.json')
