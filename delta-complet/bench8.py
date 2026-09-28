import numpy as np, math, time, json
import delta8 as D8, duel as DU

def circuit(n, depth, seed, kind='monstre', theta=0.6):
    if kind == 'monstre':
        rng = np.random.default_rng(seed)
        return [(d % 2, [DU.haar4(rng) for _ in range(d % 2, n - 1, 2)], None) for d in range(depth)]
    C = DU.circuit_faible(n, depth, seed, theta)
    out = []
    for par, g, g1 in C:
        bonds = list(range(par, n - 1, 2))
        gates = [g[i] for i in bonds]
        p1 = [np.eye(2, dtype=complex) for _ in range(n)]
        for q, U in g1.items(): p1[q] = U
        out.append((par, gates, p1))
    return out

n = 16
print('=== A. VALIDATION : X à la demande contre la vérité exacte 2^n ===')
for kind, depth in [('faible', 20), ('monstre', 12)]:
    circ = circuit(n, depth, 7, kind)
    Cd = [(p, {i: U for i, U in zip(range(p, n - 1, 2), g2)}, {q: U for q, U in enumerate(g1)} if g1 else {}) for p, g2, g1 in circ]
    ref = DU.exact(n, Cd)
    for eps in [1e-3, 1e-6, 1e-10]:
        s = D8.executer(D8.Delta8(n, X_base=256, eps=eps), circ)
        v = s.dense(); F = abs(np.vdot(ref, v))**2 / np.vdot(v, v).real
        print(f'{kind:8s} prof={depth:3d} eps={eps:<6} X_pic={s.X_pic:4d} mem={s.memoire()/1024:8.1f} Kio  '
              f'fid VRAIE={F:.8f}  fid estimée={s.fid:.8f}')

print('\n=== B. X FIXE contre X À LA DEMANDE (même circuit, fidélité visée identique) ===')
circ = circuit(n, 12, 7, 'monstre')
Cd = [(p, {i: U for i, U in zip(range(p, n - 1, 2), g2)}, {}) for p, g2, g1 in circ]
ref = DU.exact(n, Cd)
for label, kw in [('X fixe = 32', dict(X_base=32, eps=0.0)), ('X fixe = 64', dict(X_base=64, eps=0.0)),
                  ('X à la demande eps=1e-4', dict(X_base=256, eps=1e-4)),
                  ('X à la demande eps=1e-6', dict(X_base=256, eps=1e-6))]:
    kw2 = dict(kw); eps = kw2.pop('eps')
    s = D8.Delta8(n, eps=max(eps, 0.0) if eps > 0 else 1.0, **kw2)
    if eps == 0.0: s.eps = 1.0   # X fixe : on garde toujours X_base, pas de demande
    t0 = time.time(); D8.executer(s, circ); dt = time.time() - t0
    v = s.dense(); F = abs(np.vdot(ref, v))**2 / np.vdot(v, v).real
    print(f'{label:26s} X_pic={s.X_pic:4d} mem={s.memoire()/1024:8.1f} Kio  fid={F:.6f}  t={dt*1000:7.1f} ms')

print('\n=== C. LES TROIS ARBITRAGES : plus petit budget qui finit sans alerte ===')
P = np.zeros(n); P[[3, 4, 11, 12]] = 1.0          # qubits à 100 %
circ = circuit(n, 10, 7, 'monstre')
Cd = [(p, {i: U for i, U in zip(range(p, n - 1, 2), g2)}, {}) for p, g2, g1 in circ]
ref = DU.exact(n, Cd)
res = {}
for arb in ['pourcent', 'proportionnel', 'merite']:
    ok = None
    for budget_kio in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        s = D8.Delta8(n, P=P, X_base=256, eps=1e-3, budget_octets=budget_kio * 1024, arbitrage=arb)
        try:
            D8.executer(s, circ)
            v = s.dense(); F = abs(np.vdot(ref, v))**2 / np.vdot(v, v).real
            ok = (budget_kio, F, s.X_pic, s.memoire() / 1024); break
        except D8.SaturationDelta as e:
            derniere = e.info
    if ok:
        print(f'{arb:15s} : finit à partir de {ok[0]:5d} Kio de budget | fidélité={ok[1]:.6f} X_pic={ok[2]} mem réelle={ok[3]:.0f} Kio')
    else:
        print(f'{arb:15s} : ne finit jamais dans 8 Mio — dernière alerte {derniere}')
    res[arb] = ok

print('\n=== D. ALERTE DE SATURATION (budget volontairement trop petit) ===')
s = D8.Delta8(n, P=P, X_base=256, eps=1e-6, budget_octets=16 * 1024, arbitrage='pourcent')
try:
    D8.executer(s, circ); print('fini')
except D8.SaturationDelta as e:
    print('ARRÊT ET ALERTE ->', e)

print('\n=== E. SVD RANDOMISÉE (le hasard pour CHERCHER) ===')
for alea in [False, True]:
    s = D8.Delta8(24, X_base=128, eps=1e-6, alea=alea, graine=1)
    c2 = circuit(24, 10, 3, 'monstre')
    t0 = time.time(); D8.executer(s, c2); dt = time.time() - t0
    print(f'alea={alea!s:5s} X_pic={s.X_pic:4d} mem={s.memoire()/1024:8.1f} Kio fid estimée={s.fid:.6f} t={dt:6.2f} s')
json.dump(res, open('resultats_delta8.json', 'w'), indent=1, default=float)
