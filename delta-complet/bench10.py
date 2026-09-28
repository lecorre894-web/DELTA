import numpy as np, math, time, json
import delta10 as D10, delta8 as D8

print('=== 1. VÉRITÉ EXACTE : symétrie contre simulation 2^n ===')
for n, depth in [(8, 4), (12, 8), (16, 10)]:
    occ = [(1 if k % 2 == 0 else 0) for k in range(n)]        # moitié des qubits à 1
    circ = D10.circuit_symetrique(n, depth, 3)
    ref = D10.dense_reference(n, circ, occ)
    s = D10.Delta10(n, occ, eps=1e-12)
    t0 = time.time()
    for par, g in circ: s.couche(par, g)
    dt = time.time() - t0
    v = s.dense(); F = abs(np.vdot(ref, v))**2 / np.vdot(v, v).real
    print(f'n={n:3d} prof={depth:3d} fid VRAIE={F:.12f}  X_pic={s.X_pic:4d}  '
          f'mem blocs={s.memoire()/1024:8.1f} Kio  mem sans blocs={s.memoire_dense()/1024:8.1f} Kio  '
          f'gain=x{s.memoire_dense()/s.memoire():.2f}  t={dt*1000:.0f} ms')

print('\n=== 2. GAIN DE LA SYMÉTRIE À L ÉCHELLE (même circuit, même état) ===')
res = []
for n in [16, 32, 64, 128]:
    occ = [(1 if k % 2 == 0 else 0) for k in range(n)]
    circ = D10.circuit_symetrique(n, 12, 3)
    s = D10.Delta10(n, occ, eps=1e-10, X_max=512)
    t0 = time.time()
    for par, g in circ: s.couche(par, g)
    t_sym = time.time() - t0
    # même circuit dans le moteur SANS symétrie (Δ8, X à la demande)
    circ8 = [(par, g, None) for par, g in circ]
    s8 = D8.Delta8(n, X_base=512, eps=1e-10)
    # état de départ identique (qubits à 1 un sur deux)
    X = np.array([[0, 1], [1, 0]], complex)
    s8.couche1([X if occ[k] else np.eye(2, dtype=complex) for k in range(n)])
    t0 = time.time(); D8.executer(s8, circ8); t_den = time.time() - t0
    r = dict(n=n, mem_sym=s.memoire(), mem_den=s8.memoire(), t_sym=t_sym, t_den=t_den,
             X_sym=s.X_pic, X_den=s8.X_pic, fid_sym=s.fid, fid_den=s8.fid)
    res.append(r)
    print(f'n={n:4d} | symétrie : {s.memoire()/1024:8.1f} Kio, {t_sym:6.2f} s, X={s.X_pic:4d} '
          f'| sans symétrie : {s8.memoire()/1024:8.1f} Kio, {t_den:6.2f} s, X={s8.X_pic:4d} '
          f'| gains mémoire x{s8.memoire()/s.memoire():.2f}, temps x{t_den/max(t_sym,1e-9):.2f}', flush=True)

print('\n=== 3. REFUS D UNE PORTE QUI VIOLE LA SYMÉTRIE ===')
s = D10.Delta10(8, [1,0,1,0,1,0,1,0])
H2 = np.kron(np.array([[1,1],[1,-1]])/math.sqrt(2), np.eye(2)).astype(complex)
try:
    s.porte2(0, H2)
    print('accepté (anormal)')
except D10.SymetrieViolee as e:
    print('REFUS ->', e)
json.dump(res, open('resultats_delta10.json','w'), indent=1, default=float)
