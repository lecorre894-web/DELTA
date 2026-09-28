import numpy as np, math, time, json, delta11 as D11

print('=== 1. CONTRÔLE : travail à la demande = même résultat que tout réveiller ===')
n, depth = 64, 12
zones = [(5, 6), (30, 8), (50, 4)]
circ = D11.circuit_zones(n, depth, 4, zones)
plein = D11.Delta11(n, tol_inerte=-1).executer(circ)        # tol négative : rien n'est sauté
malin = D11.Delta11(n).executer(circ)                        # portes inertes sautées
print(f'tout réveiller : {plein.executees:5d} portes exécutées, {plein.sautees:5d} sautées, '
      f'{plein.temps*1000:7.1f} ms, {plein.flops/1e6:8.1f} Mflops')
print(f'à la demande   : {malin.executees:5d} portes exécutées, {malin.sautees:5d} sautées, '
      f'{malin.temps*1000:7.1f} ms, {malin.flops/1e6:8.1f} Mflops')
ec = max(abs(plein.z(q) - malin.z(q)) for q in range(n))
print(f'écart max sur <Z> de tous les qubits : {ec:.2e}   -> {"identique" if ec < 1e-12 else "DIFFÉRENT"}')
print(f'économie : travail x{plein.flops/max(malin.flops,1):.1f}, temps x{plein.temps/max(malin.temps,1e-9):.1f}')

print('\n=== 2. CÔNE DE LUMIÈRE : une question sur 3 qubits dans un grand registre ===')
res = []
for n in [200, 1000, 4000]:
    zones = [(k, 6) for k in range(5, n - 10, 97)]
    circ = D11.circuit_zones(n, 10, 4, zones)
    a = D11.Delta11(n).executer(circ)
    t_a, f_a = a.temps, a.flops
    q = [n // 2, n // 2 + 1, n // 2 + 2]
    b = D11.Delta11(n).executer(circ, sites=q)
    zs_a = [a.z(x) for x in q]; zs_b = [b.z(x) for x in q]
    ec = max(abs(x - y) for x, y in zip(zs_a, zs_b))
    r = dict(n=n, t_tout=t_a, t_cone=b.temps, flops_tout=f_a, flops_cone=b.flops,
             portes_tout=a.executees, portes_cone=b.executees, ecart=ec,
             E_tout=a.energie_J(), E_cone=b.energie_J())
    res.append(r)
    print(f'n={n:5d} | tout : {a.executees:6d} portes, {t_a*1000:8.1f} ms, {a.energie_J():.4f} J '
          f'| cône : {b.executees:4d} portes, {b.temps*1000:6.1f} ms, {b.energie_J():.6f} J '
          f'| économie x{t_a/max(b.temps,1e-9):7.1f} | écart sur la réponse {ec:.1e}', flush=True)

print('\n=== 3. PORTES INERTES : registre presque endormi (90 % de portes identité) ===')
n = 400
circ = D11.circuit_zones(n, 12, 4, [(0, n)], inertes=0.9)
p = D11.Delta11(n, tol_inerte=-1).executer(circ)
m = D11.Delta11(n).executer(circ)
print(f'tout réveiller : {p.executees} portes, {p.temps*1000:.0f} ms, {p.energie_J():.4f} J (estimation 2 W)')
print(f'à la demande   : {m.executees} portes, {m.temps*1000:.0f} ms, {m.energie_J():.4f} J')
print(f'écart max <Z> : {max(abs(p.z(q)-m.z(q)) for q in range(n)):.2e}  '
      f'| énergie économisée : {(1-m.energie_J()/max(p.energie_J(),1e-12))*100:.1f} %')
json.dump(res, open('resultats_delta11.json','w'), indent=1, default=float)
