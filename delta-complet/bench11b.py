import numpy as np, delta11 as D11, json
print('=== CÔNE DE LUMIÈRE, cas HONNÊTE : les qubits interrogés sont dans une zone ACTIVE ===')
res=[]
for n in [200,1000,4000,20000]:
    zones=[(k,6) for k in range(5,n-10,97)]
    circ=D11.circuit_zones(n,10,4,zones)
    zactive=[z for z in zones if z[0]>n//2][0]
    q=[zactive[0],zactive[0]+1,zactive[0]+2]        # au cœur d une zone qui travaille
    a=D11.Delta11(n).executer(circ)
    b=D11.Delta11(n).executer(circ,sites=q)
    ec=max(abs(a.z(x)-b.z(x)) for x in q)
    res.append(dict(n=n,portes_tout=a.executees,portes_cone=b.executees,t_tout=a.temps,t_cone=b.temps,
                    E_tout=a.energie_J(),E_cone=b.energie_J(),ecart=ec))
    print(f'n={n:6d} | tout : {a.executees:6d} portes, {a.temps*1000:8.1f} ms, {a.energie_J():.4f} J '
          f'| cône : {b.executees:3d} portes, {b.temps*1000:6.1f} ms, {b.energie_J():.6f} J '
          f'| économie d énergie {100*(1-b.energie_J()/max(a.energie_J(),1e-12)):5.2f} % | écart réponse {ec:.1e}',flush=True)
json.dump(res,open('resultats_delta11b.json','w'),indent=1,default=float)
