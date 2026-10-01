f="delta_hpl.py";s=open(f).read()
R=[('rng=np.random.default_rng(42);best=0;ok=True','rng=np.random.default_rng(42);best=0;ok=True;Q={"HPL":0.0,"DGEMM":0.0,"JOBS":0.0};T0=time.perf_counter()'),
('        t,g,r=hpl(n,rng);ts.append(t);gs.append(g);ok&=r<16','        t,g,r=hpl(n,rng);ts.append(t);gs.append(g);ok&=r<16;Q["HPL"]+=2/3*n**3+2*n**2'),
('A@B;t0=time.perf_counter();A@B;t=time.perf_counter()-t0','A@B;t0=time.perf_counter();A@B;t=time.perf_counter()-t0;Q["DGEMM"]+=2*2*n**3'),
('X=np.linalg.solve(S,v[:,:,None]);t=time.perf_counter()-t0','X=np.linalg.solve(S,v[:,:,None]);t=time.perf_counter()-t0;Q["JOBS"]+=(k+100)*(2/3*32**3+2*32**2)'),
('print("HPL_VALIDATION=%s"','tot=sum(Q.values());W=time.perf_counter()-T0\nprint("--- QUANTITE DE CALCUL REELLEMENT EXECUTEE ---")\nfor kk,vv in Q.items():print("QTE_%s=%.3e operations (%.2f GFLOP)"%(kk,vv,vv/1e9))\nprint("QTE_TOTALE=%.3e operations flottantes FP64 en %.1f s de run = %.1f milliards d\'operations, resultats HPL verifies (residu<16)"%(tot,W,tot/1e9))\nprint("EQUIVALENT_HUMAIN=%.0f ans a 1 operation/seconde"%(tot/31557600))\nprint("HPL_VALIDATION=%s"')]
for a,b in R:
    assert s.count(a)==1,"ANCRE "+a[:40];s=s.replace(a,b)
open(f,"w").write(s);print("PATCH QUANTITE OK")
