import numpy as np, time, json, delta4 as D4
out=[]
for theta,depth in [(0.6,10),(0.6,20)]:
    L=D4.circuit_uniforme(depth,5,theta)
    t0=time.time(); inf=D4.run_infini(16,L); ti=time.time()-t0
    zi=inf.z_moyen()
    print(f'\n=== θ={theta} profondeur={depth} ===')
    print(f'CHAÎNE INFINIE  χ=16 : <Z> = {zi:+.8f}   mémoire = {inf.memory()} o   temps = {ti*1000:.1f} ms   fidélité/site = {np.exp(inf.logfid_site):.9f}')
    for n in [100, 1000, 4000]:
        t0=time.time(); zf,mem=D4.run_fini_centre(n,16,L); tf=time.time()-t0
        print(f'chaîne finie n={n:5d}   : <Z> centre = {zf:+.8f}   écart = {abs(zf-zi):.1e}   mémoire = {mem/1024:.0f} Kio   temps = {tf:.2f} s')
        out.append(dict(theta=theta,depth=depth,n=n,z_fini=zf,z_infini=zi,ecart=abs(zf-zi),mem_fini=mem,mem_infini=inf.memory()))
json.dump(out,open('resultats_delta4.json','w'),indent=1)
