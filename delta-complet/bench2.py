import numpy as np, delta2 as D2, time, math, json
def human(b):
    for u in ['o','Kio','Mio','Gio']:
        if b<1024: return f'{b:.0f} {u}'
        b/=1024
    return f'{b:.1f} Tio'
out={}
print('=== 1. VÉRITÉ TERRAIN n=16 : profondeur croissante (l intrication monte) θ=0.6 ===')
r1=[]
for depth in [10,20,40]:
    L=D2.circuit_faible(16,depth,5,0.6); ref=D2.dense_reference(16,L)
    for chi in [2,4,8,16]:
        s=D2.run2(D2.Delta2(16,chi),L); v=s.dense(); F=abs(np.vdot(ref,v))**2/np.vdot(v,v).real
        r1.append(dict(depth=depth,chi=chi,fid_est=s.fid,fid_vraie=F))
        print(f'profondeur={depth:3d} χ={chi:2d}  fid_est={s.fid:.6f}  fid_VRAIE={F:.6f}')
out['verite']=r1
print('\n=== 2. PASSAGE À L ÉCHELLE : θ=0.6, profondeur 10, χ=4, complex64, par lots ===')
r2=[]
for n in [1000,10000,100000,1000000]:
    L=D2.circuit_faible(n,10,5,0.6)
    t0=time.time(); s=D2.run2(D2.Delta2(n,4),L); dt=time.time()-t0
    lg=math.log10(8)+n*math.log10(2)
    per_q=math.exp(s.logfid/n)
    r2.append(dict(n=n,t=dt,portes=s.gates,portes_s=s.gates/dt,mem=s.memory(),fid_globale_log10=s.logfid/math.log(10),fid_par_qubit=per_q,dense_log10=lg))
    print(f'n={n:8d}  t={dt:7.2f}s  portes={s.gates:9d}  {s.gates/dt:10.0f} portes/s  mem={human(s.memory()):>9s}  '
          f'vecteur 2^n = 10^{lg:.0f} o  fid globale=10^{s.logfid/math.log(10):.2e}  perte max/lien={s.lost_max:.1e}')
out['echelle']=r2
json.dump(out,open('resultats_delta2.json','w'),indent=1,default=float)
