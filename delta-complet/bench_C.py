import time, math, json, numpy as np, delta1 as D
def human(b):
    for u in ['o','Kio','Mio','Gio']:
        if b<1024: return f'{b:.0f} {u}'
        b/=1024
    return f'{b:.1f} Tio'
out={}
print('=== C1. vérité terrain n=18, circuit faible d=10 ===')
n,depth=18,10
ops=D.c_faible(n,depth,np.random.default_rng(5),theta=0.15)
ref=D.run(D.Dense(n),ops).vec()
P=np.full(n,0.04); P[[0,6,12]]=1.0; P[[1,7,13]]=0.71; P[[2,8,14]]=0.92
rowsC=[]
for label,eps,PP,regle in [('exact ε=0',0,None,'strict'),
  ('uniforme ε=1e-4',1e-4,np.zeros(n),'strict'),('uniforme ε=1e-3',1e-3,np.zeros(n),'strict'),('uniforme ε=1e-2',1e-2,np.zeros(n),'strict'),
  ('proportionné strict ε=1e-2',1e-2,P,'strict'),('proportionné détaché ε=1e-2',1e-2,P,'detache'),('proportionné détaché ε=1e-3',1e-3,P,'detache')]:
    t0=time.time(); s=D.Delta1(n,eps=eps,P=PP,regle=regle); D.run(s,ops); dt=time.time()-t0
    st=s.dense(); Ft=abs(np.vdot(ref,st))**2
    # fidélité locale des qubits importants : état réduit exact vs Δ
    def red(v,q):
        t=v.reshape((2,)*n); M=np.moveaxis(t,q,0).reshape(2,-1); return M@M.conj().T
    imp=[0,6,12]
    floc=np.mean([np.real(np.trace(red(ref,q)@red(st,q))) + 2*math.sqrt(max(np.linalg.det(red(ref,q)).real,0)*max(np.linalg.det(red(st,q)).real,0)) for q in imp])
    rowsC.append(dict(mode=label,kmax=s.kmax,mem=s.mem_peak,fid_est=s.fid,fid_vraie=Ft,fid_qubits_100pct=floc,tronc=s.trunc,t=dt))
    print(f'{label:30s} k_max={s.kmax:3d} mem={human(s.mem_peak):>8s} fid_est={s.fid:.5f} fid_VRAIE={Ft:.5f} fid(qubits 100%)={floc:.6f} tronc={s.trunc:4d}')
out['C1']=rowsC
print('\n=== C2. grande taille, circuit faible d=10, ε=1e-3 uniforme ===')
rowsC2=[]
for n in [40,100,300,1000]:
    ops=D.c_faible(n,depth,np.random.default_rng(5),theta=0.15)
    t0=time.time(); s=D.Delta1(n,eps=1e-3,P=np.zeros(n),klimit=20)
    try: D.run(s,ops); mur=None
    except MemoryError as e: mur=str(e)
    dt=time.time()-t0
    lg=math.log10(16)+n*math.log10(2)
    rowsC2.append(dict(n=n,kmax=s.kmax,mem=s.mem_peak,fid_est=s.fid,t=dt,mur=mur,dense_log10=lg))
    print(f'n={n:5d} k_max={s.kmax:3d} mem={human(s.mem_peak):>8s} dense=10^{lg:.0f} o  fid_est={s.fid:.4f}  t={dt:.1f}s {mur or ""}')
print('\n=== C3. même circuit, intrication plus forte (θ=0.6) ===')
rowsC3=[]
n=18; ops=D.c_faible(n,depth,np.random.default_rng(5),theta=0.6); ref=D.run(D.Dense(n),ops).vec()
for eps in [1e-3,1e-2,5e-2]:
    s=D.Delta1(n,eps=eps,P=np.zeros(n)); D.run(s,ops); Ft=abs(np.vdot(ref,s.dense()))**2
    rowsC3.append(dict(eps=eps,kmax=s.kmax,mem=s.mem_peak,fid_est=s.fid,fid_vraie=Ft))
    print(f'ε={eps:<6} k_max={s.kmax:3d} mem={human(s.mem_peak):>8s} fid_est={s.fid:.4f} fid_VRAIE={Ft:.4f}')
out['C2']=rowsC2; out['C3']=rowsC3
json.dump(out,open('resultats_C.json','w'),indent=1,default=float)
