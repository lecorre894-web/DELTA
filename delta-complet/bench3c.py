import numpy as np, math, time, json, resource
import delta2 as D2, delta3 as D3
n=10_000_000
L=D2.circuit_faible(n,10,5,0.6)
t0=time.time(); s=D2.run2(D3.Delta3(n,np.r_[1,[4]*(n-1),1],tranche=250_000),L); dt=time.time()-t0
mc=s.memory_compacte(); rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
r=dict(n=n,t=dt,portes=s.gates,portes_s=s.gates/dt,mem_compacte=mc,o_par_qubit=mc/n,fid=math.exp(s.logfid),logfid=s.logfid,lost_max=s.lost_max,ram_pic_gio=rss)
print(json.dumps(r,indent=1)); json.dump(r,open('resultats_delta3_10M.json','w'),indent=1)
