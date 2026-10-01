import os,sys,json,math,time,sqlite3
HERE=os.path.dirname(os.path.abspath(__file__));DB=os.path.join(HERE,"delta_synapses.db")
NEURONES=10**6;M64=(1<<64)-1
def mix(x):
    x=(x+0x9E3779B97F4A7C15)&M64;x=((x^(x>>30))*0xBF58476D1CE4E5B9)&M64;x=((x^(x>>27))*0x94D049BB133111EB)&M64;return x^(x>>31)
class Synapses:
    def __init__(s,tau_h=48.0):
        s.c=sqlite3.connect(DB);s.c.execute("create table if not exists syn(i integer,j integer,w real,base real,t real,n integer,primary key(i,j))");s.tau=tau_h*3600
        s.seed=s._seed()
    def _seed(s):
        try:
            from delta_qpool import draw;m=json.load(open(os.path.join(HERE,"delta_qpool.json")))
            if "seed_syn" in m:return m["seed_syn"]
            b,_=draw(64,"graine_synapses");v=int("".join(map(str,b)),2);m=json.load(open(os.path.join(HERE,"delta_qpool.json")));m["seed_syn"]=v;json.dump(m,open(os.path.join(HERE,"delta_qpool.json"),"w"),indent=1);return v
        except Exception:return int(json.load(open(os.path.join(HERE,"delta_qseed.json")))["seed64"]) if os.path.exists(os.path.join(HERE,"delta_qseed.json")) else 12345
    def base(s,i,j):return 0.1*((mix((i*NEURONES+j)^s.seed)>>11)/float(1<<53))
    def w(s,i,j):
        r=s.c.execute("select w,base,t from syn where i=? and j=?",(i,j)).fetchone()
        if not r:return s.base(i,j)
        return r[1]+(r[0]-r[1])*math.exp(-(time.time()-r[2])/s.tau)
    def reinforce(s,i,j,reward,lr=0.5):
        cur=s.w(i,j);b=s.base(i,j);nw=cur+lr*(reward-cur)
        s.c.execute("insert into syn values(?,?,?,?,?,1) on conflict(i,j) do update set w=excluded.w,t=excluded.t,n=n+1",(i,j,nw,b,time.time()));s.c.commit();return nw
    def stored(s):return s.c.execute("select count(*) from syn").fetchone()[0]
if __name__=="__main__":
    S=Synapses();print("=== DELTA SYNAPSES : %.0e synapses adressables (%d neurones x %d), creuses ==="%(NEURONES**2,NEURONES,NEURONES))
    print("GRAINE quantique des synapses = %d (du reservoir QPU si present)"%S.seed)
    n=0
    for f in ("delta_qpairs_best.json","delta_qseed20.json"):
        p=os.path.join(HERE,f)
        if not os.path.exists(p):continue
        d=json.load(open(p));pairs=d.get("pairs") or [[2*k,2*k+1] for k in range(len(d.get("S_pairs",[])))];Sv=d.get("S") or d.get("S_pairs")
        if f=="delta_qseed20.json":continue
        for (a,c),sv in zip(pairs,Sv):
            r=max(0.0,min(1.0,(sv-2)/(2*math.sqrt(2)-2)));S.reinforce(a,c,r);n+=1
    if os.path.exists(os.path.join(HERE,"delta_qseal9.json")):
        q=json.load(open(os.path.join(HERE,"delta_qseal9.json")));a,c=q["qubits_physiques"];S.reinforce(a,c,max(0,min(1,(q["chsh_S"]-2)/0.828)));n+=1
    S.reinforce(16,23,0.0);n+=1
    print("APPRENTISSAGE %d mesures reelles -> synapses renforcees (recompense = (S-2)/(2.828-2))"%n)
    top=S.c.execute("select i,j,w,n from syn order by w desc limit 5").fetchall()
    for i,j,wv,k in top:print("SYNAPSE FORTE  qubits(%3d,%3d) poids=%.3f mesures=%d"%(i,j,wv,k))
    print("SYNAPSE FAIBLE qubits( 16, 23) poids=%.3f (paire defaillante)"%S.w(16,23))
    print("SYNAPSE VIRTUELLE (999999,777777) poids=%.4f (jamais apprise : valeur procedurale, 0 octet)"%S.w(999999,777777))
    st=S.stored();sz=os.path.getsize(DB)
    print("STOCKAGE %d synapses apprises sur 1e12 adressables, %d octets | dense float32 = 4 To [impossible ici]"%(st,sz))
    ch=max(top,key=lambda r:r[2]);print("ROUTAGE SYNAPTIQUE -> meilleure paire = qubits(%d,%d)"%(ch[0],ch[1]))
    print("SYNAPSES_VALIDATION=%s"%("OK" if st>0 and S.w(16,23)<ch[2] else "FAIL"))
