import os
f="delta_qpu_cache.py";s=open(f).read()
R=[("from qiskit import QuantumCircuit,qasm2","from qiskit import QuantumCircuit,qasm2\nfrom delta_pack import pack,unpack",1),
("json.loads(row[1])","unpack(row[1])",2),("json.dumps(counts)","pack(counts)",1),("json.dumps(c),json.dumps(o)","pack(c),json.dumps(o)",1)]
for a,b,n in R:
    assert s.count(a)==n,"ANCRE %s (%d)"%(a,s.count(a));s=s.replace(a,b)
open(f,"w").write(s)
g="delta_mirror.py";m=open(g).read()
R2=[("import sys,json,sqlite3,os,time","import sys,json,sqlite3,os,time,base64\nfrom delta_pack import unpack\ndef _e(v):return \"b64:\"+base64.b64encode(v).decode() if isinstance(v,bytes) else v\ndef _d(v):return base64.b64decode(v[4:]) if isinstance(v,str) and v.startswith(\"b64:\") else v",1),
('"entries":[dict(zip(("k","mode","counts","origin","created"),r)) for r in rows]','"entries":[dict(zip(("k","mode","counts","origin","created"),(r[0],r[1],_e(r[2]),r[3],r[4]))) for r in rows]',1),
('    for e in d["entries"]:\n','    for e in d["entries"]:\n        e["counts"]=_d(e["counts"])\n',1),
("c=json.loads(counts);top","c=unpack(counts);top",1)]
for a,b,n in R2:
    assert m.count(a)==n,"ANCRE MIROIR %s"%a[:40];m=m.replace(a,b)
open(g,"w").write(m)
import sqlite3,json
from delta_pack import pack
db=os.path.join(os.path.dirname(os.path.abspath(f)),"delta_qpu_cache.db")
if os.path.exists(db):
    c=sqlite3.connect(db);before=after=0;n=0
    for k,v in c.execute("select k,counts from r").fetchall():
        if isinstance(v,str):
            p=pack(json.loads(v));before+=len(v);after+=len(p);n+=1;c.execute("update r set counts=? where k=?",(p,k))
    c.commit();c.execute("vacuum")
    print("MIGRATION entrees=%d COUNTS_AVANT=%d o APRES=%d o GAIN=x%.1f"%(n,before,after,before/max(after,1)))
print("PATCH PACK OK")
