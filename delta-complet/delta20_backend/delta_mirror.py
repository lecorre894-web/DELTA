import sys,json,sqlite3,os,time,base64
from delta_pack import unpack
def _e(v):return "b64:"+base64.b64encode(v).decode() if isinstance(v,bytes) else v
def _d(v):return base64.b64decode(v[4:]) if isinstance(v,str) and v.startswith("b64:") else v
DB=os.environ.get("DELTA_CACHE_DB",os.path.join(os.path.dirname(os.path.abspath(__file__)),"delta_qpu_cache.db"))
def con():
    c=sqlite3.connect(DB);c.execute("create table if not exists r(k text primary key,mode text,counts text,origin text,created real,hits integer default 0)");return c
def export(path):
    rows=con().execute("select k,mode,counts,origin,created from r").fetchall()
    json.dump({"format":"DELTA_MIRROR_V1","exported":time.time(),"entries":[dict(zip(("k","mode","counts","origin","created"),(r[0],r[1],_e(r[2]),r[3],r[4]))) for r in rows]},open(path,"w"),indent=1)
    print("MIRROR_EXPORT entries=%d -> %s"%(len(rows),path))
def imp(path):
    d=json.load(open(path));assert d.get("format")=="DELTA_MIRROR_V1","FORMAT";c=con();new=same=conflict=0
    for e in d["entries"]:
        e["counts"]=_d(e["counts"])
        row=c.execute("select counts,origin from r where k=?",(e["k"],)).fetchone()
        if row is None:c.execute("insert into r(k,mode,counts,origin,created) values(?,?,?,?,?)",(e["k"],e["mode"],e["counts"],e["origin"],e["created"]));new+=1
        elif row==(e["counts"],e["origin"]):same+=1
        else:conflict+=1
    c.commit();print("MIRROR_IMPORT new=%d identical=%d conflicts_kept_local=%d"%(new,same,conflict))
    print("MIRROR_VALIDATION=%s"%("OK" if conflict==0 else "CHECK_CONFLICTS"))
def lst(_):
    for k,mode,counts,origin,created in con().execute("select k,mode,counts,origin,created from r order by created"):
        c=unpack(counts);top=sorted(c.items(),key=lambda x:-x[1])[:2];o=json.loads(origin)
        print("%s %s %s QUBITS=%d SHOTS=%d TOP=%s ORIGINE=%s"%(k[:12],mode.upper(),time.strftime("%Y-%m-%d %H:%M",time.localtime(created)),len(next(iter(c))),sum(c.values()),dict(top),o.get("job_id",o.get("engine"))))
if __name__=="__main__":
    {"--export":export,"--import":imp,"--list":lst}[sys.argv[1]](sys.argv[2] if len(sys.argv)>2 else "delta_mirror.json")
