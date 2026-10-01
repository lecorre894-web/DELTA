import sys,json,sqlite3,os,time
from delta_qpu_cache import DB
def export(path):
    c=sqlite3.connect(DB);rows=c.execute("select k,mode,counts,origin,created from r").fetchall()
    json.dump({"format":"DELTA_MIRROR_V1","exported":time.time(),"entries":[dict(zip(("k","mode","counts","origin","created"),r)) for r in rows]},open(path,"w"),indent=1)
    print("MIRROR_EXPORT entries=%d -> %s"%(len(rows),path))
def imp(path):
    d=json.load(open(path));assert d.get("format")=="DELTA_MIRROR_V1","FORMAT";c=sqlite3.connect(DB)
    c.execute("create table if not exists r(k text primary key,mode text,counts text,origin text,created real,hits integer default 0)")
    new=same=conflict=0
    for e in d["entries"]:
        row=c.execute("select counts,origin from r where k=?",(e["k"],)).fetchone()
        if row is None:c.execute("insert into r(k,mode,counts,origin,created) values(?,?,?,?,?)",(e["k"],e["mode"],e["counts"],e["origin"],e["created"]));new+=1
        elif row==(e["counts"],e["origin"]):same+=1
        else:conflict+=1
    c.commit();print("MIRROR_IMPORT new=%d identical=%d conflicts_kept_local=%d"%(new,same,conflict))
    print("MIRROR_VALIDATION=%s"%("OK" if conflict==0 else "CHECK_CONFLICTS"))
if __name__=="__main__":
    {"--export":export,"--import":imp}[sys.argv[1]](sys.argv[2] if len(sys.argv)>2 else "delta_mirror.json")
