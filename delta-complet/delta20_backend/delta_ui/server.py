"""DELTA CONTROL UI : serveur local, bibliotheque standard uniquement.
Securite : ecoute 127.0.0.1 par defaut (0.0.0.0 seulement avec --host 0.0.0.0), commandes UNIQUEMENT depuis
command_registry.json (liste argv, jamais de shell, jamais d'eval), aucun secret envoye au navigateur."""
import json,os,re,signal,subprocess,sys,threading,time,argparse
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import discovery
UI=Path(__file__).resolve().parent;ROOT=UI.parent;STATIC=UI/"static";STATE=UI/"state";STATE.mkdir(exist_ok=True)
LOCK=threading.Lock();LOG=[];SEQ=[0];JOBS={};PROC={};INV={"t":0,"d":None}
MASQUE=[re.compile(r"(?i)\b([A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API_KEY|KAGGLE_KEY)[A-Z0-9_]*)\s*[=:]\s*\S+"),re.compile(r"crn:v1:\S+")]
def masque(t):
    t=MASQUE[0].sub(lambda m:m.group(1)+"=***",t);return MASQUE[1].sub("crn:***",t)
def log(ligne):
    with LOCK:
        SEQ[0]+=1;LOG.append({"n":SEQ[0],"t":time.time(),"line":masque(ligne)});del LOG[:-1500]
def registre():
    try:cmds=json.loads((UI/"command_registry.json").read_text())["commands"]
    except Exception as e:log("[ui] registre illisible : %s"%e);return []
    out=[]
    for c in cmds:
        argv=c.get("argv")
        if not (isinstance(argv,list) and argv and all(isinstance(a,str) for a in argv)):continue
        script=next((a for a in argv[1:] if a.endswith(".py")),None)
        c=dict(c,available=bool(script is None or (ROOT/script).is_file()),running=any(j["command_id"]==c["id"] and j["status"]=="RUNNING" for j in JOBS.values()))
        out.append(c)
    return out
def inventaire():
    if time.time()-INV["t"]>5 or INV["d"] is None:INV["d"]=discovery.inventaire();INV["t"]=time.time()
    return INV["d"]
def sauver_jobs():
    try:(STATE/"jobs.json").write_text(json.dumps(list(JOBS.values())[-200:],indent=1))
    except Exception:pass
def charger_jobs():
    try:
        for j in json.loads((STATE/"jobs.json").read_text()):
            if j.get("status")=="RUNNING":j["status"]="FAILED";j["note"]="serveur redemarre pendant l'execution"
            JOBS[j["job_id"]]=j
    except Exception:pass
def lancer(cid,confirme):
    c=next((x for x in registre() if x["id"]==cid),None)
    if not c:return 404,{"error":"commande inconnue (absente de command_registry.json)"}
    if not c["available"]:return 409,{"error":"programme introuvable dans le depot"}
    if c["running"]:return 409,{"error":"deja en cours"}
    if c.get("dangerous") and not confirme:return 428,{"error":"confirmation requise"}
    jid="J%d"%int(time.time()*1000);j={"job_id":jid,"command_id":cid,"label":c["label"],"started":time.time(),"ended":None,
       "duration_us":None,"duration_ms":None,"returncode":None,"status":"RUNNING"};JOBS[jid]=j;sauver_jobs()
    def travail():
        t0=time.perf_counter_ns();log("[%s] lancement : %s"%(cid," ".join(c["argv"])))
        try:
            p=subprocess.Popen(c["argv"],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors="replace",start_new_session=True)
            PROC[jid]=p
            for l in p.stdout:log("[%s] %s"%(cid,l.rstrip()))
            rc=p.wait()
        except Exception as e:log("[%s] erreur : %s"%(cid,e));rc=-999
        ns=time.perf_counter_ns()-t0
        j.update(ended=time.time(),duration_us=round(ns/1e3,1),duration_ms=round(ns/1e6,3),returncode=rc,
                 status="PASSED" if rc==0 else ("STOPPED" if j.get("stop") else "FAILED"))
        PROC.pop(jid,None);log("[%s] termine : %s en %.3f ms (code %s)"%(cid,j["status"],j["duration_ms"],rc));sauver_jobs();INV["t"]=0
    threading.Thread(target=travail,daemon=True).start();return 202,j
def arreter():
    n=0
    for jid,p in list(PROC.items()):
        try:JOBS[jid]["stop"]=True;os.killpg(os.getpgid(p.pid),signal.SIGTERM);n+=1
        except Exception:pass
    log("[ui] arret demande (%d job(s))"%n);return {"stopped":n}
def statut():
    inv=inventaire();js=list(JOBS.values())
    dspc=[m for m in inv["modules"] if m["type"]=="DSPC"]+[c for c in inv["components"] if c["type"]=="DSPC"]
    val=next((r for r in inv["results"] if r["validation"]),None)
    fin=[j for j in js if j["status"]!="RUNNING"];der=max(fin,key=lambda j:j["ended"] or 0) if fin else None
    return {"delta":"ONLINE","timestamp":time.time(),"dspc":{"detected":len(dspc)} if dspc else None,
            "components":len(inv["components"]),"modules":len(inv["modules"]),"unknown":len(inv["unknown"]),
            "jobs":{"total":len(js),"running":sum(j["status"]=="RUNNING" for j in js),"passed":sum(j["status"]=="PASSED" for j in js),"failed":sum(j["status"]=="FAILED" for j in js)},
            "last_validation":{"file":val["file"],"mtime":val["mtime"]} if val else None,
            "last_job":{k:der[k] for k in ("label","status","ended","duration_ms")} if der else None}
TYPES={".html":"text/html; charset=utf-8",".css":"text/css; charset=utf-8",".js":"text/javascript; charset=utf-8",".svg":"image/svg+xml"}
class H(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def envoi(self,code,corps,typ="application/json; charset=utf-8"):
        b=corps if isinstance(corps,bytes) else json.dumps(corps).encode()
        self.send_response(code);self.send_header("Content-Type",typ);self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(b)
    def do_GET(self):
        u=urlparse(self.path);p=u.path;q=parse_qs(u.query)
        if p=="/api/status":return self.envoi(200,statut())
        if p=="/api/components":return self.envoi(200,inventaire()["components"])
        if p=="/api/results":return self.envoi(200,inventaire()["results"])
        if p=="/api/modules":i=inventaire();return self.envoi(200,{"modules":i["modules"],"unknown":i["unknown"]})
        if p=="/api/commands":return self.envoi(200,[{k:c[k] for k in ("id","label","argv","category","dangerous","available","running")} for c in registre()])
        if p=="/api/jobs":return self.envoi(200,sorted(JOBS.values(),key=lambda j:-j["started"])[:50])
        if p=="/api/log":
            n=int((q.get("since") or ["0"])[0] or 0)
            with LOCK:return self.envoi(200,{"last":SEQ[0],"lines":[l for l in LOG if l["n"]>n][-500:]})
        f=(STATIC/("index.html" if p=="/" else p.lstrip("/"))).resolve()
        if STATIC in f.parents and f.is_file():return self.envoi(200,f.read_bytes(),TYPES.get(f.suffix,"application/octet-stream"))
        self.envoi(404,{"error":"introuvable"})
    def do_POST(self):
        u=urlparse(self.path)
        if u.path.startswith("/api/run/"):
            code,r=lancer(u.path[9:],parse_qs(u.query).get("confirm")==["1"]);return self.envoi(code,r)
        if u.path=="/api/stop":return self.envoi(200,arreter())
        self.envoi(404,{"error":"introuvable"})
if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--host",default="127.0.0.1",help="0.0.0.0 uniquement si tu le choisis");a.add_argument("--port",type=int,default=8080)
    o=a.parse_args();charger_jobs();log("[ui] DELTA Control demarre sur %s:%d (racine %s)"%(o.host,o.port,ROOT))
    print("DELTA_CONTROL_UI=PASSED | http://%s:%d | racine %s"%(o.host,o.port,ROOT),flush=True)
    try:ThreadingHTTPServer((o.host,o.port),H).serve_forever()
    except KeyboardInterrupt:arreter()
