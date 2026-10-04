import json,time,subprocess,threading,os,re,signal
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler

ROOT=Path(__file__).resolve().parent.parent
LOG=[]; JOBS={}; PROC={}; LOCK=threading.Lock()

CMDS={
 "full":["python3","delta_world_benchmark.py"],
 "xeon":["python3","delta_final_benchmark.py"],
 "delta":["python3","delta_overlay_scheduler.py"],
 "ryzen":["python3","delta_ryzen_emu.py"],
 "rtx":["python3","delta_rtx_emu_fast.py"],
 "qpu":["python3","delta_backend/ibm_qpu_execute.py"],
}

def exists(cmd):
 return len(cmd)<2 or (ROOT/cmd[1]).exists()

def number(v):
 return isinstance(v,(int,float)) and not isinstance(v,bool)

def walk(o,path=""):
 out=[]
 if isinstance(o,dict):
  for k,v in o.items(): out+=walk(v,path+"."+str(k).lower())
 elif isinstance(o,list):
  for i,v in enumerate(o): out+=walk(v,path+f".{i}")
 elif number(o): out.append((path,o))
 return out

def metric(data,names):
 for p,v in walk(data):
  if any(x in p for x in names): return v
 return None

def load_jsons():
 out=[]
 for p in ROOT.rglob("*.json"):
  if any(x in str(p) for x in ["/.git/","node_modules"]): continue
  try:
   d=json.loads(p.read_text(errors="ignore"))
   out.append((p.stat().st_mtime,p.name,d))
  except: pass
 return sorted(out,reverse=True)

def latest():
 fs=load_jsons()
 # priorité au benchmark monde frais
 for _,n,d in fs:
  if n=="delta_world_benchmark_fresh.json": return d,n
 for _,n,d in fs:
  if "result" in n.lower(): return d,n
 return ({},None)

def engines():
    d,src=latest()

    aliases={
        "XEON":["xeon"],
        "DELTA":["delta"],
        "RYZEN 9 9950X3D":["ryzen"],
        "RTX 4080":["rtx"],
        "IBM QPU":["qpu","ibm"]
    }

    results=d.get("results",[]) if isinstance(d,dict) else []
    res=[]

    for name,keys in aliases.items():
        block={}

        # Format DELTA_WORLD_FRESH_V1 :
        # les moteurs sont stockés dans results[]
        for item in results:
            if not isinstance(item,dict):
                continue
            label=str(item.get("label","")).lower()
            if any(k in label for k in keys):
                block=item
                break

        us=block.get("time_us")
        ms=block.get("time_ms")
        gf=block.get("gflops")

        qb=(
            block.get("qbits")
            if block.get("qbits") is not None
            else block.get("qubits")
        )
        if qb is None:
            qb=block.get("physicalQubits")

        jobs=block.get("jobs")
        if jobs is None and isinstance(block.get("job_ids"),list):
            jobs=len(block["job_ids"])

        status=block.get(
            "status",
            block.get("validation","N/A")
        ) if block else "N/A"

        classification=block.get(
            "classification","N/A"
        ) if block else "N/A"

        res.append(dict(
            name=name,
            status=status,
            classification=classification,
            us=us,
            ms=ms,
            gflops=gf,
            qbits=qb,
            jobs=jobs,
            source=src
        ))

    return res

def modules():
 a=[]
 for p in sorted(ROOT.glob("delta_*.py")):
  n=p.stem.lower()
  t="QPU" if "qpu" in n else "GPU" if "gpu" in n or "rtx" in n else \
    "CPU" if "cpu" in n or "ryzen" in n else "DELTA"
  a.append({"name":p.stem,"type":t})
 return a

def network():
 try:
  ints=[x.split(":")[0].strip() for x in
   Path("/proc/net/dev").read_text().splitlines()[2:]
   if x.split(":")[0].strip()!="lo"]
  t=time.perf_counter_ns()
  r=subprocess.run(["ping","-c","1","-W","1","1.1.1.1"],
    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  ms=(time.perf_counter_ns()-t)/1e6
  return {"status":"ONLINE" if r.returncode==0 else "LOCAL",
          "interfaces":ints,"latency_ms":round(ms,3)}
 except Exception as e:return {"status":"UNKNOWN","interfaces":[],"latency_ms":None}

def start(kind):
 cmd=CMDS.get(kind)
 if not cmd or not exists(cmd): return {"error":"PROGRAM_NOT_FOUND","kind":kind}
 jid=str(int(time.time()*1000))
 JOBS[jid]={"id":jid,"kind":kind,"status":"RUNNING","us":None,"ms":None}
 def work():
  t=time.perf_counter_ns()
  try:
   p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,text=True,start_new_session=True)
   PROC[jid]=p
   for line in p.stdout:
    with LOCK:
     LOG.append(f"[{kind}] {line.rstrip()}")
     del LOG[:-500]
   rc=p.wait()
   st="PASSED" if rc==0 else "STOPPED" if rc<0 else "FAILED"
  except Exception as e:
   LOG.append(f"[{kind}] ERROR {e}");st="FAILED"
  ns=time.perf_counter_ns()-t
  JOBS[jid].update(status=st,us=round(ns/1e3,3),ms=round(ns/1e6,6))
  PROC.pop(jid,None)
 threading.Thread(target=work,daemon=True).start()
 return JOBS[jid]

def stop():
 for jid,p in list(PROC.items()):
  try: os.killpg(os.getpgid(p.pid),signal.SIGTERM)
  except: pass
 return {"status":"STOP_REQUESTED","count":len(PROC)}

HTML=r'''<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>DELTA CONTROL V2</title><style>
*{box-sizing:border-box}body{margin:0;background:#06131c;color:#dcecf3;font:14px system-ui}
header,main{padding:18px 4%}header{font-size:25px;font-weight:800;border-bottom:1px solid #28404d}
small{color:#7f9aa8}.bar{position:sticky;top:0;z-index:5;display:flex;gap:8px;
overflow:auto;padding:10px 4%;background:#081823;border-bottom:1px solid #28404d}
.pill,.card,.box{border:1px solid #294858;background:#0b1d28;border-radius:10px}
.pill{padding:10px;white-space:nowrap}.green,b{color:#43e28a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.card{padding:15px}.card h2{margin:4px 0 12px}.row{display:flex;justify-content:space-between;
padding:6px;border-top:1px solid #203a47}.box{padding:14px;margin-top:15px}
button{padding:11px 14px;margin:4px;background:#143447;color:#fff;border:1px solid #477087;
border-radius:8px;font-weight:700}button:active{transform:scale(.97)}
.stop{border-color:#a94c4c}pre{height:240px;overflow:auto;background:#02080b;
padding:12px;color:#a8e9bd}details{margin-top:15px}summary{cursor:pointer;font-weight:bold}
.mod{padding:5px;border-bottom:1px solid #18313d}</style></head><body>
<header>Δ DELTA CONTROL V2<br><small>CONTROL • ÉTAT • MESURES</small></header>
<div class=bar>
<div class=pill>DELTA <b id=online>ONLINE</b></div>
<div class=pill>RÉSEAU <b id=net>...</b></div>
<div class=pill>LATENCE <b id=lat>N/A</b></div>
<div class=pill>DSPC <b>1/1</b></div>
<div class=pill>JOBS <b id=jobs>0</b></div>
<div class=pill>µs <b id=us>N/A</b></div>
<div class=pill>ms <b id=ms>N/A</b></div>
<div class=pill>GFLOPS <b id=gf>N/A</b></div>
<div class=pill>QBITS <b id=qb>N/A</b></div></div>
<main>
<div class=box><b>COMMANDES</b><br>
<button onclick="go('full')">BENCHMARK COMPLET</button>
<button onclick="go('xeon')">XEON</button>
<button onclick="go('delta')">DELTA</button>
<button onclick="go('ryzen')">RYZEN</button>
<button onclick="go('rtx')">RTX</button>
<button onclick="go('qpu')">IBM QPU</button>
<button onclick="refresh()">ACTUALISER</button>
<button class=stop onclick="halt()">STOP</button></div>
<h2>MOTEURS</h2><div class=grid id=eng></div>
<div class=box><b>RÉSEAU</b><div id=network></div></div>
<div class=box><b>LIVE LOG</b><pre id=log>DELTA prêt.</pre></div>
<details class=box><summary id=mt>MODULES</summary><div id=mods></div></details>
</main><script>
const $=x=>document.getElementById(x);
const A=async(u,o)=>await(await fetch(u,o)).json();
const V=x=>x===null||x===undefined?"N/A":x;
const row=(a,b)=>`<div class=row><span>${a}</span><b>${V(b)}</b></div>`;
async function go(k){await A("/run/"+k,{method:"POST"});refresh()}
async function halt(){await A("/stop",{method:"POST"});refresh()}
async function refresh(){
 let [e,n,j,l,m]=await Promise.all([A("/engines"),A("/network"),A("/jobs"),A("/log"),A("/modules")]);
 $("net").textContent=n.status;$("lat").textContent=n.latency_ms==null?"N/A":n.latency_ms+" ms";
 $("jobs").textContent=j.length;
 $("network").innerHTML=row("ÉTAT",n.status)+row("INTERFACES",n.interfaces.join(", "))+row("LATENCE",n.latency_ms+" ms");
 $("eng").innerHTML=e.map(x=>`<div class=card><small>${x.classification}</small><h2>${x.name}</h2>
 ${row("ÉTAT",x.status)}${row("JOBS",x.jobs)}${row("µs",x.us)}${row("ms",x.ms)}
 ${row("GFLOPS",x.gflops)}${row("QBITS",x.qbits)}<small>${x.source||""}</small></div>`).join("");
 let vals=e.filter(x=>x.status!="N/A");if(vals.length){let x=vals[0];$("us").textContent=V(x.us);$("ms").textContent=V(x.ms);
 $("gf").textContent=V(x.gflops);$("qb").textContent=V(x.qbits)}
 $("log").textContent=l.join("\n")||"DELTA prêt.";$("log").scrollTop=$("log").scrollHeight;
 $("mt").textContent=`MODULES (${m.length})`;
 $("mods").innerHTML=m.map(x=>`<div class=mod><b>${x.type}</b> ${x.name}</div>`).join("")
}refresh();setInterval(refresh,2000)</script></body></html>'''

class H(BaseHTTPRequestHandler):
 def out(self,x,code=200,typ="application/json"):
  b=x.encode() if isinstance(x,str) else json.dumps(x).encode()
  self.send_response(code);self.send_header("Content-Type",typ)
  self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(b)
 def do_GET(self):
  p=self.path
  if p=="/":return self.out(HTML,typ="text/html;charset=utf-8")
  if p=="/engines":return self.out(engines())
  if p=="/network":return self.out(network())
  if p=="/jobs":return self.out(list(JOBS.values()))
  if p=="/log":return self.out(LOG)
  if p=="/modules":return self.out(modules())
  self.out({"error":"404"},404)
 def do_POST(self):
  if self.path.startswith("/run/"):return self.out(start(self.path[5:]),202)
  if self.path=="/stop":return self.out(stop())
  self.out({"error":"404"},404)

print("DELTA_CONTROL_V2=PASSED")
print("http://0.0.0.0:8080")
ThreadingHTTPServer(("0.0.0.0",8080),H).serve_forever()
