import os,sys,json,time,shutil,numpy as np
from delta_mem import DeltaMem
from delta_qpu_cache import DeltaQPUCache,ghz,DB
HERE=os.path.dirname(os.path.abspath(__file__))
class DeltaCore:
    def __init__(s,tol=1e-3,budget_mb=None):
        s.mem=DeltaMem(budget_mb=budget_mb,tol=tol,cold_dir=os.path.join(HERE,"delta_mem_cold"));s.qpu=DeltaQPUCache();s.loaded=s._load()
        # DELTA_CORE_STATION_OVERLAY_V1
        s.station_route=s._load_station_route()
    def _load(s):
        p=os.path.join(s.mem.dir,"index.json");n=0
        if os.path.exists(p):
            for k,m in json.load(open(p)).items():
                if m.get("tier")=="COLD_DISK" and os.path.exists(m.get("path","")):s.mem.meta[k]=m;n+=1
        return n
    def _load_station_route(s):
        import shutil;o=os.path.join(HERE,"delta_station_route_overlay.json");p=o if os.path.exists(o) and not shutil.which("nvidia-smi") else os.path.join(HERE,"delta_station_route.json")
        if not os.path.exists(p):
            return {"available":False,"route":{"OVERLAY":"UNAVAILABLE"}}
        try:
            r=json.load(open(p))
            r["available"]=True
            return r
        except Exception as e:
            return {"available":False,"route":{"OVERLAY":"UNAVAILABLE"},"error":str(e)}

    def compute_route(s):
        r=s.station_route
        route=r.get("route",{})
        ov=r.get("overlay",{})
        return {
            "station_available":r.get("available",False),
            "overlay_route":route.get("OVERLAY","UNAVAILABLE"),
            "overlay_active":ov.get("active",False),
            "execution_view":ov.get("execution_view","NONE"),
            "cpu":ov.get("cpu",{}),
            "gpu":ov.get("gpu",{}),
            "host":ov.get("host",{}),
            "measurement":ov.get("measurement",{})
        }

    # DELTA_CORE_COMPUTE_V1
    def compute(s,name="delta_compute",work=200000):
        """
        Execute une charge deterministe par la route DELTA Station/Overlay.

        L'overlay fournit la topologie logique.
        Le calcul physique reste execute par le host disponible.
        """
        import subprocess

        route=s.compute_route()

        if not route.get("station_available"):
            raise RuntimeError("DELTA Station route unavailable")

        if not route.get("overlay_active"):
            raise RuntimeError("DELTA Overlay inactive")

        env=os.environ.copy()
        env["DELTA_OVERLAY_WORK"]=str(int(work))

        p=subprocess.run(
            [sys.executable,
             os.path.join(HERE,"delta_overlay_scheduler.py")],
            cwd=HERE,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if p.returncode != 0:
            raise RuntimeError(
                "DELTA Overlay execution failed: "
                + p.stderr[-1000:]
            )

        rp=os.path.join(
            HERE,
            "delta_overlay_scheduler_result.json"
        )

        if not os.path.exists(rp):
            raise RuntimeError(
                "DELTA Overlay result missing"
            )

        r=json.load(open(rp))

        if r.get("validation")!="PASSED":
            raise RuntimeError(
                "DELTA Overlay validation failed"
            )

        m=r["measurement"]

        result={
            "name":name,
            "engine":"DELTA_OVERLAY",
            "execution_view":
                "RYZEN_RTX_OVER_XEON",
            "host_execution":
                r["host"]["execution"],
            "overlay_cpu":
                r["overlay_cpu"]["model"],
            "overlay_threads":
                r["overlay_cpu"]["threads"],
            "scheduler":
                r["overlay_cpu"]["scheduler"],
            "overlay_gpu":
                r["overlay_gpu"]["model"],
            "overlay_gpu_execution":
                r["overlay_gpu"]["execution"],
            "logical_lanes":
                m["logical_lanes"],
            "physical_slots":
                m["physical_slots_used"],
            "iterations":
                m["iterations"],
            "wall_s":
                m["wall_s"],
            "cpu_s":
                m["cpu_s"],
            "signature":
                m["signature"],
            "validation":
                r["validation"]
        }

        return result

    def put(s,name,x,tol=None):return s.mem.put(name,x,tol)
    def get(s,name):return s.mem.get(name)
    def quantum(s,name,qc,shots=1000,physical=False,policy="quality",tol=None):
        r=s.qpu.run(qc,shots,physical,policy);n=qc.num_qubits
        if n<=20:
            p=np.zeros(2**n)
            for b,c in r["counts"].items():p[int(b,2)]=c/shots
            s.mem.put(name,p,tol if tol is not None else 1e-6)
        return r
    def checkpoint(s):
        idx={}
        for k,m in s.mem.meta.items():
            m=dict(m)
            if m["tier"]=="HOT_RAM":
                p=os.path.join(s.mem.dir,k+".bin");s.mem.hot[k].tofile(p);m["tier"]="COLD_DISK";m["path"]=p
            idx[k]=m
        json.dump(idx,open(os.path.join(s.mem.dir,"index.json"),"w"),indent=1);s.qpu.flush();return len(idx)
    def status(s):
        cold=sum(os.path.getsize(os.path.join(s.mem.dir,f)) for f in os.listdir(s.mem.dir))
        db=os.path.getsize(DB) if os.path.exists(DB) else 0;st=s.qpu.stats();rk=os.path.join(HERE,"delta_multiqpu.json")
        age=(time.time()-json.load(open(rk))["date"])/3600 if os.path.exists(rk) else -1
        return {"ram_hot_MB":round(s.mem.used/2**20,2),"ram_budget_MB":round(s.mem.budget/2**20),"objets_memoire":len(s.mem.meta),"charges_depuis_disque":s.loaded,
                "disque_froid_MB":round(cold/2**20,2),"cache_qpu_entrees":st["entries"],"cache_qpu_hits":st["hits_total"],"cache_qpu_MB":round(db/2**20,3),
                "classement_qpu_age_h":round(age,1),"disque_libre_GB":round(shutil.disk_usage(HERE).free/2**30,1),
                "compute_route":s.compute_route()}
if __name__=="__main__":
    phys="--qpu" in sys.argv;D=DeltaCore();ok=True
    print("=== DELTA CORE : memoire typee + disque + cache QPU, soudes ===")
    print("REPRISE objets recharges depuis le disque = %d"%D.loaded)
    if "ghz8_probs" in D.mem.meta:
        p=D.get("ghz8_probs");print("RELU_DU_DISQUE ghz8_probs P(00000000)=%.3f P(11111111)=%.3f TYPE=%s"%(p[0],p[-1],D.mem.meta["ghz8_probs"]["kind"]))
    rng=np.random.default_rng(1);x=rng.standard_normal(2**20)
    m=D.put("donnees_capteur",x);e=float(np.sqrt(np.mean((D.get("donnees_capteur")-x)**2)/np.mean(x*x)));ok&=e<=1e-3
    print("PUT donnees_capteur TYPE=%s %d->%d Ko ERR=%.1e %s"%(m["kind"],m["raw_bytes"]>>10,m["bytes"]>>10,e,m["tier"]))
    r=D.quantum("ghz8_probs",ghz(8),1000,physical=phys);o=r["origin"]
    print("QUANTUM ghz8 NIVEAU=%s MOTEUR=%s QPU=%s JOB=%s ROUTAGE=%s"%(r["level"],o.get("engine"),o.get("backend","-"),o.get("job_id","-"),o.get("routing")))
    p=D.get("ghz8_probs");pop=p[0]+p[-1];print("MEMOIRE ghz8_probs TYPE=%s POP_GHZ=%.3f"%(D.mem.meta["ghz8_probs"]["kind"],pop));ok&=abs(p.sum()-1)<1e-3
    print("CHECKPOINT %d objets ecrits sur disque"%D.checkpoint())
    print("STATUS "+json.dumps(D.status(),ensure_ascii=False))
    print("DELTA_CORE_VALIDATION=%s"%("OK" if ok else "FAIL"))
