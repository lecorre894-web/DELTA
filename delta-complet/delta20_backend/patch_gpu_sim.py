s=open("delta_qpu_cache.py",encoding="utf-8").read();a="    def _sim(s,qc,shots):\n"
assert s.count(a)==1 and "delta_gpu_sim" not in s,"ANCRE ou deja patche"
b=a+'        if os.environ.get("DELTA_SIM_CPU")!="1":\n            try:\n                import torch\n                if torch.cuda.is_available():\n                    from delta_gpu_sim import sim_gpu;return sim_gpu(qc,shots)\n            except ImportError:pass\n'
open("delta_qpu_cache.py","w",encoding="utf-8",newline="\n").write(s.replace(a,b));print("PATCH_SIM_OK")
