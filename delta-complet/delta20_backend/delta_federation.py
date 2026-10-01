import json
R=[json.loads(l) for l in open("delta_grappes_results.jsonl") if l.strip()]
best={}
for r in R:
    if r["valid"] and (r["machine"] not in best or r["rmax_gflops"]>best[r["machine"]]["rmax_gflops"]):best[r["machine"]]=r
print("=== DELTA FEDERATION : meilleur run valide par machine ===")
for m,r in sorted(best.items(),key=lambda x:-x[1]["rmax_gflops"]):
    print("%-24s %-40s CPU=%3d GRAPPES=%2d RMAX=%8.2f GFLOPS %s"%(m,r["cpu"][:40],r["cpus"],r["best_grappes"],r["rmax_gflops"],r["date"]))
tot=sum(r["rmax_gflops"] for r in best.values())
print("FEDERATION machines=%d SOMME=%.2f GFLOPS | entree TOP500 2026 = 2.66 PFLOPS -> x%.0f"%(len(best),tot,2.66e6/tot))
print("NOTE: somme de runs separes = capacite federee, pas un HPL distribue unique (non classable TOP500)")
