c=open("delta_core.py").read();a='        p=os.path.join(HERE,"delta_station_route.json")'
assert c.count(a)==1,"ANCRE CORE"
c=c.replace(a,'        import shutil;o=os.path.join(HERE,"delta_station_route_overlay.json");p=o if os.path.exists(o) and not shutil.which("nvidia-smi") else os.path.join(HERE,"delta_station_route.json")')
s=open("delta_station.py").read();b='json.dump(R,open("delta_station_route.json","w"),indent=1)';d='print("ROUTAGE ECRIT -> delta_station_route.json")'
assert s.count(b)==1 and s.count(d)==1,"ANCRE STATION"
s=s.replace(b,'RF="delta_station_route.json" if torch.cuda.is_available() else "delta_station_route_overlay.json";json.dump(R,open(RF,"w"),indent=1)').replace(d,'print("ROUTAGE ECRIT -> "+RF)')
open("delta_core.py","w").write(c);open("delta_station.py","w").write(s);print("PATCH_SPLIT_OK")
