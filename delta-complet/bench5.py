import numpy as np, time, json, delta4 as D4, delta5 as D5
chi, theta, depth = 16, 0.6, 10
layers = D4.circuit_uniforme(depth, 5, theta)
rng = np.random.default_rng(42)
def portes_grappe(debut, largeur):
    return {(c, debut + k): D5.ry(rng.uniform(-1.5, 1.5)) for c in range(depth) for k in range(largeur)}

print('=== 1. CONTRÔLE : hybride contre une chaîne finie complète de 4 000 qubits ===')
P = portes_grappe(2000, 64)
t0 = time.time(); ref = D5.run_chaine(4000, chi, layers, P, 0); tr = time.time() - t0
zref = ref.z_profil()
t0 = time.time(); h = D5.Hybride(chi, layers); mg = h.allumer(2000, 64, P); th = time.time() - t0
sites = list(range(1900, 2200)) + [100, 1000, 3000, 3900]
ecart = max(abs(h.z(s) - zref[s]) for s in sites)
ecart_centre = max(abs(h.z(s) - zref[s]) for s in range(1950, 2114))
print(f'chaîne complète 4000 : {ref.memory()/1024:.0f} Kio, {tr:.2f} s')
print(f'hybride              : {h.memory()/1024:.0f} Kio, {th:.2f} s  (fond {h.fond.memory()} o + grappe {mg/1024:.0f} Kio)')
print(f'écart max <Z> sur {len(sites)} sites (grappe, zone d influence, bords, lointains) : {ecart:.1e}')
print(f'effet réel de la grappe : <Z> fond = {h.z_fond[0]:+.4f} ; dans la grappe de {min(zref[2000:2064]):+.4f} à {max(zref[2000:2064]):+.4f}')

print('\n=== 2. TROIS GRAPPES ALLUMÉES DANS UNE CHAÎNE INFINIE ===')
h2 = D5.Hybride(chi, layers)
pos = [0, 10**6, 10**15]
for p in pos:
    t0 = time.time(); m = h2.allumer(p, 64, portes_grappe(p, 64))
    print(f'grappe de 64 qubits individuels au site {p:>20,} : {m/1024:.0f} Kio, {time.time()-t0:.2f} s')
print(f'mémoire totale : {h2.memory()/1024:.0f} Kio pour une infinité de qubits, dont {3*64} individuels')
for p in [32, 10**6 + 32, 10**15 + 32, 5 * 10**14, 10**18]:
    print(f'   <Z> au site {p:>22,} = {h2.z(p):+.6f}')
json.dump(dict(ecart_max=ecart, mem_ref=ref.memory(), mem_hybride=h.memory()), open('resultats_delta5.json', 'w'), indent=1)
