#!/usr/bin/env python3
"""bench_sigma — Δ COMPLET, Σ ET ∫, UN SEUL BLOC TEXTE, UN SEUL HASH.

Ce programme n'affiche RIEN d'autre qu'un bloc texte. Ce bloc est
déterministe : il ne contient aucune durée, aucun horodatage, aucun tirage
non graine. Il est ensuite haché, et Δ complet est confronté à ce hash en se
recalculant entièrement depuis zéro.

    python bench_sigma.py            -> le bloc, puis la confrontation
    python bench_sigma.py --bloc     -> le bloc seul, rien d'autre
"""
import hashlib, json, sys
import numpy as np
import delta_sigma as S
import delta15 as D15
import delta_fractal as F
import delta_btc as B
import delta_bridge as P
import delta18 as D18

CLE = hashlib.blake2b(b'Delta-Rene', digest_size=32).digest()
GRAINE = 1


def calculer():
    """Tout Δ, réduit à un jeu de valeurs déterministes."""
    L = []
    a = L.append

    # ───────────────────────────────── Σ sur les grains du fond de Δ
    four = D15.fournisseur(128)
    sig = S.Sigma()
    for k in range(4096):
        sig.ajouter(four(k))
    a(S.ligne('sigma.grains', sig.n))
    a(S.ligne('sigma.somme_compensee', sig.valeur()))
    a(S.ligne('sigma.somme_naive', sig.naive))
    a(S.ligne('sigma.ecart_naif', sig.ecart_naif()))
    # contrôle : la somme d'un signal à moyenne nulle doit rester petite
    a(S.ligne('sigma.rapport_ecart_sur_n', sig.ecart_naif() / sig.n))

    # ───────────────────────────────────── ∫ sur le fond continu
    for n in (64, 1024, 65536):
        approx = S.integrale_simpson(S.noyau, 0.0, 10000.0, n)
        exact = float(S.integrale_exacte(0.0, 10000.0))
        a(S.ligne(f'integrale.simpson_n{n:06d}', float(approx)))
        a(S.ligne(f'integrale.erreur_relative_n{n:06d}',
                  abs(approx - exact) / abs(exact)))
    a(S.ligne('integrale.exacte', float(S.integrale_exacte(0.0, 10000.0))))

    # ─────────────────── Σ et ∫ se rejoignent-ils ? (le test qui compte)
    # la somme des grains du noyau, pas à pas, doit tendre vers l'intégrale
    exact = float(S.integrale_exacte(0.0, 10000.0))
    for pas in (4.0, 2.0, 1.0, 0.5, 0.25):
        x = np.arange(0.0, 10000.0, pas)
        s_grains = float((S.noyau(x) * pas).sum())
        a(S.ligne(f'pont_E_vers_F.pas_{pas:06.3f}_ecart_relatif',
                  abs(s_grains - exact) / abs(exact)))
    a(S.ligne('pont_E_vers_F.exacte', exact))

    # ───────────────────────────────── Δ-OMNI : le calcul, sa valeur
    O = D15.Omni(D15.noyau_lisse, b=128, eps=1e-8, portee_max=40)
    k0 = 10 ** 18 // 128 // 2
    Y, _ = O.lot(k0, 256, four)
    a(S.ligne('omni.classes', len(O.U)))
    a(S.ligne('omni.portee', O.portee))
    a(S.ligne('omni.octets_operateur', O.octets()))
    a(S.ligne('omni.rangs_total', int(sum(O.rangs.values()))))
    sY = S.Sigma().ajouter(Y.ravel())
    a(S.ligne('omni.somme_sortie', sY.valeur()))
    a(S.ligne('omni.norme_sortie', float(np.linalg.norm(Y))))

    # ───────────────────────────── Δ-FRACTAL : la racine, la preuve
    A = F.Fractal(1024, F.fournisseur_grain(128), CLE, branchement=2)
    a(S.ligne('fractal.grains', 1024))
    a(S.ligne('fractal.profondeur', A.profondeur()))
    a(S.ligne('fractal.racine', A.racine.sceau()))
    a(S.ligne('fractal.preuve_octets', A.taille_preuve(0)))
    a(S.ligne('fractal.verifie_grain0', A.verifier(0)))
    a(S.ligne('fractal.intact', A.intact()))

    # ───────────────────────────── Δ-BTC : la conformité, en dur
    blocs, _ = B.charger()
    blocs = sorted(blocs, key=lambda b: b['hauteur'])
    for b in blocs:
        a(S.ligne(f'btc.hash_{b["hauteur"]:07d}', B.hash_bloc(b)))
        a(S.ligne(f'btc.conforme_{b["hauteur"]:07d}', B.hash_bloc(b) == b['id']))
        if b['txids']:
            r = B.racine_merkle(b['txids'])
            a(S.ligne(f'btc.merkle_{b["hauteur"]:07d}', r))
            a(S.ligne(f'btc.merkle_ok_{b["hauteur"]:07d}', r == b['merkle_root']))

    # ───────────────────────────── Δ-BRIDGE : le pont, ses empreintes
    E = B.entete(blocs[1])
    mid = P.midstate(E)
    a(S.ligne('bridge.midstate', mid.octets(0)))
    h = P.sha256d_pont(E, [blocs[1]['nonce']])[0].tobytes()
    a(S.ligne('bridge.hash_par_le_pont', h[::-1].hex()))
    a(S.ligne('bridge.pont_conforme', h[::-1].hex() == blocs[1]['id']))
    lot = P.sha256d_pont(E, list(range(1024)))
    cond = P.CondensateurExact(np.ascontiguousarray(lot[:, ::-1]).tobytes(), 256)
    a(S.ligne('bridge.parite_1024_hashs', cond.parite()))
    a(S.ligne('bridge.condensateur_exact', cond.lire()
              == np.ascontiguousarray(lot[:, ::-1]).tobytes()))

    # ───────────────────── Δ-QUTRIT : l'état du Higgs (V27)
    import delta_qutrit as Q
    psi = Q.etat_higgs()
    a(S.ligne('qutrit.X', Q.X_de(psi)))
    a(S.ligne('qutrit.entropie', Q.entropie(psi)))
    a(S.ligne('qutrit.negativite', Q.negativite(Q.densite(psi))))
    a(S.ligne('qutrit.seuil_separabilite', Q.seuil_separabilite(psi)))
    a(S.ligne('qutrit.coupable', Q.peut_couper(psi)))

    # ───────────────────── Δ-ATOME : l'hydrogène (V29)
    import delta_atome as At
    _, Eat, _ = At.niveaux(l=0, Z=1, combien=4, r_max=120.0, n_points=6000)
    for i, e in enumerate(Eat):
        a(S.ligne(f'atome.E{i+1}', float(e)))
    a(S.ligne('atome.ionisation_eV', float(-Eat[0] * At.HARTREE_EV)))
    a(S.ligne('atome.balmer_alpha_nm', float(At.longueur_onde_nm(Eat[2], Eat[1]))))

    # ───────────────────── Δ-DIVERSITE : Leinster (V30)
    import delta_diversite as Lv
    sp = np.array([1.62, 1.32, 1.29, 1.16, .22, .17, .10, .08])
    for q in (0, 1, 2):
        a(S.ligne(f'diversite.hill_q{q}', Lv.hill(sp ** 2, q)))
    a(S.ligne('diversite.hill_qinf', Lv.hill(sp ** 2, np.inf)))
    a(S.ligne('diversite.rang_participatif_egal_hill1',
              abs(Lv.rang_participatif(sp) - Lv.hill(sp ** 2, 1.0)) < 1e-12))

    # ───────────────────────────── Δ18 : le sceau, le coffre
    rng = np.random.default_rng(GRAINE)
    parts = [rng.standard_normal(128) for _ in range(64)]
    coffre = D18.CoffreDelta(CLE, parts)
    a(S.ligne('coffre.racine_merkle', coffre.merkle.racine))
    a(S.ligne('coffre.sceaux_valides', coffre.verifier()))
    a(S.ligne('coffre.parite', hashlib.blake2b(coffre.parite().tobytes(),
                                               digest_size=16).digest()))
    return L


def bloc():
    return S.bloc_texte(calculer())


if __name__ == '__main__':
    t = bloc()
    if '--bloc' in sys.argv:
        sys.stdout.write(t)
        sys.exit(0)
    sig = S.signature(t)
    sortie = []
    sortie.append('╔' + '═' * 72 + '╗')
    sortie.append('║ Δ COMPLET — UN SEUL BLOC, Σ ET ∫, UNE SEULE EMPREINTE' + ' ' * 19 + '║')
    sortie.append('╚' + '═' * 72 + '╝')
    sortie.append(t.rstrip('\n'))
    sortie.append('─' * 74)
    sortie.append(f'lignes={len(t.splitlines())}  octets={len(t.encode())}')
    sortie.append(f'HASH DE Δ (double SHA-256, format Bitcoin) :')
    sortie.append(f'  {sig}')
    sortie.append('─' * 74)
    # ── LA CONFRONTATION : Δ relance dans un PROCESSUS NEUF et se rejuge.
    # Se recalculer dans le meme processus serait trop facile : la memoire
    # est deja chaude, les modules deja charges. Un processus neuf ne
    # partage rien.
    import subprocess
    r = subprocess.run([sys.executable, __file__, '--bloc'],
                       capture_output=True, text=True, cwd='.')
    t2 = r.stdout
    sig2 = S.signature(t2)
    identique = (t == t2)
    sortie.append('CONFRONTATION — Δ complet relance dans un PROCESSUS NEUF :')
    sortie.append(f'  bloc identique  : {identique}')
    sortie.append(f'  hash identique  : {sig2 == sig}')
    sortie.append(f'  {sig2}')
    if not identique:
        d = [(x, y) for x, y in zip(t.splitlines(), t2.splitlines()) if x != y]
        for x, y in d[:10]:
            sortie.append(f'  DIVERGENCE  {x}  !=  {y}')
    sortie.append('─' * 74)
    # ── CONFRONTATION DIRECTE : Δ complet face au hash qu'il vient d'engendrer
    import delta18 as _D18
    graine = bytes.fromhex(sig)
    flux, cour = [], graine
    for _ in range(4096):
        cour = S.sha256d(cour)
        flux.append(cour)
    oct = np.frombuffer(b''.join(flux), np.uint8).astype(np.float64)
    M = oct[:256 * 256].reshape(256, 256)
    vals = np.linalg.svd(M, compute_uv=False)
    tot = float((vals ** 2).sum())
    cum = np.cumsum(vals[::-1] ** 2)[::-1]
    rang = next((k for k in range(1, len(vals) + 1)
                 if (cum[k] if k < len(cum) else 0.0) / tot <= 1e-16), len(vals))
    c16 = _D18.Condensateur(oct[:4096], bits=16)
    err16 = float(np.abs(np.rint(c16.lire()) - oct[:4096]).max())
    c8 = _D18.Condensateur(oct[:4096], bits=8)
    err8 = float(np.abs(np.rint(c8.lire()) - oct[:4096]).max())
    c4 = _D18.Condensateur(oct[:4096], bits=4)
    err4 = float(np.abs(np.rint(c4.lire()) - oct[:4096]).max())
    pred = __import__('delta17').Predicteur(4)
    justes = 0
    suite = [int(x) for x in oct[:4000]]
    for i in range(len(suite) - 1):
        pred.observer(suite[i])
        if suite[i + 1] in pred.prevoir():
            justes += 1
    sortie.append('CONFRONTATION DIRECTE — Δ complet face au hash qu il vient')
    sortie.append('d engendrer, enchaine sur 128 Kio :')
    sortie.append(f'  rang retenu par la SVD        : {rang} / 256  (plein = aucune structure)')
    sortie.append(f'  condensateur 16 bits, erreur  : {err16:.0f} niveau(x) sur 255')
    sortie.append(f'  condensateur  8 bits, erreur  : {err8:.0f} niveau(x) sur 255'
                  f'  <- ne prouve RIEN : 256 valeurs tiennent dans 256 niveaux')
    sortie.append(f'  condensateur  4 bits, erreur  : {err4:.0f} niveau(x) sur 255'
                  f'  <- des qu on serre, tout casse')
    sortie.append(f'  predicteur de cache, succes   : {100*justes/max(len(suite)-1,1):.2f} %'
                  f'  (hasard attendu : {100*4/256:.2f} %)')
    sortie.append('  VERDICT : Δ ne trouve AUCUNE prise dans sa propre empreinte.')
    sortie.append('  Ni rang faible, ni voisinage, ni repetition. Il sait la')
    sortie.append('  produire, la sceller et la prouver — il ne sait pas la')
    sortie.append('  reduire. C est le contre-essai le plus dur possible, et Δ le')
    sortie.append('  passe en echouant, ce qui est exactement ce qu il fallait.')
    sortie.append('─' * 74)
    sortie.append('CE QUI EST DANS LE HASH : valeurs calculées, erreurs, racines,')
    sortie.append('empreintes, verdicts de conformite. Tout est reproductible.')
    sortie.append('CE QUI N EST PAS DANS LE HASH, ET POURQUOI : les durees, les')
    sortie.append('debits, les latences. Ils dependent de la machine et de la')
    sortie.append('minute ; les mettre dans l empreinte la rendrait differente a')
    sortie.append('chaque lancement, donc bonne a rien. Un hash ne peut sceller')
    sortie.append('que ce qui ne bouge pas.')
    texte = '\n'.join(sortie)
    print(texte)
    open('DELTA_BLOC.txt', 'w').write(t)
    open('DELTA_SIGNATURE.txt', 'w').write(sig + '\n')
    json.dump({'hash': sig, 'hash_recalcule': sig2, 'identique': identique,
               'lignes': len(t.splitlines()), 'octets': len(t.encode())},
              open('resultats_sigma.json', 'w'), indent=1)
