#!/usr/bin/env python3
"""DUEL Δ V0.6 contre QPU simulé — arbitre : la vérité exacte 2^n.

QPU simulé : processeur supraconducteur réaliste (ordre de grandeur 2025-2026)
  - erreur de porte 2 qubits : p2 = 0,5 % (dépolarisation : erreur de Pauli aléatoire)
  - simulé par TRAJECTOIRES quantiques : K tirages d'états purs bruités ;
    fidélité du QPU = moyenne des |<ψ_exact|ψ_bruité>|²
  - honnêteté : ce QPU « en simulation » est lui-même un calcul classique 2^n.
    Son temps de simulation ne dit rien d'un vrai QPU ; on donne à part
    un temps physique estimé (≈ 60 ns par couche de portes, lecture exclue).

Δ : moteur Δ2 (liens χ, lots, complex64) — fidélité mesurée contre l'exact.
"""
import sys, time, math, json
import numpy as np
import delta2 as D2

I2 = np.eye(2, dtype=np.complex128)
PAULI = [I2, np.array([[0, 1], [1, 0]], np.complex128),
         np.array([[0, -1j], [1j, 0]], np.complex128), np.diag([1, -1]).astype(np.complex128)]
H = np.array([[1, 1], [1, -1]], np.complex128) / math.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], np.complex128)
ID4 = np.eye(4, dtype=np.complex128)


def haar4(rng):
    z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


# un circuit = liste de couches ; couche = (parité, {lien i: porte 4x4}) ; lien absent = identité
def circuit_faible(n, depth, seed, theta):
    rng = np.random.default_rng(seed)
    C = []
    for d in range(depth):
        a1 = rng.uniform(-theta, theta, n)
        par = d % 2
        bonds = list(range(par, n - 1, 2))
        a2 = rng.uniform(-theta, theta, len(bonds))
        g = {}
        for k, i in enumerate(bonds):
            ph = np.exp(-0.5j * a2[k])
            zz = np.diag([ph, ph.conj(), ph.conj(), ph])
            rxa = D2.rx_batch(np.array([a1[i]]))[0].astype(np.complex128)
            rxb = D2.rx_batch(np.array([a1[i + 1]]))[0].astype(np.complex128)
            g[i] = zz @ np.kron(rxa, rxb)
        # sites non couverts par un lien : on rattache leur rx au lien voisin suivant
        C.append((par, g, {q: D2.rx_batch(np.array([a1[q]]))[0].astype(np.complex128)
                           for q in range(n) if not any(q in (i, i + 1) for i in bonds)}))
    return C


def circuit_ghz(n):
    C = [(0, {0: CNOT @ np.kron(H, I2)}, {})]
    for i in range(1, n - 1):
        C.append((i % 2, {i: CNOT}, {}))
    return C


def circuit_monstre(n, depth, seed):
    """Échantillonnage de circuit aléatoire (style « suprématie quantique ») :
    briques de portes 2 qubits tirées au hasard selon la mesure de Haar."""
    rng = np.random.default_rng(seed)
    return [(d % 2, {i: haar4(rng) for i in range(d % 2, n - 1, 2)}, {}) for d in range(depth)]


def nb_portes2(C):
    return sum(len(g) for _, g, _ in C)


# ------------------------------------------------------------- vérité exacte
def exact(n, C, bruit=0.0, rng=None):
    t = np.zeros((2,) * n, np.complex128); t[(0,) * n] = 1
    for par, g, g1 in C:
        for q, U in g1.items():
            t = np.moveaxis(np.tensordot(U, t, axes=([1], [q])), 0, q)
        for i, U in g.items():
            t = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [i, i + 1]))
            t = np.moveaxis(t, [0, 1], [i, i + 1])
            if bruit > 0 and rng.random() < bruit:
                a, b = 0, 0
                while a == 0 and b == 0:
                    a, b = rng.integers(0, 4), rng.integers(0, 4)
                P = np.kron(PAULI[a], PAULI[b]).reshape(2, 2, 2, 2)
                t = np.tensordot(P, t, axes=([2, 3], [i, i + 1]))
                t = np.moveaxis(t, [0, 1], [i, i + 1])
    return t.reshape(-1)


def qpu(n, C, ref, p2=0.005, K=40, seed=1):
    rng = np.random.default_rng(seed)
    F = [abs(np.vdot(ref, exact(n, C, p2, rng))) ** 2 for _ in range(K)]
    return float(np.mean(F)), float(np.std(F) / math.sqrt(K))


def delta(n, C, chi):
    s = D2.Delta2(n, chi)
    for par, g, g1 in C:
        if g1:
            gates1 = np.repeat(np.eye(2, dtype=D2.CT)[None], n, 0)
            for q, U in g1.items():
                gates1[q] = U.astype(D2.CT)
            s.layer1(gates1)
        nb = len(range(par, n - 1, 2))
        G = np.repeat(ID4.astype(D2.CT)[None], nb, 0)
        for k, i in enumerate(range(par, n - 1, 2)):
            if i in g:
                G[k] = g[i].astype(D2.CT)
        s.layer2(par, G.reshape(nb, 2, 2, 2, 2))
    return s


def epreuve(nom, n, C, chis, K=40):
    print(f'\n━━━ {nom} ━━━  n={n}, {len(C)} couches, {nb_portes2(C)} portes à 2 qubits')
    t0 = time.time(); ref = exact(n, C); t_ref = time.time() - t0
    t0 = time.time(); Fq, eq = qpu(n, C, ref, K=K); t_q = time.time() - t0
    t_phys = len(C) * 60e-9
    print(f'QPU simulé (p2=0,5 %) : fidélité = {Fq:.4f} ± {eq:.4f}   '
          f'[attendu ≈ (1-p2)^portes = {(1-0.005)**nb_portes2(C):.4f}]   '
          f'temps physique estimé ≈ {t_phys*1e6:.2f} µs par tir')
    res = dict(epreuve=nom, n=n, couches=len(C), portes2=nb_portes2(C), qpu_fid=Fq, qpu_err=eq, qpu_t_phys=t_phys, delta=[])
    for chi in chis:
        t0 = time.time(); s = delta(n, C, chi); dt = time.time() - t0
        v = s.dense(); F = abs(np.vdot(ref, v)) ** 2 / np.vdot(v, v).real
        res['delta'].append(dict(chi=chi, fid=float(F), fid_est=s.fid, t=dt, mem=s.memory()))
        print(f'Δ χ={chi:4d}            : fidélité = {F:.6f}   estimée = {s.fid:.6f}   '
              f'temps = {dt*1000:8.1f} ms   mémoire = {s.memory()/1024:8.1f} Kio')
    return res


if __name__ == '__main__':
    n = 16
    R = []
    R.append(epreuve('ÉPREUVE 1 — intrication faible (θ=0,6)', n, circuit_faible(n, 10, 5, 0.6), [2, 4, 8]))
    R.append(epreuve('ÉPREUVE 2 — GHZ, intrication totale', n, circuit_ghz(n), [1, 2]))
    for depth in [8, 16, 24]:
        R.append(epreuve(f'ÉPREUVE 3 — LE MONSTRE : circuit aléatoire de Haar, profondeur {depth}',
                         n, circuit_monstre(n, depth, 7), [4, 16, 64, 256]))
    json.dump(R, open('resultats_duel.json', 'w'), indent=1, default=float)
