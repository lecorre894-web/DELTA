#!/usr/bin/env python3
import random

N=36
M=162
SEED=20261005

r=random.Random(SEED)

# Utilisé uniquement par le générateur pour garantir SAT.
# Cette affectation n'est PAS écrite dans le CNF.
hidden=[r.getrandbits(1) for _ in range(N)]

clauses=[]

for _ in range(M):
    vs=r.sample(range(N),3)

    lits=[]

    for v in vs:
        positive=bool(r.getrandbits(1))
        lits.append((v,positive))

    if not any(
        hidden[v] == (1 if positive else 0)
        for v,positive in lits
    ):
        v,_=lits[0]
        lits[0]=(v,bool(hidden[v]))

    clauses.append(lits)

with open("challenge_36.cnf","w",encoding="ascii") as f:
    f.write("c DELTA external-format 3-SAT challenge\n")
    f.write(f"p cnf {N} {M}\n")

    for clause in clauses:
        out=[]

        for v,positive in clause:
            lit=v+1
            out.append(str(lit if positive else -lit))

        f.write(" ".join(out)+" 0\n")

print("CREATED=challenge_36.cnf")
print(f"VARIABLES={N}")
print(f"CLAUSES={M}")
print("HIDDEN_ASSIGNMENT_WRITTEN_TO_CNF=NO")
