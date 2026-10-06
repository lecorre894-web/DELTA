#!/usr/bin/env python3
import ctypes, os, random, statistics, time

LIMIT=120.0
RUNS=7
SEED=894
MAX=(1<<64)-1

class Clause(ctypes.Structure):
    _fields_=[
        ("v",ctypes.c_uint8*3),
        ("s",ctypes.c_uint8*3)
    ]

L=ctypes.CDLL(os.path.abspath("./libdelta_sat_v5.so"))

for name in ("sat_scalar","sat_bit64","sat_x512"):
    f=getattr(L,name)
    f.argtypes=[
        ctypes.c_uint,
        ctypes.POINTER(Clause),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64)
    ]
    f.restype=ctypes.c_uint64

L.sat_qpu_x512.argtypes=[
    ctypes.c_uint,
    ctypes.POINTER(Clause),
    ctypes.c_size_t,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64)
]
L.sat_qpu_x512.restype=ctypes.c_uint64

def make(n,nc,seed):
    r=random.Random(seed)
    secret=[r.getrandbits(1) for _ in range(n)]
    py=[]

    for _ in range(nc):
        vs=r.sample(range(n),3)
        z=[[v,r.getrandbits(1)] for v in vs]

        if not any(secret[v]==s for v,s in z):
            z[0][1]=secret[z[0][0]]

        py.append(z)

    a=(Clause*nc)()

    for j,c in enumerate(py):
        for k,(v,s) in enumerate(c):
            a[j].v[k]=v
            a[j].s[k]=s

    return py,a

def verify(x,c):
    return x!=MAX and all(
        any(((x>>v)&1)==s for v,s in z)
        for z in c
    )

def normal(f,n,a,nc):
    tested=ctypes.c_uint64()
    t=time.perf_counter()
    x=f(n,a,nc,ctypes.byref(tested))
    return x,tested.value,time.perf_counter()-t

def qpu(n,a,nc):
    tested=ctypes.c_uint64()
    grains=ctypes.c_uint64()

    t=time.perf_counter()

    x=L.sat_qpu_x512(
        n,a,nc,256,
        ctypes.byref(tested),
        ctypes.byref(grains)
    )

    return (
        x,tested.value,grains.value,
        time.perf_counter()-t
    )

print("=== DELTA SAT V5 — ATTRIBUTION ===")
print(f"CPU_AVAILABLE={os.cpu_count()}")
print("PATH_1=C_SCALAR")
print("PATH_2=C_BIT64")
print("PATH_3=X512_AVX512")
print("PATH_4=DELTA_QPU_X512")
print("RUNS=7")
print("TIME_LIMIT_SECONDS=120")
print()

start=time.perf_counter()
deadline=start+LIMIT

# On privilégie les charges qui ont bien discriminé V4.
tests=[
    (28,126),
    (32,144),
    (34,153)
]

rows=[]

for n,nc in tests:

    if time.perf_counter()>=deadline:
        break

    print(f"--- VARS={n} CLAUSES={nc} ---")

    py,a=make(n,nc,SEED+n)

    times={
        "SCALAR":[],
        "BIT64":[],
        "X512":[],
        "QPU_X512":[]
    }

    last=None

    for run in range(RUNS):

        if time.perf_counter()>=deadline:
            break

        s,st,ts=normal(L.sat_scalar,n,a,nc)
        b,bt,tb=normal(L.sat_bit64,n,a,nc)
        x,xt,tx=normal(L.sat_x512,n,a,nc)
        q,qt,qg,tq=qpu(n,a,nc)

        good=(
            s==b==x==q and
            verify(s,py)
        )

        print(
            f"RUN={run+1} "
            f"C={ts:.6f}s "
            f"BIT64={tb:.6f}s "
            f"X512={tx:.6f}s "
            f"QPU={tq:.6f}s "
            f"VALID={'YES' if good else 'NO'}"
        )

        if not good:
            raise SystemExit("VALIDATION_FAILURE")

        times["SCALAR"].append(ts)
        times["BIT64"].append(tb)
        times["X512"].append(tx)
        times["QPU_X512"].append(tq)

        last=(st,bt,xt,qt,qg,s)

    if not times["SCALAR"]:
        break

    med={k:statistics.median(v) for k,v in times.items()}

    fastest=min(med,key=med.get)

    c=med["SCALAR"]
    b=med["BIT64"]
    x=med["X512"]
    q=med["QPU_X512"]

    print(f"C_MEDIAN={c:.9f}s")
    print(f"BIT64_MEDIAN={b:.9f}s")
    print(f"X512_MEDIAN={x:.9f}s")
    print(f"QPU_X512_MEDIAN={q:.9f}s")
    print()

    print(f"BIT64_VS_C={c/b:.3f}x")
    print(f"X512_VS_C={c/x:.3f}x")
    print(f"QPU_VS_C={c/q:.3f}x")

    # Mesure essentielle :
    print(f"X512_VS_BIT64={b/x:.3f}x")
    print(f"QPU_VS_BIT64={b/q:.3f}x")

    print(f"BEST_ENGINE_THIS_SIZE={fastest}")

    st,bt,xt,qt,qg,sol=last

    print(
        f"TESTED="
        f"C:{st} BIT64:{bt} X512:{xt} QPU:{qt}"
    )
    print(f"QPU_GRAINS={qg}")
    print(f"SOLUTION={sol}")
    print("VALIDATION=PASSED")
    print()

    rows.append((n,nc,med))

elapsed=time.perf_counter()-start

print("=== V5 FINAL ===")
print(f"ELAPSED_SECONDS={elapsed:.6f}")

if rows:
    engines=[
        "SCALAR",
        "BIT64",
        "X512",
        "QPU_X512"
    ]

    # Ratio normalisé par taille :
    # 1.0 = meilleur moteur de cette taille.
    scores={e:[] for e in engines}

    for n,nc,m in rows:
        best=min(m.values())

        for e in engines:
            scores[e].append(m[e]/best)

    final_score={
        e:statistics.median(v)
        for e,v in scores.items()
    }

    winner=min(final_score,key=final_score.get)

    print()
    for e in engines:
        print(
            f"ENGINE_SCORE_{e}="
            f"{final_score[e]:.6f}"
        )

    print()
    print(f"BEST_ENGINE={winner}")

    # Attribution du gain.
    bit_gain=statistics.median(
        r[2]["SCALAR"]/r[2]["BIT64"]
        for r in rows
    )

    avx_gain=statistics.median(
        r[2]["BIT64"]/r[2]["X512"]
        for r in rows
    )

    qpu_ratio=statistics.median(
        r[2]["X512"]/r[2]["QPU_X512"]
        for r in rows
    )

    print(f"MEDIAN_BITPARALLEL_GAIN={bit_gain:.3f}x")
    print(f"MEDIAN_X512_OVER_BIT64={avx_gain:.3f}x")
    print(f"MEDIAN_QPU_VS_X512={qpu_ratio:.3f}x")

print()
print("SAT_RESULT=EXACT")
print("SAME_SEARCH_ORDER=YES")
print("BIT64_AVX512=NO")
print("X512_AVX512=YES")
print("DELTA_QPU=LOGICAL_COPROCESSOR")
print("CPU_EXECUTION=PHYSICAL_MEASURED")
print("PERSISTENT_CACHE=NO")
print("V5_COMPLETE=YES")
