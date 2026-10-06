#!/usr/bin/env python3
"""
DELTA X512 DUAL
Deux processeurs X512 logiciels indépendants :
  PROCESSOR_A = workers A
  PROCESSOR_B = workers B

Le X512 original n'est pas modifié.
Même charge totale, partitionnée A/B, fusion ordonnée et checksum final.
"""

import os
import time
import hashlib
from delta_x512 import DeltaX512

class DeltaX512Dual:
    def __init__(self, workers_a=16, workers_b=16, prog="micro"):
        self.wa = workers_a
        self.wb = workers_b
        self.prog = prog

        self.A = DeltaX512(workers_a, prog)
        self.B = DeltaX512(workers_b, prog)

    def compute(self, reqs):
        # Partition déterministe alternée.
        # On garde la position originale pour la fusion finale.
        ra = []
        rb = []

        for pos, req in enumerate(reqs):
            if pos & 1:
                rb.append((pos, req))
            else:
                ra.append((pos, req))

        import threading

        out_a = None
        out_b = None
        err_a = None
        err_b = None

        def run_a():
            nonlocal out_a, err_a
            try:
                out_a = self.A.compute([x[1] for x in ra])
            except Exception as e:
                err_a = e

        def run_b():
            nonlocal out_b, err_b
            try:
                out_b = self.B.compute([x[1] for x in rb])
            except Exception as e:
                err_b = e

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)

        t0 = time.perf_counter()

        ta.start()
        tb.start()

        ta.join()
        tb.join()

        wall = time.perf_counter() - t0

        if err_a:
            raise err_a
        if err_b:
            raise err_b

        resa, calca, nia, tca = out_a
        resb, calcb, nib, tcb = out_b

        merged = [None] * len(reqs)

        for (pos, _), value in zip(ra, resa):
            merged[pos] = value

        for (pos, _), value in zip(rb, resb):
            merged[pos] = value

        return (
            merged,
            calca + calcb,
            nia + nib,
            max(tca, tcb),
            wall
        )

    def close(self):
        self.A.close()
        self.B.close()


def signature(values):
    return hashlib.sha256(
        repr(values).encode()
    ).hexdigest()[:16]
