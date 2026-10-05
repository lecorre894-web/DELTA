#!/usr/bin/env python3
import atexit
import threading
import time
import delta_hardware
from collections import OrderedDict

# DELTA hardware profile: détection ponctuelle, RAM uniquement
_HARDWARE_PROFILE = delta_hardware.detect()
_DELTA_PROFILE = _HARDWARE_PROFILE['profile']


_lock = threading.Lock()
_engine = None
_build_ms = None
_requests = 0


def _build():
    global _engine, _build_ms

    if _engine is not None:
        return _engine

    with _lock:
        if _engine is None:
            t0 = time.perf_counter_ns()

            import delta_x512
            _engine = delta_x512.DeltaX512()

            _build_ms = (time.perf_counter_ns() - t0) / 1e6

    return _engine


def engine():
    return _build()


def compute(reqs):
    global _requests

    e = _build()

    t0 = time.perf_counter_ns()
    result = e.compute(reqs)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

    _requests += 1

    return result, {
        "request": _requests,
        "compute_ms": elapsed_ms,
        "engine_build_ms": _build_ms,
        "engine_resident": True,
        "workers_persistent": True,
    }


def status():
    return {
        "engine_loaded": _engine is not None,
        "engine_build_ms": _build_ms,
        "requests": _requests,
        "policy": "ONE_RESIDENT_X512_ENGINE",
    }


def close():
    global _engine

    with _lock:
        if _engine is not None:
            try:
                _engine.close()
            finally:
                _engine = None


atexit.register(close)


# DELTA bounded RAM memoization
# Aucun disque, aucun daemon, maximum 32 ensembles de résultats.
_MEMO_MAX = 32
_MEMO = OrderedDict()
_MEMO_HITS = 0
_MEMO_MISSES = 0

def _memo_key(reqs):
    """Clé immuable pour une liste de requêtes X512."""
    return tuple(tuple(r) for r in reqs)

def memo_compute(reqs):
    global _MEMO_HITS, _MEMO_MISSES

    key = _memo_key(reqs)

    if key in _MEMO:
        _MEMO_HITS += 1
        functional = _MEMO.pop(key)
        _MEMO[key] = functional

        telemetry = {
            "cache_hit": True,
            "compute_executed": False,
            "cache_entries": len(_MEMO),
            "cache_limit": _MEMO_MAX,
        }
        return functional, telemetry

    _MEMO_MISSES += 1
    result = compute(reqs)

    # On mémorise seulement le résultat fonctionnel.
    functional = result[0] if isinstance(result, tuple) else result

    _MEMO[key] = functional
    while len(_MEMO) > _MEMO_MAX:
        _MEMO.popitem(last=False)

    telemetry = {
        "cache_hit": False,
        "compute_executed": True,
        "cache_entries": len(_MEMO),
        "cache_limit": _MEMO_MAX,
    }
    return functional, telemetry

def memo_status():
    return {
        "entries": len(_MEMO),
        "limit": _MEMO_MAX,
        "hits": _MEMO_HITS,
        "misses": _MEMO_MISSES,
        "persistent": False,
        "storage": "RAM_ONLY",
    }

def memo_clear():
    _MEMO.clear()

