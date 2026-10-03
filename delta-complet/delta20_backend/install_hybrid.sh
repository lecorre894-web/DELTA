grep -q "compute_bool" delta_core.py || printf '\ndef _delta_compute_bool(s,A,B,C,mode="auto"):\n    from delta_hybrid import compute_bool;return compute_bool(A,B,C,mode)\nDeltaCore.compute_bool=_delta_compute_bool\n' >> delta_core.py
grep -q "libdelta_hybrid.so" .gitignore 2>/dev/null || echo "libdelta_hybrid.so" >> .gitignore
python3 delta_hybrid.py | tee delta_hybrid_result.txt
python3 -c "import numpy as np;from delta_core import DeltaCore as D;r=np.random.default_rng(1);A=r.integers(0,2**62,(4,4096),dtype=np.uint64);print('CORE_COMPUTE_BOOL=%s'%('OK' if np.array_equal(D.compute_bool(None,A,A,A[0]),D.compute_bool(None,A,A,A[0],'standard')) else 'FAIL'))"
echo "SECRETS=$(git diff --cached -U0 2>/dev/null | grep -ciE 'IQP_API_TOKEN=|crn:v1|KAGGLE_KEY=')"
