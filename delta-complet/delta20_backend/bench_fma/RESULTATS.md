# Bench FMA (kernel OpenCL de René) sur Xeon 8573C, 2 oct 2026
OpenCL PoCL : 143578 ms, 11.69 GF
AVX-512 direct -O2 : 35422 ms, 47.36 GF
x8 chaines -O2 : 27789 ms, 60.37 GF
x8 chaines -O3 -march=native : 9291 ms, 180.58 GF
x16 chaines -O3 -march=native : 9182 ms, 182.72 GF = PLAFOND (2 FMA x 16 x 2 x ~2.85 GHz = 182.4 GF)
Checksum identique partout : 1260.948853. Gain x15.6 a calcul identique.
Non retenu : forme affine 1.7 ms (checksum 1261.685303, calcul different).

## Socle Ryzen emule (KVM, -cpu host, 2 vCPU), 2 oct 2026
x16 chaines : 9607 ms, 174.63 GF (+4.4% vs Xeon natif 9182 ms) | Checksum 1260.948853
Dedup DELTA : 35.06 ms (Xeon natif 12.31 ms) | Checksum 1260.948853
NOTE : dedup = temps de RESULTAT (calcul identique fait une fois), pas une puissance ; ne jamais le citer en GFLOPS.
