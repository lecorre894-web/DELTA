# DELTA V13 — Horner V11 + IVF Vector Search (mesuré)

Mise à jour du 30 septembre 2026. Complète `README_DELTA_V10_VIRTUAL_SILICON.md`.

Règle de ce document : **tout chiffre ci-dessous est mesuré** sur la plateforme
indiquée, avec checksum ou validation. Aucune projection n'est mélangée aux mesures.

---

## 1. Plateforme

- Motorola moto g75 5G — Qualcomm Snapdragon 6 Gen 3
  (4 × Cortex-A78 2,4 GHz + 4 × Cortex-A55 1,8 GHz), GPU Adreno 710
- Android / Termux, AArch64, clang, pthread, OpenCL
- `long double` = 113 bits (quad émulé logiciel)

## 2. Compilation — changement important

Le lanceur utilise désormais :

    -mcpu=cortex-a78

au lieu de `-march=native`. Sous Termux, `-march=native` ne détecte pas
les extensions FP16 et produit scalaire entier (SDOT/UDOT) du Cortex-A78.

Effet mesuré :

- FP16 vectoriel : ×0,18 → ×3,56 (vs float32)
- INT8 produit scalaire : ×3,98 → ×6,86 (vs float32)
- Horner double (V9) : neutre (46,5 vs 48,4 GF/s, dans le bruit de run)

Lanceur sauvegardé : `delta_v10_virtual_silicon.sh.bak`

---

## 3. V11 — Horner degré 127 : ×6,1 réel

### Diagnostic

Horner est une chaîne de dépendances : une seule évaluation par thread
est bornée par la latence FMA (~4 cycles), pas par le débit.

### Correctifs appliqués dans `v8_multi_worker`

1. **ILP16** : 16 évaluations indépendantes entrelacées par thread
2. **Blocs dynamiques** : compteur atomique au lieu du partage statique
   (les A78 absorbent plus de travail que les A55)
3. **Table de x** : `(i%10000)/10000.0*0.5` précalculé une fois
   (même expression → valeurs identiques au bit) ; supprime une division
   double par évaluation

### Résultats V9_FINAL (1e9 évaluations, 254e9 FLOPs, guard 4,38e-16)

| Étape | GFLOP/s | Facteur |
|---|---|---|
| V9 d'origine | 7,96 | ×1,0 |
| + ILP16 + blocs dynamiques | 27,90 | ×3,5 |
| + table de x | 48,42 | ×6,1 |
| Run intégré V13 (-mcpu) | 46,50 | ×5,8 |

L'autotuner retient 7 workers (le 8e coûte plus qu'il ne rapporte).
Référence scalaire mono-thread (V7) : 1,34 GF/s. V9 à 1 worker : 9,16 GF/s.

### Bench isolé `horner_ilp.c` / `horner_dyn.c` (checksum MATCH au bit)

| Mode | GFLOP/s |
|---|---|
| M=1 statique | 7,78 |
| M=8 statique | 44,26 |
| M=16 statique | 44,97 |
| M=16 dynamique | 50,82 |

---

## 4. Précision — oracle MPFR

### Erreur réelle du double (oracle 2048 bits, 10 000 x distincts)

- MAX_REL = 8,54e-17 (0,77 ulp) — le double est quasi exact sur ce polynôme
- float32 : MAX_REL = 4,97e-8 (0,83 ulp24) pour seulement ×1,08 de vitesse
  → **float32 rejeté**

### Coût par bit (`bitsweep.c`, 1 cœur, oracle 8192 bits)

| Précision | MFLOPS | Ralentissement vs double HW | Bits corrects |
|---|---|---|---|
| HW double 53 | 9 203 | ×1 | 53,5 |
| HW long double 113 | 45,2 | ×204 | 113,3 |
| MPFR 53 | 48,1 | ×190 | 53,5 |
| MPFR 113 | 25,7 | ×356 | 113,4 |
| MPFR 256 | 20,7 | ×442 | 256,4 |
| MPFR 512 | 13,4 | ×683 | 512,5 |
| MPFR 1024 | 5,57 | ×1 643 | 1024,5 |
| MPFR 2048 | 2,08 | ×4 394 | 2048,5 |
| MPFR 4096 | 0,707 | ×12 939 | 4096,4 |

Conclusions :

- falaise matériel/logiciel : ≥ ×177 dès qu'on quitte le FPU, même à précision égale
- chaque précision livre ses bits (+0,4) : polynôme bien conditionné
- **le double reste le point d'efficacité absolu pour DELTA**

---

## 5. Recherche vectorielle — chemin de validation

### Données synthétiques (1 cœur, N=100 000, D=128)

| Méthode | Accélération vs float32 | Recall@10 |
|---|---|---|
| FP16 NEON | ×3,56 | 1,000 |
| INT8 plein | ×6,86 | 0,972 |
| INT8 → reclassement float32 (C=20) | ×7,39 | 1,000 |
| 2 bits seul | ×8,22 | 0,244 → rejeté |

### Données réelles SIFT (INRIA/IRISA, 128 dimensions, métrique L2)

SIFT tient sans perte en uint8 : le scan uint8 est exact.

- SimHash multi-tables : **perd** contre la force brute uint8
  (meilleur point ×5,45 à 0,972 contre ×7,49 à 1,000) ;
  tiroirs déséquilibrés + accès mémoire dispersé
- IVF k-means : **gagne**

Passage au million (`sift_bridge.c`, 7 cœurs, 1 000 requêtes, 2 048 listes) :

| Méthode | Temps 1 000 requêtes | vs uint8 MT | Recall@10 |
|---|---|---|---|
| float32 MT (vérité terrain) | 35,1 s | — | 1,000 |
| uint8 plein MT | 8,61 s | ×1 | 1,000 |
| IVF nprobe=64 | 0,354 s | ×24,3 | 0,988 |
| IVF nprobe=128 | 0,676 s | ×12,7 | 0,998 |

Soit environ **0,35 ms par requête sur 1 000 000 de vecteurs**.
Construction k-means : 51 s, une seule fois.

### Goulot identifié : bande passante RAM

- uint8 vs float32 plein = ×4,08, exactement le rapport des octets lus (128 vs 512)
- multithread IVF : gain qui fond avec nprobe (×3,6 à nprobe=1, ×1,7 à nprobe=64)
- estimation : ~13 Go/s lus à nprobe=64, proche du plafond LPDDR4X

---

## 6. V13 — section intégrée dans DELTA

Section `§DELTA V13 IVF VECTOR SEARCH`, exécutée avant le moteur V12.1.

Configuration : N=200 000, Q=1 000, 1 024 listes, 7 threads, C=20, L2.

Run documenté :

    V13_FULL_FLOAT32_MT WALL=6.801 s (GROUND_TRUTH)
    V13_FULL_UINT8_MT   WALL_MED=1.532 s SPEEDUP=x4.44 RECALL@10=1.000
    V13_KMEANS          BUILD_WALL=9.49 s IMBALANCE=x4.38
    V13_IVF NPROBE=16   SCANNED=1.94% VS_U8FULL=x39.70 RECALL@10=0.894
    V13_IVF NPROBE=32   SCANNED=3.77% VS_U8FULL=x21.69 RECALL@10=0.963
    V13_IVF NPROBE=64   SCANNED=7.30% VS_U8FULL=x11.29 RECALL@10=0.992
    V13_VALIDATION=OK (RECALL@10 NPROBE64 >= 0.98)
    DELTA V12.1 FULL ENGINE VALIDATION: OK

Règles :

- validation propre (recall ≥ 0,98 à nprobe=64)
- n'altère jamais la validation V12.1 canonique
- `V13_SKIPPED` si `sift/sift_base.fvecs` ou `sift/sift_query.fvecs` absent

Données (≈160 Mo archive, ≈550 Mo extrait) :

    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz

---

## 7. Requalification de V10

- **Virtual Multiplier** : multiplication du GFLOP/s mesuré par des constantes
  (32, π, √10, 1000, 100 000, lanes). Aucune information mesurée.
  Statut : projection arithmétique, **à ne jamais citer comme performance**.
- **Virtual Silicon** : mesure réelle = débit de hachage 64 bits
  (~118,7 M pulses/s). Aucun état de cellule n'est stocké ni propagé.
  1,25e23 = taille du codomaine d'adressage, pas une capacité.

Le facteur réel à citer à la place du Virtual Multiplier est :

    V11 : ×6,1 mesuré (7,96 → 48,42 GF/s, checksum/guard OK)

---

## 8. Relancer

    cd ~/DELTA_GITHUB/delta-complet/delta20_backend
    ./delta_v10_virtual_silicon.sh

Filtre des lignes clés :

    ./delta_v10_virtual_silicon.sh 2>&1 | grep -E "V13_|V9_FINAL|ENGINE VALIDATION"

## 9. Sauvegardes

    .c.bak_v10   avant V11
    .c.bak_v11   avant V11b (table de x)
    .c.bak_v12   avant V13
    delta_v10_virtual_silicon.sh.bak   lanceur avant -mcpu

## 10. Réserves

- un seul appareil, une seule session de mesure ; bruit inter-run ~5–10 %
- recherche vectorielle : 100 à 1 000 requêtes SIFT sur 10 000 disponibles
- données synthétiques = bancs d'essai ; seules les lignes SIFT valent données réelles
- pic théorique FP64 (~100 GF/s) = estimation, non mesurée
