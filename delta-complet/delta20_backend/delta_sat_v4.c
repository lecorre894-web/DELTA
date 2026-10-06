#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t v[3];
    uint8_t s[3];
} clause3_t;

/* ==========================================================
   Référence C scalaire
   ========================================================== */

static inline int sat_one(
    uint64_t x,
    const clause3_t *c,
    size_t nc)
{
    for (size_t j=0; j<nc; ++j) {
        int ok=0;

        for (int k=0;k<3;k++) {
            uint64_t b=(x >> c[j].v[k]) & 1ULL;
            ok |= (b == c[j].s[k]);
        }

        if (!ok) return 0;
    }

    return 1;
}

uint64_t delta_sat_scalar_v4(
    unsigned nvars,
    const clause3_t *clauses,
    size_t nc,
    uint64_t *tested)
{
    uint64_t limit=1ULL << nvars;
    *tested=0;

    for (uint64_t x=0;x<limit;x++) {
        ++*tested;

        if (sat_one(x,clauses,nc))
            return x;
    }

    return UINT64_MAX;
}

/* ==========================================================
   Générateur DIRECT des motifs d'une variable.

   Pas de boucle sur les 64 affectations.

   var 0 : 101010...
   var 1 : 11001100...
   var 2 : 11110000...
   ...

   Pour var >= 6, les blocs de 64 sont entièrement
   à zéro ou entièrement à un.
   ========================================================== */

static inline uint64_t pattern64(
    uint64_t start,
    unsigned var)
{
    static const uint64_t PAT[6] = {
        0xAAAAAAAAAAAAAAAAULL,
        0xCCCCCCCCCCCCCCCCULL,
        0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL,
        0xFFFF0000FFFF0000ULL,
        0xFFFFFFFF00000000ULL
    };

    if (var < 6) {
        /*
           start est multiple de 64 dans notre moteur.
           Le motif est donc aligné.
        */
        return PAT[var];
    }

    return ((start >> var) & 1ULL)
           ? UINT64_MAX
           : 0ULL;
}

/* ==========================================================
   Noyau X512 optimisé.

   8 lanes x 64 candidats = 512 affectations.

   Aucune allocation dynamique.
   Aucun tableau NumPy.
   Aucun masque construit bit par bit.
   ========================================================== */

static inline uint64_t solve_block512(
    uint64_t base,
    uint64_t limit,
    const clause3_t *clauses,
    size_t nc)
{
    uint64_t alive_lane[8] = {
        UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX,
        UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX
    };

    /*
       Le calcul logique par clause est préparé sous forme
       de 8 mots de 64 bits, puis traité en vecteur 512 bits.
    */

    __m512i alive =
        _mm512_set1_epi64((long long)-1);

    const __m512i ALL =
        _mm512_set1_epi64((long long)-1);

    const __m512i ZERO =
        _mm512_setzero_si512();

    for (size_t j=0;j<nc;j++) {

        uint64_t q[8];

        for (int lane=0;lane<8;lane++) {

            uint64_t start =
                base + (uint64_t)lane*64ULL;

            uint64_t clause_mask=0;

            for (int k=0;k<3;k++) {

                uint64_t m =
                    pattern64(start,
                              clauses[j].v[k]);

                if (!clauses[j].s[k])
                    m = ~m;

                clause_mask |= m;
            }

            q[lane]=clause_mask;
        }

        __m512i clause =
            _mm512_loadu_si512((const void *)q);

        alive =
            _mm512_and_si512(alive,clause);

        __mmask8 nz =
            _mm512_cmpneq_epi64_mask(
                alive,ZERO);

        if (!nz)
            return UINT64_MAX;
    }

    _mm512_storeu_si512(
        (void *)alive_lane,
        alive);

    for (int lane=0;lane<8;lane++) {

        uint64_t bits=alive_lane[lane];

        while (bits) {

            unsigned bit=
                (unsigned)__builtin_ctzll(bits);

            uint64_t x=
                base +
                (uint64_t)lane*64ULL +
                bit;

            if (x < limit &&
                sat_one(x,clauses,nc))
                return x;

            bits &= bits-1;
        }
    }

    return UINT64_MAX;
}

/* ==========================================================
   X512 pur : parcours séquentiel des blocs
   ========================================================== */

uint64_t delta_sat_x512_v4(
    unsigned nvars,
    const clause3_t *clauses,
    size_t nc,
    uint64_t *tested)
{
    uint64_t limit=1ULL << nvars;
    *tested=0;

    for (uint64_t base=0;
         base<limit;
         base+=512ULL) {

        uint64_t x=
            solve_block512(
                base,limit,clauses,nc);

        uint64_t end=base+512ULL;

        if (end>limit)
            end=limit;

        if (x != UINT64_MAX) {
            *tested=x+1;
            return x;
        }

        *tested=end;
    }

    return UINT64_MAX;
}

/* ==========================================================
   DELTA QPU LOGIQUE / COPROCESSEUR

   Le QPU logique découpe l'espace en grains.
   Chaque grain est exécuté par X512.

   Cette fonction est réellement différente de x512_v4 :
   elle constitue le chemin coprocesseur DELTA.

   Pas de cache disque.
   Pas d'état persistant.
   Working set borné.
   ========================================================== */

uint64_t delta_qpu_sat_v4(
    unsigned nvars,
    const clause3_t *clauses,
    size_t nc,
    uint64_t grain_blocks,
    uint64_t *tested,
    uint64_t *qpu_grains)
{
    uint64_t limit=1ULL << nvars;

    if (grain_blocks==0)
        grain_blocks=256;

    uint64_t grain_size=
        grain_blocks*512ULL;

    *tested=0;
    *qpu_grains=0;

    for (uint64_t gbase=0;
         gbase<limit;
         gbase+=grain_size) {

        ++*qpu_grains;

        uint64_t gend=
            gbase+grain_size;

        if (gend>limit)
            gend=limit;

        /*
           DELTA QPU logical scheduling:
           grain -> X512 blocks.
        */

        for (uint64_t base=gbase;
             base<gend;
             base+=512ULL) {

            uint64_t x=
                solve_block512(
                    base,
                    gend,
                    clauses,
                    nc);

            uint64_t end=
                base+512ULL;

            if (end>gend)
                end=gend;

            if (x != UINT64_MAX) {
                *tested=x+1;
                return x;
            }

            *tested=end;
        }
    }

    return UINT64_MAX;
}
