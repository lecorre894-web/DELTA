#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t v[3];
    uint8_t s[3];
} clause3_t;

/* ---------------------------------------------------------
   Validation scalaire d'une affectation
   --------------------------------------------------------- */
static inline int sat_one(uint64_t x,
                          const clause3_t *c,
                          size_t nc)
{
    for (size_t j=0; j<nc; ++j) {
        int ok = 0;

        for (int k=0; k<3; ++k) {
            uint64_t b = (x >> c[j].v[k]) & 1ULL;
            ok |= (b == c[j].s[k]);
        }

        if (!ok) return 0;
    }

    return 1;
}

/* ---------------------------------------------------------
   C scalaire exact
   --------------------------------------------------------- */
uint64_t delta_sat_scalar(
    unsigned nvars,
    const clause3_t *clauses,
    size_t nclauses,
    uint64_t *tested)
{
    uint64_t limit = 1ULL << nvars;
    *tested = 0;

    for (uint64_t x=0; x<limit; ++x) {
        ++*tested;

        if (sat_one(x, clauses, nclauses))
            return x;
    }

    return UINT64_MAX;
}

/* ---------------------------------------------------------
   AVX-512

   Une __m512i contient 8 lanes uint64.
   Chaque lane représente 64 affectations.

   => 8 x 64 = 512 affectations / bloc.

   Pour chaque variable, on construit son masque de vérité
   sur les 512 candidats du bloc.

   alive = ensemble des candidats encore possibles.
   --------------------------------------------------------- */

static inline __m512i variable_mask(
    uint64_t base,
    unsigned var)
{
    uint64_t m[8];

    for (int lane=0; lane<8; ++lane) {
        uint64_t bits = 0;
        uint64_t b = base + (uint64_t)lane*64ULL;

        for (unsigned i=0; i<64; ++i) {
            uint64_t x = b + i;

            if ((x >> var) & 1ULL)
                bits |= 1ULL << i;
        }

        m[lane] = bits;
    }

    return _mm512_loadu_si512((const void *)m);
}


uint64_t delta_sat_avx512(
    unsigned nvars,
    const clause3_t *clauses,
    size_t nclauses,
    uint64_t *tested)
{
    const uint64_t limit = 1ULL << nvars;

    const __m512i ALL =
        _mm512_set1_epi64((long long)-1);

    *tested = 0;

    for (uint64_t base=0; base<limit; base+=512ULL) {

        __m512i alive = ALL;

        for (size_t j=0; j<nclauses; ++j) {

            __m512i clause =
                _mm512_setzero_si512();

            for (int k=0; k<3; ++k) {

                __m512i vm =
                    variable_mask(base,
                                  clauses[j].v[k]);

                if (!clauses[j].s[k])
                    vm = _mm512_xor_si512(vm, ALL);

                clause =
                    _mm512_or_si512(clause, vm);
            }

            alive =
                _mm512_and_si512(alive, clause);

            __mmask8 nonzero =
                _mm512_cmpneq_epi64_mask(
                    alive,
                    _mm512_setzero_si512()
                );

            if (!nonzero)
                break;
        }

        uint64_t out[8];
        _mm512_storeu_si512((void *)out, alive);

        for (int lane=0; lane<8; ++lane) {

            if (out[lane]) {

                unsigned bit =
                    (unsigned)__builtin_ctzll(out[lane]);

                uint64_t candidate =
                    base +
                    (uint64_t)lane*64ULL +
                    bit;

                if (candidate < limit) {
                    *tested = candidate + 1;

                    /* vérification indépendante locale */
                    if (sat_one(candidate,
                                clauses,
                                nclauses))
                        return candidate;
                }
            }
        }

        *tested =
            (base + 512ULL < limit)
            ? base + 512ULL
            : limit;
    }

    return UINT64_MAX;
}
