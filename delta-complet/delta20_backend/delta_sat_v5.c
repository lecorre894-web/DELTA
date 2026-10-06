#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t v[3];
    uint8_t s[3];
} clause3_t;

static inline int sat_one(
    uint64_t x,
    const clause3_t *c,
    size_t nc)
{
    for (size_t j=0;j<nc;j++) {
        int ok=0;
        for (int k=0;k<3;k++)
            ok |= (((x >> c[j].v[k]) & 1ULL) == c[j].s[k]);
        if (!ok) return 0;
    }
    return 1;
}

static inline uint64_t pattern64(
    uint64_t start,
    unsigned var)
{
    static const uint64_t P[6]={
        0xAAAAAAAAAAAAAAAAULL,
        0xCCCCCCCCCCCCCCCCULL,
        0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL,
        0xFFFF0000FFFF0000ULL,
        0xFFFFFFFF00000000ULL
    };

    if (var < 6)
        return P[var];

    return ((start >> var)&1ULL)
        ? UINT64_MAX : 0ULL;
}

/* =========================================================
   1. SCALAIRE — une affectation à la fois
   ========================================================= */

uint64_t sat_scalar(
    unsigned nv,
    const clause3_t *c,
    size_t nc,
    uint64_t *tested)
{
    uint64_t limit=1ULL<<nv;
    *tested=0;

    for (uint64_t x=0;x<limit;x++) {
        ++*tested;
        if (sat_one(x,c,nc))
            return x;
    }
    return UINT64_MAX;
}

/* =========================================================
   2. BIT64 — 64 affectations / uint64_t
   PAS d'AVX-512
   ========================================================= */

uint64_t sat_bit64(
    unsigned nv,
    const clause3_t *c,
    size_t nc,
    uint64_t *tested)
{
    uint64_t limit=1ULL<<nv;
    *tested=0;

    for (uint64_t base=0;base<limit;base+=64ULL) {

        uint64_t alive=UINT64_MAX;

        for (size_t j=0;j<nc;j++) {

            uint64_t cm=0;

            for (int k=0;k<3;k++) {
                uint64_t m=
                    pattern64(base,c[j].v[k]);

                if (!c[j].s[k])
                    m=~m;

                cm |= m;
            }

            alive &= cm;

            if (!alive)
                break;
        }

        while (alive) {

            unsigned bit=
                (unsigned)__builtin_ctzll(alive);

            uint64_t x=base+bit;

            if (x<limit && sat_one(x,c,nc)) {
                *tested=x+1;
                return x;
            }

            alive &= alive-1;
        }

        uint64_t e=base+64ULL;
        *tested=(e<limit)?e:limit;
    }

    return UINT64_MAX;
}

/* =========================================================
   3. X512 — 512 affectations
   ========================================================= */

static inline uint64_t block512(
    uint64_t base,
    uint64_t limit,
    const clause3_t *c,
    size_t nc)
{
    const __m512i ZERO=_mm512_setzero_si512();
    __m512i alive=_mm512_set1_epi64(-1LL);

    for (size_t j=0;j<nc;j++) {

        uint64_t q[8];

        for (int lane=0;lane<8;lane++) {

            uint64_t start=
                base+(uint64_t)lane*64ULL;

            uint64_t cm=0;

            for (int k=0;k<3;k++) {

                uint64_t m=
                    pattern64(start,c[j].v[k]);

                if (!c[j].s[k])
                    m=~m;

                cm |= m;
            }

            q[lane]=cm;
        }

        __m512i clause=
            _mm512_loadu_si512((const void*)q);

        alive=_mm512_and_si512(alive,clause);

        if (!_mm512_cmpneq_epi64_mask(alive,ZERO))
            return UINT64_MAX;
    }

    uint64_t out[8];
    _mm512_storeu_si512(out,alive);

    for (int lane=0;lane<8;lane++) {

        uint64_t bits=out[lane];

        while (bits) {

            unsigned b=
                (unsigned)__builtin_ctzll(bits);

            uint64_t x=
                base+(uint64_t)lane*64ULL+b;

            if (x<limit && sat_one(x,c,nc))
                return x;

            bits &= bits-1;
        }
    }

    return UINT64_MAX;
}

uint64_t sat_x512(
    unsigned nv,
    const clause3_t *c,
    size_t nc,
    uint64_t *tested)
{
    uint64_t limit=1ULL<<nv;
    *tested=0;

    for (uint64_t base=0;base<limit;base+=512ULL) {

        uint64_t x=block512(base,limit,c,nc);

        if (x!=UINT64_MAX) {
            *tested=x+1;
            return x;
        }

        uint64_t e=base+512ULL;
        *tested=(e<limit)?e:limit;
    }

    return UINT64_MAX;
}

/* =========================================================
   4. DELTA QPU LOGIQUE → X512
   ========================================================= */

uint64_t sat_qpu_x512(
    unsigned nv,
    const clause3_t *c,
    size_t nc,
    uint64_t grain_blocks,
    uint64_t *tested,
    uint64_t *grains)
{
    uint64_t limit=1ULL<<nv;

    if (!grain_blocks)
        grain_blocks=256;

    uint64_t grain=grain_blocks*512ULL;

    *tested=0;
    *grains=0;

    for (uint64_t gb=0;gb<limit;gb+=grain) {

        ++*grains;

        uint64_t ge=gb+grain;
        if (ge>limit) ge=limit;

        for (uint64_t base=gb;base<ge;base+=512ULL) {

            uint64_t x=block512(base,ge,c,nc);

            if (x!=UINT64_MAX) {
                *tested=x+1;
                return x;
            }

            uint64_t e=base+512ULL;
            if (e>ge) e=ge;
            *tested=e;
        }
    }

    return UINT64_MAX;
}
