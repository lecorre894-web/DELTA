#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t v[3];
    uint8_t s[3];
} clause3_t;

static inline uint64_t pattern64(uint64_t start, unsigned var)
{
    static const uint64_t P[6] = {
        0xAAAAAAAAAAAAAAAAULL,
        0xCCCCCCCCCCCCCCCCULL,
        0xF0F0F0F0F0F0F0F0ULL,
        0xFF00FF00FF00FF00ULL,
        0xFFFF0000FFFF0000ULL,
        0xFFFFFFFF00000000ULL
    };

    if (var < 6)
        return P[var];

    return ((start >> var) & 1ULL)
        ? UINT64_MAX : 0ULL;
}

static inline int verify_one(
    uint64_t x,
    const clause3_t *c,
    size_t nc)
{
    for (size_t j=0;j<nc;j++) {
        int ok=0;

        for (int k=0;k<3;k++)
            ok |= (((x >> c[j].v[k]) & 1ULL) == c[j].s[k]);

        if (!ok)
            return 0;
    }

    return 1;
}

uint64_t delta_sat_bit64(
    unsigned nv,
    const clause3_t *c,
    size_t nc,
    uint64_t *tested,
    uint64_t max_candidates)
{
    if (nv >= 63)
        return UINT64_MAX;

    uint64_t limit=1ULL << nv;

    if (max_candidates && max_candidates < limit)
        limit=max_candidates;

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

        /* Masque du dernier bloc si la limite
           n'est pas multiple de 64. */
        uint64_t remain=limit-base;

        if (remain < 64)
            alive &= (1ULL << remain)-1ULL;

        while (alive) {

            unsigned bit=
                (unsigned)__builtin_ctzll(alive);

            uint64_t x=base+bit;

            if (x < limit && verify_one(x,c,nc)) {
                *tested=x+1;
                return x;
            }

            alive &= alive-1;
        }

        uint64_t end=base+64ULL;
        *tested=(end < limit) ? end : limit;
    }

    return UINT64_MAX;
}

int delta_sat_verify(
    uint64_t x,
    const clause3_t *c,
    size_t nc)
{
    return verify_one(x,c,nc);
}
