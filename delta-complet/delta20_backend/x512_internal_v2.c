#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

/*
 * DELTA X512 INTERNAL V2
 *
 * Chemin spécialisé pour le programme micro actuel :
 *
 *   xtern X4,A,B,C,0x6A
 *   xtern X5,A,B,C,0x6A
 *   accumulation du nombre total de bits
 *
 * Aucun état persistant n'est nécessaire au résultat final.
 *
 * 4 accumulateurs indépendants :
 * réduction de la chaîne de dépendance vpaddq.
 */

static inline uint64_t hsum512(__m512i x)
{
    __m256i lo = _mm512_castsi512_si256(x);
    __m256i hi = _mm512_extracti64x4_epi64(x,1);

    __m256i s256 = _mm256_add_epi64(lo,hi);

    __m128i lo128 =
        _mm256_castsi256_si128(s256);

    __m128i hi128 =
        _mm256_extracti128_si256(s256,1);

    __m128i s128 =
        _mm_add_epi64(lo128,hi128);

    return
        (uint64_t)_mm_cvtsi128_si64(s128) +
        (uint64_t)_mm_extract_epi64(s128,1);
}


uint64_t
x512_micro_v2(
    const uint64_t *A,
    const uint64_t *B,
    const uint64_t *C,
    long nv,
    long *ni
)
{
    /*
     * nv = nombre de vecteurs 512 bits,
     * identique à l'ABI X512 actuelle.
     */

    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    __m512i acc2 = _mm512_setzero_si512();
    __m512i acc3 = _mm512_setzero_si512();

    long k=0;

    /*
     * Chaque itération traite 4 vecteurs de 512 bits.
     * Les quatre chaînes d'accumulation sont indépendantes.
     */

    for(; k+3<nv; k+=4)
    {
        __m512i a0 =
            _mm512_load_si512((const void*)(A+8*(k+0)));
        __m512i b0 =
            _mm512_load_si512((const void*)(B+8*(k+0)));
        __m512i c0 =
            _mm512_load_si512((const void*)(C+8*(k+0)));

        __m512i a1 =
            _mm512_load_si512((const void*)(A+8*(k+1)));
        __m512i b1 =
            _mm512_load_si512((const void*)(B+8*(k+1)));
        __m512i c1 =
            _mm512_load_si512((const void*)(C+8*(k+1)));

        __m512i a2 =
            _mm512_load_si512((const void*)(A+8*(k+2)));
        __m512i b2 =
            _mm512_load_si512((const void*)(B+8*(k+2)));
        __m512i c2 =
            _mm512_load_si512((const void*)(C+8*(k+2)));

        __m512i a3 =
            _mm512_load_si512((const void*)(A+8*(k+3)));
        __m512i b3 =
            _mm512_load_si512((const void*)(B+8*(k+3)));
        __m512i c3 =
            _mm512_load_si512((const void*)(C+8*(k+3)));

        __m512i x0 =
            _mm512_ternarylogic_epi64(
                a0,b0,c0,0x6A);

        __m512i x1 =
            _mm512_ternarylogic_epi64(
                a1,b1,c1,0x6A);

        __m512i x2 =
            _mm512_ternarylogic_epi64(
                a2,b2,c2,0x6A);

        __m512i x3 =
            _mm512_ternarylogic_epi64(
                a3,b3,c3,0x6A);

        acc0 = _mm512_add_epi64(
            acc0,
            _mm512_popcnt_epi64(x0));

        acc1 = _mm512_add_epi64(
            acc1,
            _mm512_popcnt_epi64(x1));

        acc2 = _mm512_add_epi64(
            acc2,
            _mm512_popcnt_epi64(x2));

        acc3 = _mm512_add_epi64(
            acc3,
            _mm512_popcnt_epi64(x3));
    }

    /*
     * Queue générique.
     */
    for(; k<nv; k++)
    {
        __m512i a =
            _mm512_load_si512((const void*)(A+8*k));

        __m512i b =
            _mm512_load_si512((const void*)(B+8*k));

        __m512i c =
            _mm512_load_si512((const void*)(C+8*k));

        __m512i x =
            _mm512_ternarylogic_epi64(
                a,b,c,0x6A);

        acc0 = _mm512_add_epi64(
            acc0,
            _mm512_popcnt_epi64(x));
    }

    __m512i sum01 =
        _mm512_add_epi64(acc0,acc1);

    __m512i sum23 =
        _mm512_add_epi64(acc2,acc3);

    __m512i total =
        _mm512_add_epi64(sum01,sum23);

    /*
     * NI sémantique du programme micro original :
     * 9 instructions X512 par paire + mread final.
     */
    if(ni)
        *ni =
            9L*(nv/2L)+1L;

    return hsum512(total);
}
