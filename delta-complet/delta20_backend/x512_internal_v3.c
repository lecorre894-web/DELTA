#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>

static inline uint64_t hsum512(__m512i x)
{
    __m256i a = _mm512_castsi512_si256(x);
    __m256i b = _mm512_extracti64x4_epi64(x,1);
    __m256i s = _mm256_add_epi64(a,b);

    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s,1);
    __m128i t = _mm_add_epi64(lo,hi);

    return
        (uint64_t)_mm_cvtsi128_si64(t) +
        (uint64_t)_mm_extract_epi64(t,1);
}

uint64_t x512_micro_v3(
    const uint64_t *__restrict A,
    const uint64_t *__restrict B,
    const uint64_t *__restrict C,
    long nv,
    long *ni)
{
    __m512i s0=_mm512_setzero_si512();
    __m512i s1=_mm512_setzero_si512();
    __m512i s2=_mm512_setzero_si512();
    __m512i s3=_mm512_setzero_si512();
    __m512i s4=_mm512_setzero_si512();
    __m512i s5=_mm512_setzero_si512();
    __m512i s6=_mm512_setzero_si512();
    __m512i s7=_mm512_setzero_si512();

    long k=0;

    for(; k+7<nv; k+=8)
    {
#define STEP(N,S) do { \
        __m512i a = _mm512_load_si512((const void *)(A+8*(k+(N)))); \
        __m512i b = _mm512_load_si512((const void *)(B+8*(k+(N)))); \
        __m512i c = _mm512_load_si512((const void *)(C+8*(k+(N)))); \
        __m512i x = _mm512_ternarylogic_epi64(a,b,c,0x6A); \
        (S) = _mm512_add_epi64((S),_mm512_popcnt_epi64(x)); \
    } while(0)

        STEP(0,s0);
        STEP(1,s1);
        STEP(2,s2);
        STEP(3,s3);
        STEP(4,s4);
        STEP(5,s5);
        STEP(6,s6);
        STEP(7,s7);

#undef STEP
    }

    /*
     * Queue éventuelle : conserve la généralité de nv.
     */
    for(; k<nv; k++)
    {
        __m512i a=
            _mm512_load_si512((const void *)(A+8*k));

        __m512i b=
            _mm512_load_si512((const void *)(B+8*k));

        __m512i c=
            _mm512_load_si512((const void *)(C+8*k));

        __m512i x=
            _mm512_ternarylogic_epi64(a,b,c,0x6A);

        s0=_mm512_add_epi64(
            s0,
            _mm512_popcnt_epi64(x));
    }

    s0=_mm512_add_epi64(s0,s1);
    s2=_mm512_add_epi64(s2,s3);
    s4=_mm512_add_epi64(s4,s5);
    s6=_mm512_add_epi64(s6,s7);

    s0=_mm512_add_epi64(s0,s2);
    s4=_mm512_add_epi64(s4,s6);

    s0=_mm512_add_epi64(s0,s4);

    if(ni)
        *ni=9L*(nv/2L)+1L;

    return hsum512(s0);
}
