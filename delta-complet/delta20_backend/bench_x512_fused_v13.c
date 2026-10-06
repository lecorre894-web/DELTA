// DELTA X512 V13 — GENERATION + TPOP FUSIONNES : les vecteurs A,B,C ne sont jamais ecrits en memoire,
// ils naissent dans les registres (xorshift64 vectoriel, 4 flux entrelaces) et sont consommes aussitot.
// Mesure DELTA reelle : 99 % du temps etait la generation NumPy (1846.7 us) contre 17.3 us de tpop.
// ATTENTION : generateur deterministe par graine, mais DIFFERENT de numpy.default_rng -> les vecteurs changent.
// Validation : meme generateur materialise en memoire -> x512_micro_v3 -> doit egaler le fusionne.
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define WORDS 65536L
#define NV (WORDS/8)
typedef uint64_t v8 __attribute__((vector_size(64)));
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static inline uint64_t smix(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline v8 xs(v8 x){ x^=x<<13; x^=x>>7; x^=x<<17; return x; }
static inline v8 pc(v8 x){
#ifdef __AVX512VPOPCNTDQ__
  return (v8)_mm512_popcnt_epi64((__m512i)x);
#else
  v8 r; for(int i=0;i<8;i++) r[i]=__builtin_popcountll(x[i]); return r;
#endif
}
static void seed12(uint64_t seed, v8 st[12]){ uint64_t s=seed; for(int g=0;g<12;g++) for(int l=0;l<8;l++){ uint64_t v=smix(&s); st[g][l]=v?v:1; } }
static uint64_t fused(uint64_t seed){
  v8 st[12]; seed12(seed,st);
  v8 a0=st[0],b0=st[1],c0=st[2],a1=st[3],b1=st[4],c1=st[5],a2=st[6],b2=st[7],c2=st[8],a3=st[9],b3=st[10],c3=st[11];
  v8 s0={0},s1={0},s2={0},s3={0};
  for(long k=0;k<NV;k+=4){
    a0=xs(a0);b0=xs(b0);c0=xs(c0); a1=xs(a1);b1=xs(b1);c1=xs(c1);
    a2=xs(a2);b2=xs(b2);c2=xs(c2); a3=xs(a3);b3=xs(b3);c3=xs(c3);
    s0+=pc(c0^(a0&b0)); s1+=pc(c1^(a1&b1)); s2+=pc(c2^(a2&b2)); s3+=pc(c3^(a3&b3)); }
  v8 s=s0+s1+s2+s3; uint64_t t=0; for(int i=0;i<8;i++) t+=s[i]; return t; }
static void materialize(uint64_t seed,uint64_t*A,uint64_t*B,uint64_t*C){
  v8 st[12]; seed12(seed,st);
  for(long k=0;k<NV;k+=4) for(int g=0;g<4;g++){ v8 *a=&st[3*g],*b=&st[3*g+1],*c=&st[3*g+2]; *a=xs(*a);*b=xs(*b);*c=xs(*c);
    for(int l=0;l<8;l++){ A[8*(k+g)+l]=(*a)[l]; B[8*(k+g)+l]=(*b)[l]; C[8*(k+g)+l]=(*c)[l]; } } }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int argc,char**argv){
  int N=argc>1?atoi(argv[1]):20000;
  uint64_t *A=aligned_alloc(64,WORDS*8),*B=aligned_alloc(64,WORDS*8),*C=aligned_alloc(64,WORDS*8);
  int ok=1; for(uint64_t sd=1;sd<=5;sd++){ materialize(sd,A,B,C); ok&=(fused(sd)==x512_micro_v3(A,B,C,NV,0)); }
  printf("V13 FUSION GEN+TPOP | WORDS=%ld | VALIDATION_5_GRAINES=%s | VPOPCNTDQ=%s\n",WORDS,ok?"PASS":"FAIL",
#ifdef __AVX512VPOPCNTDQ__
  "OUI"
#else
  "NON(scalaire)"
#endif
  );
  double best=1e30; uint64_t sink=0;
  for(int r=0;r<5;r++){ double t0=now(); for(int i=0;i<N;i++) sink+=fused(1000+i); double us=(now()-t0)/N; if(us<best)best=us; }
  double tm=1e30; for(int r=0;r<3;r++){ double t0=now(); for(int i=0;i<N/20+1;i++){ materialize(2000+i,A,B,C); sink+=x512_micro_v3(A,B,C,NV,0);} double us=(now()-t0)/(N/20+1); if(us<tm)tm=us; }
  printf("V13 FUSED_US=%.3f  GEN_MEMOIRE+V3_US=%.3f  VS_NUMPY_1846.7=%.0fx  VS_PIPELINE_NUMPY+TPOP_1864=%.0fx  SINK=%04llx\n",best,tm,1846.7/best,1864.0/best,(unsigned long long)(sink&0xffff));
  return ok?0:1; }
