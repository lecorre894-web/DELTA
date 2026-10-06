// DELTA X512 V12 — MISE A JOUR INCREMENTALE (changement de contrat, pas de magie)
// total = popcount(C XOR (A AND B)) maintenu en continu ; a chaque appel, seuls M mots de C changent
// cout d'un appel = M mots, plus 65536 : valable SEULEMENT si DELTA sait quels mots de C changent
// mono-coeur, aucun thread, aucune synchro. Validation : total incremental == recalcul complet (v3)
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define WORDS 65536L
#define NV (WORDS/8)
#define TARGET_US 0.1
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static uint64_t rng=0x9E3779B97F4A7C15ULL; static inline uint64_t xr(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
int main(void){
  uint64_t *A=aligned_alloc(64,WORDS*8),*B=aligned_alloc(64,WORDS*8),*C=aligned_alloc(64,WORDS*8),*AB=aligned_alloc(64,WORDS*8);
  for(long i=0;i<WORDS;i++){A[i]=xr();B[i]=xr();C[i]=xr();AB[i]=A[i]&B[i];}
  int64_t total=(int64_t)x512_micro_v3(A,B,C,NV,0);
  printf("V12 INCREMENTAL | WORDS=%ld | TOTAL_INITIAL=%lld (v3)\n",WORDS,(long long)total);
  int Ms[]={8,16,32,64,128,256,1024,4096};
  for(int t=0;t<8;t++){ int M=Ms[t]; long POOL=(1L<<20)/M; if(POOL<64)POOL=64;
    uint32_t *idx=malloc(POOL*M*4); uint64_t *nv=malloc(POOL*M*8);
    for(long j=0;j<POOL*M;j++){ idx[j]=(uint32_t)(xr()%WORDS); nv[j]=xr(); }
    long CALLS=4000000/M; if(CALLS<20000)CALLS=20000; double best=1e30;
    for(int r=0;r<5;r++){ double t0=now();
      for(long c=0;c<CALLS;c++){ const uint32_t *ix=idx+(c%POOL)*M; const uint64_t *nw=nv+(c%POOL)*M; int64_t d=0;
        for(int j=0;j<M;j++){ uint32_t i=ix[j]; uint64_t ab=AB[i]; d+=__builtin_popcountll(nw[j]^ab)-__builtin_popcountll(C[i]^ab); C[i]=nw[j]; }
        total+=d; }
      double us=(now()-t0)/CALLS; if(us<best)best=us; }
    int64_t full=(int64_t)x512_micro_v3(A,B,C,NV,0);
    printf("V12 M=%-5d mots/appel  VALIDATION=%s  BEST_US=%.6f  NS_PAR_MOT=%.2f  VS_45579=%.0fx  TARGET_0.1US=%s\n",
      M,full==total?"PASS":"FAIL",best,best*1000.0/M,45.579/best,best<=TARGET_US?"PASS":"FAIL");
    free(idx); free(nv); }
  return 0;
}
