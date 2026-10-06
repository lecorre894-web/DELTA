// DELTA X512 MT V5 — drapeaux de fin par thread (1 ligne de cache chacun), sans compteur partage ni remise a zero
// mode 0 = plat ; mode 1 = hierarchique par CCD (chef CCD1 = cpu 8 agrege 9..15)
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define WORDS 65536L
#define NV (WORDS/8)
#define SAMPLES 100000
#define TARGET_US 0.455790
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static uint64_t *A,*B,*C; static int T,MODE;
static _Alignas(64) atomic_long gen; static _Alignas(64) atomic_int stop;
typedef struct { _Alignas(64) atomic_long flag; uint64_t v; } slot_t;
static slot_t sl[64];
static inline uint64_t slice(int id){ long per=NV/T,k0=per*id,n=(id==T-1)?NV-k0:per; return x512_micro_v3(A+8*k0,B+8*k0,C+8*k0,n,0); }
static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);}
static inline void waitf(int i,long g){ while(atomic_load_explicit(&sl[i].flag,memory_order_acquire)!=g)_mm_pause(); }
static void *worker(void *p){ int id=(int)(long)p; pin(id); long seen=0;
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen){ if(atomic_load_explicit(&stop,memory_order_relaxed))return 0; _mm_pause(); }
    seen=g; uint64_t v=slice(id);
    if(MODE&&T>8&&id==8){ for(int i=9;i<T;i++){ waitf(i,g); v+=sl[i].v; } }
    sl[id].v=v; atomic_store_explicit(&sl[id].flag,g,memory_order_release); } }
static uint64_t pass(void){ long g=atomic_fetch_add_explicit(&gen,1,memory_order_release)+1;
  uint64_t s=slice(0); int last=(MODE&&T>8)?9:T;
  for(int i=1;i<last;i++){ waitf(i,g); s+=sl[i].v; } return s; }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int argc,char**argv){ T=argc>1?atoi(argv[1]):16; MODE=argc>2?atoi(argv[2]):0; if(T<1||T>64)T=16;
  A=aligned_alloc(64,WORDS*8);B=aligned_alloc(64,WORDS*8);C=aligned_alloc(64,WORDS*8);
  uint64_t x=0x9E3779B97F4A7C15ULL;
  for(long i=0;i<WORDS;i++){x^=x<<13;x^=x>>7;x^=x<<17;A[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;B[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;C[i]=x;}
  pin(0); uint64_t ref=x512_micro_v3(A,B,C,NV,0);
  pthread_t th[64]; for(int i=1;i<T;i++)pthread_create(&th[i],0,worker,(void*)(long)i);
  for(int w=0;w<2000;w++){pass();} uint64_t got=pass();
  double best=1e30,sum=0; uint64_t sink=0;
  for(int r=0;r<5;r++){double t0=now();for(int i=0;i<SAMPLES;i++)sink+=pass();double us=(now()-t0)/SAMPLES;sum+=us;if(us<best)best=us;}
  printf("T=%-2d MODE=%d SLICE_KB=%-4ld VALIDATION=%s BEST_US=%.6f MEAN_US=%.6f GAIN_V3=%.3fx TARGET=%s SINK=%04llx\n",T,MODE,(3*WORDS*8/T)/1024,ref==got?"PASS":"FAIL",best,sum/5,9.232866/best,best<=TARGET_US?"PASS":"FAIL",(unsigned long long)(sink&0xffff));
  atomic_store(&stop,1); atomic_fetch_add(&gen,1); for(int i=1;i<T;i++)pthread_join(th[i],0); return ref==got?0:1; }
