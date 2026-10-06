// DELTA X512 MT V6 — V4 + un thread par coeur physique (WSL: SMT adjacents, pin 2*id)
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
static uint64_t *A,*B,*C; static int T;
static _Alignas(64) atomic_long gen; static _Alignas(64) atomic_long done; static _Alignas(64) atomic_int stop;
static struct { _Alignas(64) uint64_t v; } part[64];
static inline uint64_t slice(int id){ long per=NV/T,k0=per*id,n=(id==T-1)?NV-k0:per; return x512_micro_v3(A+8*k0,B+8*k0,C+8*k0,n,0); }
static void pin(int cpu){cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpu,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);}
static void *worker(void *p){ int id=(int)(long)p; pin(2*id); long seen=0;
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen){ if(atomic_load_explicit(&stop,memory_order_relaxed))return 0; _mm_pause(); }
    seen=g; part[id].v=slice(id); atomic_fetch_add_explicit(&done,1,memory_order_release); } }
static uint64_t pass(void){ atomic_store_explicit(&done,0,memory_order_relaxed); atomic_fetch_add_explicit(&gen,1,memory_order_release);
  part[0].v=slice(0); while(atomic_load_explicit(&done,memory_order_acquire)<T-1)_mm_pause();
  uint64_t s=0; for(int i=0;i<T;i++)s+=part[i].v; return s; }
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
int main(int argc,char**argv){ T=argc>1?atoi(argv[1]):16; if(T<1||T>64)T=16;
  A=aligned_alloc(64,WORDS*8);B=aligned_alloc(64,WORDS*8);C=aligned_alloc(64,WORDS*8);
  uint64_t x=0x9E3779B97F4A7C15ULL;
  for(long i=0;i<WORDS;i++){x^=x<<13;x^=x>>7;x^=x<<17;A[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;B[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;C[i]=x;}
  pin(0); uint64_t ref=x512_micro_v3(A,B,C,NV,0);
  pthread_t th[64]; for(int i=1;i<T;i++)pthread_create(&th[i],0,worker,(void*)(long)i);
  for(int w=0;w<2000;w++)pass(); uint64_t got=pass();
  printf("THREADS=%d WORDS=%ld NV=%ld SLICE_KB=%ld\n",T,WORDS,(long)NV,(3*WORDS*8/T)/1024);
  printf("REF_V3=%llu MT_V4=%llu VALIDATION=%s\n",(unsigned long long)ref,(unsigned long long)got,ref==got?"PASS":"FAIL");
  double best=1e30,sum=0; uint64_t sink=0;
  for(int r=0;r<5;r++){double t0=now();for(int i=0;i<SAMPLES;i++)sink+=pass();double us=(now()-t0)/SAMPLES;sum+=us;if(us<best)best=us;printf("RUN%d_US=%.6f\n",r,us);}
  printf("V4_MEAN_US=%.6f V4_BEST_US=%.6f\nV4_GAIN_VS_V3_9_232866=%.3fx\nTARGET_%.6f_US=%s\nSINK=%016llx\n",sum/5,best,9.232866/best,TARGET_US,best<=TARGET_US?"PASS":"FAIL",(unsigned long long)sink);
  atomic_store(&stop,1); atomic_fetch_add(&gen,1); for(int i=1;i<T;i++)pthread_join(th[i],0); return ref==got?0:1; }
