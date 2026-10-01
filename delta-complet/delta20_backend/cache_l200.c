#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#define LV 200
#define BASE 64
#define HOT 500000ULL
#define DOMAIN 1000000000000ULL
typedef struct{uint64_t key,val;}E;
static E*L[LV];static uint32_t cap[LV];static long hits[LV],miss;static int ROUNDS;static uint64_t rs=88172645463325252ULL;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t rnd(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return rs;}
static uint64_t mix(uint64_t x){x+=0x9E3779B97F4A7C15ULL;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;x=(x^(x>>27))*0x94D049BB133111EBULL;return x^(x>>31);}
static uint64_t compute(uint64_t i){uint64_t v=i;for(int r=0;r<ROUNDS;r++)v=mix(v);return v;}
static void insert(uint64_t k,uint64_t v){E e={k,v};for(int l=0;l<LV;l++){E*s=&L[l][e.key%cap[l]];if(!s->key){*s=e;return;}E t=*s;*s=e;e=t;}}
static uint64_t get(uint64_t i){uint64_t k=i+1;for(int l=0;l<LV;l++){E*s=&L[l][k%cap[l]];if(s->key==k){hits[l]++;uint64_t v=s->val;if(l){s->key=0;insert(k,v);}return v;}}miss++;uint64_t v=compute(i);insert(k,v);return v;}
static E*F;static size_t FC;static long fh,fm;
static uint64_t getflat(uint64_t i){uint64_t k=i+1;E*s=&F[k%FC];if(s->key==k){fh++;return s->val;}fm++;uint64_t v=compute(i);s->key=k;s->val=v;return v;}
static uint64_t pick(void){return (rnd()%10)<9?(rnd()%HOT)*2000003ULL%DOMAIN:rnd()%DOMAIN;}
static void reset(void){for(int l=0;l<LV;l++)memset(L[l],0,(size_t)cap[l]*sizeof(E));memset(hits,0,sizeof hits);miss=0;memset(F,0,FC*sizeof(E));fh=fm=0;}
int main(int argc,char**argv){long N=argc>1?atol(argv[1]):1000000;size_t tot=0;
 for(int l=0;l<LV;l++){cap[l]=BASE*(l+1);L[l]=calloc(cap[l],sizeof(E));tot+=cap[l];}FC=tot;F=calloc(FC,sizeof(E));
 printf("=== DSPC CACHE L1-L200 (exclusif, promotion L1, cascade d'eviction) ===\n");
 printf("LEVELS=%d CAPACITY=%zu ENTRIES MEM=%.1f MB ACCESSES=%ld PATTERN=90%% HOT(%llu) + 10%% UNIFORM(1T)\n",LV,tot,tot*sizeof(E)/1048576.0,N,HOT);
 int rr[3]={1,100,1000};int ok=1;
 for(int q=0;q<3;q++){ROUNDS=rr[q];reset();
  rs=88172645463325252ULL;volatile uint64_t sk=0;double t0=now();for(long n=0;n<N;n++)sk^=compute(pick());double tn=now()-t0;
  rs=88172645463325252ULL;uint64_t sc=0;t0=now();for(long n=0;n<N;n++)sc^=get(pick());double tc=now()-t0;
  rs=88172645463325252ULL;uint64_t sf=0;t0=now();for(long n=0;n<N;n++)sf^=getflat(pick());double tf=now()-t0;
  if(sc!=sk||sf!=sk)ok=0;
  for(int c=0;c<100000;c++){uint64_t i=pick();if(get(i)!=compute(i))ok=0;}
  long h=0,h1=hits[0],h2=0,h3=0;for(int l=0;l<LV;l++)h+=hits[l];for(int l=1;l<10;l++)h2+=hits[l];for(int l=10;l<LV;l++)h3+=hits[l];double A=(double)(h+miss);
  printf("ROUNDS=%4d NOCACHE_NS=%8.1f CACHE_NS=%8.1f SPEEDUP=x%6.2f HIT=%.1f%% L1=%.1f%% L2-L10=%.1f%% L11-L200=%.1f%% MISS=%.1f%%\n",
   ROUNDS,tn/N*1e9,tc/N*1e9,tn/tc,100*h/A,100*h1/A,100*h2/A,100*h3/A,100*miss/A);
  printf("            FLAT_1LEVEL_NS=%8.1f SPEEDUP=x%6.2f HIT=%.1f%% (meme capacite, 1 niveau)\n",tf/N*1e9,tn/tf,100.0*fh/(fh+fm));}
 printf("CACHE_VALIDATION=%s (checksum cache == recalcul + 100000 controles)\n",ok?"OK":"FAIL");
 return 0;}
