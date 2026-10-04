/* DELTA DDR5 experimentale : cartographie L1 -> L2 -> L3 -> RAM, version optimisee.
   Meme travail que le banc de Rene : lire x, ecrire x+1, sommer x. Octets comptes = lus + ecrits.
   Optimisations : boucle continue restrict (AVX auto), passe d'echauffement, threads epingles sur des coeurs distincts,
   nombre de passes adapte a la taille (duree ~0,25 s), stores non-temporels au-dela du L3 (variante NT). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
typedef struct{uint8_t*a,*b;size_t n;long passes;int nt,cpu;uint64_t sum;pthread_barrier_t*bar;}W;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t pass_std(const uint64_t*restrict s,uint64_t*restrict d,size_t n){uint64_t sum=0;for(size_t i=0;i<n;i++){uint64_t x=s[i];d[i]=x+1;sum+=x;}return sum;}
static uint64_t pass_nt(const uint64_t*restrict s,uint64_t*restrict d,size_t n){__m512i acc=_mm512_setzero_si512(),one=_mm512_set1_epi64(1);
 for(size_t i=0;i<n;i+=8){__m512i x=_mm512_load_si512(s+i);_mm512_stream_si512((__m512i*)(d+i),_mm512_add_epi64(x,one));acc=_mm512_add_epi64(acc,x);}
 _mm_sfence();return (uint64_t)_mm512_reduce_add_epi64(acc);}
static uint64_t pass_v4(const uint64_t*restrict s,uint64_t*restrict d,size_t n){__m512i a0=_mm512_setzero_si512(),a1=a0,a2=a0,a3=a0,one=_mm512_set1_epi64(1);size_t i=0;
 for(;i+32<=n;i+=32){__m512i x0=_mm512_load_si512(s+i),x1=_mm512_load_si512(s+i+8),x2=_mm512_load_si512(s+i+16),x3=_mm512_load_si512(s+i+24);
  _mm512_store_si512(d+i,_mm512_add_epi64(x0,one));_mm512_store_si512(d+i+8,_mm512_add_epi64(x1,one));_mm512_store_si512(d+i+16,_mm512_add_epi64(x2,one));_mm512_store_si512(d+i+24,_mm512_add_epi64(x3,one));
  a0=_mm512_add_epi64(a0,x0);a1=_mm512_add_epi64(a1,x1);a2=_mm512_add_epi64(a2,x2);a3=_mm512_add_epi64(a3,x3);}
 uint64_t r=(uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi64(_mm512_add_epi64(a0,a1),_mm512_add_epi64(a2,a3)));for(;i<n;i++){uint64_t x=s[i];d[i]=x+1;r+=x;}return r;}
typedef uint64_t(*PF)(const uint64_t*restrict,uint64_t*restrict,size_t);
static void*travail(void*p){W*w=p;cpu_set_t c;CPU_ZERO(&c);CPU_SET(w->cpu,&c);pthread_setaffinity_np(pthread_self(),sizeof c,&c);
 uint64_t*s=(uint64_t*)w->a,*d=(uint64_t*)w->b,sum=0;
 ((PF[]){pass_std,pass_nt,pass_v4})[w->nt](s,d,w->n);                       /* echauffement (non compte) */
 pthread_barrier_wait(w->bar);
 for(long k=0;k<w->passes;k++){sum+=((PF[]){pass_std,pass_nt,pass_v4})[w->nt](s,d,w->n);uint64_t*t=s;s=d;d=t;}
 pthread_barrier_wait(w->bar);w->sum=sum;return 0;}
static double mesure(size_t bytes,int nthr,int nt,uint64_t*ck){size_t n=bytes/8;long passes=(long)(5e9/(double)(bytes*2));if(passes<4)passes=4;
 W w[2];pthread_t th[2];pthread_barrier_t bar;pthread_barrier_init(&bar,0,nthr+1);
 for(int i=0;i<nthr;i++){w[i]=(W){aligned_alloc(64,bytes),aligned_alloc(64,bytes),n,passes,nt,i,0,&bar};
  for(size_t j=0;j<n;j++)((uint64_t*)w[i].a)[j]=j*2654435761ULL;memset(w[i].b,0,bytes);pthread_create(&th[i],0,travail,&w[i]);}
 pthread_barrier_wait(&bar);double t0=now();pthread_barrier_wait(&bar);double dt=now()-t0;
 *ck=0;for(int i=0;i<nthr;i++){pthread_join(th[i],0);*ck+=w[i].sum;free(w[i].a);free(w[i].b);}pthread_barrier_destroy(&bar);
 return (double)bytes*2*passes*nthr/dt/1e9;}
/* reference independante : meme calcul en scalaire simple, pour valider la somme */
static uint64_t ref(size_t bytes,long passes){size_t n=bytes/8;uint64_t*s=malloc(bytes),*d=calloc(n,8),sum=0;for(size_t j=0;j<n;j++)s[j]=j*2654435761ULL;
 for(size_t j=0;j<n;j++)d[j]=s[j]+1;  /* echauffement */
 for(long k=0;k<passes;k++){for(size_t j=0;j<n;j++){sum+=s[j];d[j]=s[j]+1;}uint64_t*t=s;s=d;d=t;}free(s);free(d);return sum;}
int main(void){
 size_t kib[]={16,32,128,512,1024,4096,8192,16384,65536,262144};const char*niv[]={"L1","L1","L2","L2","L2/L3","L3","L3","L3/RAM","RAM","RAM"};
 printf("=== CARTE MEMOIRE DELTA (taille = UNE zone ; empreinte = source+destination = 2x) ===\n");
 printf("%9s %-7s | %9s %9s %9s | %9s | %9s %9s | %s\n","zone","niveau","auto 1c","auto 2c","x dual","NT 1c","AVX512 1c","AVX512 2c","validation");
 int ok=1;
 for(int i=0;i<10;i++){size_t b=kib[i]*1024;uint64_t c1,c2,c3;double g1=mesure(b,1,0,&c1),g2=mesure(b,2,0,&c2),g3=b>=4096*1024?mesure(b,1,1,&c3):0;uint64_t c4,c5;double g4=mesure(b,1,2,&c4),g5=mesure(b,2,2,&c5);
  long passes=(long)(5e9/(double)(b*2));if(passes<4)passes=4;
  int v=1;if(b<=8*1024*1024){uint64_t r=ref(b,passes);v=(c1==r)&&(c2==2*r)&&(!g3||c3==r)&&c4==r&&c5==2*r;}ok&=v;
  char nt[16];if(g3)snprintf(nt,sizeof nt,"%9.1f",g3);else snprintf(nt,sizeof nt,"%9s","-");
  printf("%6zu Kio %-7s | %9.1f %9.1f %8.2fx | %s | %9.1f %9.1f | %s\n",kib[i],niv[i],g1,g2,g2/g1,nt,g4,g5,b<=8*1024*1024?(v?"exact":"FAUX"):"non verifie (taille)");fflush(stdout);}
 printf("GB/s = octets lus + ecrits par seconde ; NT = stores non-temporels AVX-512 (contournent le cache)\nCARTE_VALIDATION=%s\n",ok?"PASSED":"FAILED");}
