/* DELTA MOTEUR L1 : grosses donnees (hors cache), K operations par mot.
   NAIF  : K passes completes sur toute la memoire (chaque passe repasse par la RAM).
   TUILE : on charge 8 Kio dans le L1, on applique les K operations en AVX-512 pendant que c'est chaud, on ecrit une seule fois.
   Operation k : x = rotl(x ^ C[k], 13) + C[k]  (non simplifiable par le compilateur). Validation par checksum contre une reference scalaire. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#define T 1024              /* mots par tuile = 8 Kio : source tuile + travail tiennent dans le L1 */
static uint64_t C[64];
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static inline uint64_t op(uint64_t x,int k){x^=C[k];x=(x<<13)|(x>>51);return x+C[k];}
static void naif(const uint64_t*s,uint64_t*d,uint64_t*tmp,size_t n,int K){   /* K passes RAM, AVX-512 lui aussi */
 const uint64_t*a=s;uint64_t*b=(K&1)?d:tmp;
 for(int k=0;k<K;k++){__m512i c=_mm512_set1_epi64(C[k]);for(size_t i=0;i<n;i+=8){__m512i x=_mm512_loadu_si512(a+i);x=_mm512_add_epi64(_mm512_rol_epi64(_mm512_xor_si512(x,c),13),c);_mm512_storeu_si512(b+i,x);}
  a=b;b=(b==d)?tmp:d;}}
static void tuile(const uint64_t*s,uint64_t*d,size_t n,int K){
 for(size_t o=0;o<n;o+=T){__m512i v[T/8];
  for(int j=0;j<T/8;j++)v[j]=_mm512_loadu_si512(s+o+8*j);            /* un seul passage RAM -> L1/registres */
  for(int k=0;k<K;k++){__m512i c=_mm512_set1_epi64(C[k]);for(int j=0;j<T/8;j++)v[j]=_mm512_add_epi64(_mm512_rol_epi64(_mm512_xor_si512(v[j],c),13),c);}
  for(int j=0;j<T/8;j++)_mm512_storeu_si512(d+o+8*j,v[j]);}}           /* un seul passage L1 -> RAM */
static uint64_t somme(const uint64_t*d,size_t n){uint64_t s=0;for(size_t i=0;i<n;i++)s+=d[i]*(i|1);return s;}
int main(int c,char**v){size_t mib=c>1?atol(v[1]):64,n=mib*1024*1024/8;
 for(int k=0;k<64;k++)C[k]=0x9E3779B97F4A7C15ULL*(k+1)^(0xD1B54A32D192ED03ULL>>k);
 uint64_t*s=aligned_alloc(64,n*8),*d=aligned_alloc(64,n*8),*tmp=aligned_alloc(64,n*8),*r=aligned_alloc(64,n*8);
 for(size_t i=0;i<n;i++)s[i]=i*2654435761ULL;memset(d,0,n*8);memset(tmp,0,n*8);
 printf("=== DELTA MOTEUR L1 : %zu Mio de donnees, tuile %d mots (%d Kio), AVX-512 ===\n",mib,T,T*8/1024);
 printf("%4s | %10s %10s | %10s %10s | %7s | %s\n","K","naif ms","GB/s log.","tuile ms","GB/s log.","gain","validation");int ok=1;
 int Ks[]={1,2,4,8,16,32};
 for(int q=0;q<6;q++){int K=Ks[q];
  for(size_t i=0;i<n;i++){uint64_t x=s[i];for(int k=0;k<K;k++)x=op(x,k);r[i]=x;}uint64_t ref=somme(r,n);
  naif(s,d,tmp,n,K);tuile(s,d,n,K);                                   /* echauffement */
  double t0=now();naif(s,d,tmp,n,K);double tn=now()-t0;uint64_t cn=somme(d,n);
  t0=now();tuile(s,d,n,K);double tt=now()-t0;uint64_t ct=somme(d,n);
  int v=(cn==ref)&&(ct==ref);ok&=v;double lb=(double)K*n*8*2/1e9;
  printf("%4d | %10.2f %10.1f | %10.2f %10.1f | x%6.2f | %s\n",K,tn*1e3,lb/tn,tt*1e3,lb/tt,tn/tt,v?"exact":"FAUX");fflush(stdout);}
 printf("GB/s logiques = K x (lus+ecrits) / temps : travail rendu, pas debit physique de la RAM\nDELTA_L1_MOTEUR=%s\n",ok?"PASSED":"FAILED");}
