/* DELTA HYBRIDE : popcount((A_i AND B_j) XOR C) pour toutes les paires i,j
   aiguilleur par tuile d'1 Mbit : plages compressees (tuile structuree) | multiplicateur 4x4 tranches 1024 o AVX-512 (tuile brute) | repli scalaire sans AVX-512 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#define TW 16384          /* mots de 64 bits par tuile = 1 048 576 bits */
#define SL 128            /* tranche 1024 octets */
#define FMAX 200          /* bascules moyennes max par vecteur et par tuile pour passer en plages */
typedef struct{uint32_t*f;int n;}R;
typedef struct{int na,nb,nt;long nw;const uint64_t*A,*B,*C;char*mode;R*rl;long nrle,mem_rle;}H;
static int avx(void){__builtin_cpu_init();return __builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512vpopcntdq");}
static int flips(const uint64_t*w,long n,uint32_t*out,uint64_t prev){int k=0;for(long i=0;i<n;i++){uint64_t x=w[i]^((w[i]<<1)|prev);prev=w[i]>>63;while(x){int b=__builtin_ctzll(x);if(out)out[k]=(uint32_t)(i*64+b);k++;x&=x-1;}}return k;}
static uint64_t rle3(R a,R b,R c,uint32_t len){int i=0,j=0,k=0,x=0,y=0,z=0;uint32_t p=0;uint64_t t=0;for(;;){uint32_t q=len;if(i<a.n&&a.f[i]<q)q=a.f[i];if(j<b.n&&b.f[j]<q)q=b.f[j];if(k<c.n&&c.f[k]<q)q=c.f[k];if((x&y)^z)t+=q-p;if(q==len)break;p=q;if(i<a.n&&a.f[i]==q){x^=1;i++;}if(j<b.n&&b.f[j]==q){y^=1;j++;}if(k<c.n&&c.f[k]==q){z^=1;k++;}}return t;}
H* hy_prepare(const uint64_t*A,int na,const uint64_t*B,int nb,const uint64_t*C,long nw){
 H*h=calloc(1,sizeof(H));h->A=A;h->B=B;h->C=C;h->na=na;h->nb=nb;h->nw=nw;h->nt=(int)((nw+TW-1)/TW);int nv=na+nb+1;
 h->mode=calloc(h->nt,1);h->rl=calloc((size_t)h->nt*nv,sizeof(R));
 for(int t=0;t<h->nt;t++){long o=(long)t*TW,len=nw-o<TW?nw-o:TW;long tot=0;
  for(int v=0;v<nv&&tot<=(long)FMAX*nv;v++){const uint64_t*w=v<na?A+(long)v*nw:v<na+nb?B+(long)(v-na)*nw:C;tot+=flips(w+o,len,0,0);}
  if(tot<=(long)FMAX*nv){h->mode[t]=1;h->nrle++;
   for(int v=0;v<nv;v++){const uint64_t*w=v<na?A+(long)v*nw:v<na+nb?B+(long)(v-na)*nw:C;int n=flips(w+o,len,0,0);R*r=&h->rl[(size_t)t*nv+v];r->n=n;r->f=malloc(4*(n+1));flips(w+o,len,r->f,0);h->mem_rle+=4L*n;}}}
 return h;}
static void multi_avx(H*h,long o,long len,uint64_t*out)__attribute__((target("avx512f,avx512vpopcntdq")));
static void multi_avx(H*h,long o,long len,uint64_t*out){long nw=h->nw;
 for(int i0=0;i0<h->na;i0+=4)for(int j0=0;j0<h->nb;j0+=4){__m512i k[4][4];for(int p=0;p<4;p++)for(int q=0;q<4;q++)k[p][q]=_mm512_setzero_si512();
  const uint64_t*pc=h->C+o,*pa[4],*pb[4];for(int p=0;p<4;p++){pa[p]=h->A+(long)(i0+p)*nw+o;pb[p]=h->B+(long)(j0+p)*nw+o;}
  for(long s=0;s<len;s+=SL){long e=s+SL<len?s+SL:len;long w=s;
   for(;w+8<=e;w+=8){__m512i c=_mm512_loadu_si512(pc+w),a0=_mm512_loadu_si512(pa[0]+w),a1=_mm512_loadu_si512(pa[1]+w),a2=_mm512_loadu_si512(pa[2]+w),a3=_mm512_loadu_si512(pa[3]+w);
    _Pragma("GCC unroll 4") for(int q=0;q<4;q++){__m512i bb=_mm512_loadu_si512(pb[q]+w);
     k[0][q]=_mm512_add_epi64(k[0][q],_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(a0,bb,c,0x6A)));k[1][q]=_mm512_add_epi64(k[1][q],_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(a1,bb,c,0x6A)));
     k[2][q]=_mm512_add_epi64(k[2][q],_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(a2,bb,c,0x6A)));k[3][q]=_mm512_add_epi64(k[3][q],_mm512_popcnt_epi64(_mm512_ternarylogic_epi64(a3,bb,c,0x6A)));}}
   for(;w<e;w++)for(int p=0;p<4;p++)for(int q=0;q<4;q++)out[(long)(i0+p)*h->nb+j0+q]+=__builtin_popcountll((pa[p][w]&pb[q][w])^pc[w]);}
  for(int p=0;p<4;p++)for(int q=0;q<4;q++)out[(long)(i0+p)*h->nb+j0+q]+=_mm512_reduce_add_epi64(k[p][q]);}}
static void multi_scal(H*h,long o,long len,uint64_t*out){long nw=h->nw;
 for(int i=0;i<h->na;i++)for(int j=0;j<h->nb;j++){const uint64_t*a=h->A+(long)i*nw+o,*b=h->B+(long)j*nw+o,*c=h->C+o;uint64_t s=0;for(long w=0;w<len;w++)s+=__builtin_popcountll((a[w]&b[w])^c[w]);out[(long)i*h->nb+j]+=s;}}
int hy_run(H*h,uint64_t*out,int force){ /* force : 0 auto, 1 dense seul, 2 scalaire standard */
 int nv=h->na+h->nb+1,useavx=avx()&&force!=2&&h->na%4==0&&h->nb%4==0;memset(out,0,8L*h->na*h->nb);
 for(int t=0;t<h->nt;t++){long o=(long)t*TW,len=h->nw-o<TW?h->nw-o:TW;
  if(h->mode[t]&&force==0){R*r=&h->rl[(size_t)t*nv];for(int i=0;i<h->na;i++)for(int j=0;j<h->nb;j++)out[(long)i*h->nb+j]+=rle3(r[i],r[h->na+j],r[nv-1],(uint32_t)(len*64));}
  else if(useavx)multi_avx(h,o,len,out);else multi_scal(h,o,len,out);}
 return useavx;}
void hy_stats(H*h,long*st){st[0]=h->nt;st[1]=h->nrle;st[2]=h->mem_rle;st[3]=avx();}
void hy_free(H*h){int nv=h->na+h->nb+1;for(long i=0;i<(long)h->nt*nv;i++)free(h->rl[i].f);free(h->rl);free(h->mode);free(h);}
