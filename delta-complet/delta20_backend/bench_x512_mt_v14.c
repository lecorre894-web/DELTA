// DELTA X512 MT V14 — fusion generation+tpop (v13) repartie sur T coeurs (Windows natif, drapeaux par thread)
// generateur en CHUNKS=8 flux fixes : le resultat ne depend pas de T ; une nouvelle graine a chaque passe (pas de cache possible)
// usage : bench_x512_mt_v14.exe T [ccd]   (T divise 8 : 1 2 4 8)
#include <windows.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#define WORDS 65536L
#define NV (WORDS/8)
#define CHUNKS 8
#define CNV (NV/CHUNKS)
typedef uint64_t v8 __attribute__((vector_size(64)));
static inline uint64_t smix(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline v8 xs(v8 x){ x^=x<<13; x^=x>>7; x^=x<<17; return x; }
static inline v8 pc(v8 x){ return (v8)_mm512_popcnt_epi64((__m512i)x); }
static void seed12(uint64_t seed,int ch,v8 st[12]){ uint64_t s=seed*0xD1B54A32D192ED03ULL+(uint64_t)ch*0x8CB92BA72F3D8DD7ULL+1; for(int g=0;g<12;g++) for(int l=0;l<8;l++){ uint64_t v=smix(&s); st[g][l]=v?v:1; } }
static uint64_t fused_chunk(uint64_t seed,int ch){
  v8 st[12]; seed12(seed,ch,st);
  v8 a0=st[0],b0=st[1],c0=st[2],a1=st[3],b1=st[4],c1=st[5],a2=st[6],b2=st[7],c2=st[8],a3=st[9],b3=st[10],c3=st[11];
  v8 s0={0},s1={0},s2={0},s3={0};
  for(long k=0;k<CNV;k+=4){
    a0=xs(a0);b0=xs(b0);c0=xs(c0); a1=xs(a1);b1=xs(b1);c1=xs(c1);
    a2=xs(a2);b2=xs(b2);c2=xs(c2); a3=xs(a3);b3=xs(b3);c3=xs(c3);
    s0+=pc(c0^(a0&b0)); s1+=pc(c1^(a1&b1)); s2+=pc(c2^(a2&b2)); s3+=pc(c3^(a3&b3)); }
  v8 s=s0+s1+s2+s3; uint64_t t=0; for(int i=0;i<8;i++) t+=s[i]; return t; }
static void materialize(uint64_t seed,uint64_t*A,uint64_t*B,uint64_t*C){
  for(int ch=0;ch<CHUNKS;ch++){ v8 st[12]; seed12(seed,ch,st); long base=(long)ch*CNV;
    for(long k=0;k<CNV;k+=4) for(int g=0;g<4;g++){ v8 *a=&st[3*g],*b=&st[3*g+1],*c=&st[3*g+2]; *a=xs(*a);*b=xs(*b);*c=xs(*c);
      for(int l=0;l<8;l++){ long w=8*(base+k+g)+l; A[w]=(*a)[l]; B[w]=(*b)[l]; C[w]=(*c)[l]; } } } }
#define SAMPLES 100000
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static int T; static _Alignas(64) atomic_long gen; static _Alignas(64) volatile uint64_t SEED;
typedef struct { _Alignas(64) atomic_long flag; uint64_t v; } slot_t; static slot_t sl[64];
static KAFFINITY coremask[64]; static int coreccd[64], ncore=0; static KAFFINITY l3mask[8]; static int nl3=0;
static inline uint64_t work(int id,uint64_t sd){ uint64_t s=0; for(int c=id*CHUNKS/T;c<(id+1)*CHUNKS/T;c++) s+=fused_chunk(sd,c); return s; }
static int lowbit(KAFFINITY m){ for(int i=0;i<64;i++) if(m&((KAFFINITY)1<<i)) return i; return -1; }
static void topo(int ccdfirst){
  DWORD len=0; GetLogicalProcessorInformationEx(RelationAll,NULL,&len);
  char *buf=malloc(len); GetLogicalProcessorInformationEx(RelationAll,(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf,&len);
  KAFFINITY raw[64]; int nr=0;
  for(DWORD off=0;off<len;){ PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p=(void*)(buf+off);
    if(p->Relationship==RelationProcessorCore && nr<64) raw[nr++]=p->Processor.GroupMask[0].Mask;
    if(p->Relationship==RelationCache && p->Cache.Level==3 && nl3<8) l3mask[nl3++]=p->Cache.GroupMask.Mask;
    off+=p->Size; }
  free(buf);
  int first=-1; for(int c=0;c<nr;c++) for(int l=0;l<nl3;l++) if(raw[c]&l3mask[l]){ if(c==0) first=l; }
  int order[8],no=0; if(first<0)first=0;
  int start=(ccdfirst&&nl3>1)?(first+1)%nl3:first;
  for(int k=0;k<nl3;k++) order[no++]=(start+k)%nl3;
  for(int k=0;k<(nl3?nl3:1);k++) for(int c=0;c<nr;c++){
    int l=-1; for(int j=0;j<nl3;j++) if(raw[c]&l3mask[j]) l=j;
    if(nl3==0 || l==order[k]){ coremask[ncore]=(KAFFINITY)1<<lowbit(raw[c]); coreccd[ncore]=l; ncore++; } }
}
static DWORD WINAPI worker(LPVOID p){ int id=(int)(INT_PTR)p; SetThreadAffinityMask(GetCurrentThread(),coremask[id]);
  SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST); long seen=0;
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen) _mm_pause();
    seen=g; sl[id].v=work(id,SEED); atomic_store_explicit(&sl[id].flag,g,memory_order_release); } }
static uint64_t pass(uint64_t sd){ SEED=sd; long g=atomic_fetch_add_explicit(&gen,1,memory_order_release)+1;
  uint64_t s=work(0,sd); for(int i=1;i<T;i++){ while(atomic_load_explicit(&sl[i].flag,memory_order_acquire)!=g)_mm_pause(); s+=sl[i].v; } return s; }
static double now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart)QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart*1e6/(double)f.QuadPart; }
int main(int argc,char**argv){
  int ccd=argc>2?atoi(argv[2]):1; topo(ccd);
  T=argc>1?atoi(argv[1]):8; if(T<1)T=1; if(T>8)T=8; while(CHUNKS%T) T--; if(T>ncore)T=ncore;
  SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
  SetThreadAffinityMask(GetCurrentThread(),coremask[0]); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST);
  uint64_t *A=_aligned_malloc(WORDS*8,64),*B=_aligned_malloc(WORDS*8,64),*C=_aligned_malloc(WORDS*8,64);
  for(int i=1;i<T;i++) CreateThread(NULL,0,worker,(LPVOID)(INT_PTR)i,0,NULL);
  for(int w=0;w<2000;w++) pass(7+w);
  int ok=1; for(uint64_t sd=1;sd<=5;sd++){ materialize(sd,A,B,C); ok&=(pass(sd)==x512_micro_v3(A,B,C,NV,0)); }
  double best=1e30,sum=0; uint64_t sink=0;
  for(int r=0;r<5;r++){double t0=now();for(int i=0;i<SAMPLES;i++)sink+=pass(100000+(uint64_t)r*SAMPLES+i);double us=(now()-t0)/SAMPLES;sum+=us;if(us<best)best=us;}
  printf("V14 T=%d CCD=%d VALIDATION_5_GRAINES=%s BEST_US=%.6f MEAN_US=%.6f VS_V13_9.197=%.2fx VS_PIPELINE_NUMPY_1864=%.0fx SINK=%04llx\n",
    T,ccd,ok?"PASS":"FAIL",best,sum/5,9.197/best,1864.0/best,(unsigned long long)(sink&0xffff));
  fflush(stdout); ExitProcess(ok?0:1);
}
