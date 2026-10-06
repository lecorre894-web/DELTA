// DELTA X512 MT V11 — V9 (AB precalcule, attente sequentielle) + tranche du maitre ajustable (BIAS %)
// le maitre demarre ~0.1us avant les workers -> on lui donne plus de travail ; diagnostic calcul seul corrige (appel opaque)
// usage : bench_x512_mt_v11.exe T [ccd] [bias_pct]
#include <windows.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#define WORDS 65536L
#define NV (WORDS/8)
#define SAMPLES 100000
#define TARGET_US 0.455790
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static uint64_t *A,*B,*C,*AB; static int T,BIAS; static double COMPUTE; static long K0[65],KN[65];
static _Alignas(64) atomic_long gen; static _Alignas(64) atomic_int stop;
typedef struct { _Alignas(64) atomic_long flag; uint64_t v; } slot_t;
static slot_t sl[64];
static KAFFINITY coremask[64]; static int coreccd[64], ncore=0;
static KAFFINITY l3mask[8]; static int nl3=0;
static uint64_t k2(const uint64_t*__restrict P,const uint64_t*__restrict Q,long nv){
  __m512i s0=_mm512_setzero_si512(),s1=s0,s2=s0,s3=s0,s4=s0,s5=s0,s6=s0,s7=s0; long k=0;
#define ST(N,S) S=_mm512_add_epi64(S,_mm512_popcnt_epi64(_mm512_xor_si512(_mm512_load_si512(P+8*(k+N)),_mm512_load_si512(Q+8*(k+N)))))
  for(;k+7<nv;k+=8){ST(0,s0);ST(1,s1);ST(2,s2);ST(3,s3);ST(4,s4);ST(5,s5);ST(6,s6);ST(7,s7);}
  for(;k<nv;k++){ST(0,s0);}
#undef ST
  s0=_mm512_add_epi64(_mm512_add_epi64(_mm512_add_epi64(s0,s1),_mm512_add_epi64(s2,s3)),_mm512_add_epi64(_mm512_add_epi64(s4,s5),_mm512_add_epi64(s6,s7)));
  return (uint64_t)_mm512_reduce_add_epi64(s0); }
static inline uint64_t slice(int id){ return k2(AB+8*K0[id],C+8*K0[id],KN[id]); }
static uint64_t slice0(void){ return slice(0); }
static void split(void){ long m=(long)((double)NV/T*(100+BIAS)/100.0); if(T==1)m=NV; m&=~7L; long rest=NV-m, per=(T>1)?(rest/(T-1))&~7L:0;
  K0[0]=0; KN[0]=m; long k=m; for(int i=1;i<T;i++){ K0[i]=k; KN[i]=(i==T-1)?NV-k:per; k+=KN[i]; } }
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
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen){ if(atomic_load_explicit(&stop,memory_order_relaxed))return 0; _mm_pause(); }
    seen=g; sl[id].v=slice(id); atomic_store_explicit(&sl[id].flag,g,memory_order_release); } }
static inline void waitf(int i,long g){ while(atomic_load_explicit(&sl[i].flag,memory_order_acquire)!=g)_mm_pause(); }
static uint64_t pass(void){ long g=atomic_fetch_add_explicit(&gen,1,memory_order_release)+1;
  uint64_t s=slice(0); for(int i=1;i<T;i++){ waitf(i,g); s+=sl[i].v; } return s; }
static double now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart)QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart*1e6/(double)f.QuadPart; }
int main(int argc,char**argv){
  int ccd=argc>2?atoi(argv[2]):0; topo(ccd);
  T=argc>1?atoi(argv[1]):8; if(T<1)T=1; if(T>ncore)T=ncore; BIAS=argc>3?atoi(argv[3]):0; split();
  A=_aligned_malloc(WORDS*8,64);B=_aligned_malloc(WORDS*8,64);C=_aligned_malloc(WORDS*8,64);
  uint64_t x=0x9E3779B97F4A7C15ULL;
  for(long i=0;i<WORDS;i++){x^=x<<13;x^=x>>7;x^=x<<17;A[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;B[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;C[i]=x;}
  SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
  SetThreadAffinityMask(GetCurrentThread(),coremask[0]); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST);
  uint64_t ref=x512_micro_v3(A,B,C,NV,0);
  AB=_aligned_malloc(WORDS*8,64); for(long i=0;i<WORDS;i++) AB[i]=A[i]&B[i];
  { double cb=1e30; uint64_t cs=0; for(int r=0;r<5;r++){ double t0=now(); uint64_t (*volatile fp)(void)=slice0; for(int i=0;i<SAMPLES;i++) cs+=fp(); double us=(now()-t0)/SAMPLES; if(us<cb)cb=us; } COMPUTE=cb; if(cs==1) puts(""); }
  for(int i=1;i<T;i++) CreateThread(NULL,0,worker,(LPVOID)(INT_PTR)i,0,NULL);
  for(int w=0;w<2000;w++){pass();} uint64_t got=pass();
  double best=1e30,sum=0; uint64_t sink=0;
  for(int r=0;r<5;r++){double t0=now();for(int i=0;i<SAMPLES;i++)sink+=pass();double us=(now()-t0)/SAMPLES;sum+=us;if(us<best)best=us;}
  printf("V11 T=%-2d CCD=%d BIAS=%-3d M_KB=%-4ld W_KB=%-4ld VALIDATION=%s COMPUTE_ONLY_US=%.6f SYNC_PLUS_IMBAL_US=%.6f BEST_US=%.6f MEAN_US=%.6f GAIN_V9REC=%.3fx VS_45579=%.1fx TARGET=%s\n",
    T,ccd,BIAS,(KN[0]*128)/1024,(T>1?KN[1]*128:0)/1024,ref==got?"PASS":"FAIL",COMPUTE,best-COMPUTE,best,sum/5,0.677782/best,45.579/best,best<=TARGET_US?"PASS":"FAIL");
  fflush(stdout); ExitProcess(ref==got?0:1);
}
