// DELTA X512 MT V8 — V7 (Windows natif, vrais coeurs, tri L3) + drapeaux de fin par thread estampilles
// + remontee hierarchique : un chef par CCD agrege son groupe, une seule traversee inter-CCD
// usage : bench_x512_mt_v8.exe T [ccd] [mode]   mode 0 = drapeaux plats ; mode 1 = hierarchique CCD
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
static uint64_t *A,*B,*C; static int T;
static _Alignas(64) atomic_long gen; static _Alignas(64) atomic_int stop;
typedef struct { _Alignas(64) atomic_long flag; uint64_t v; } slot_t;
static slot_t sl[64]; static int MODE, lead[64], isleader[64];
static KAFFINITY coremask[64]; static int coreccd[64], ncore=0;
static KAFFINITY l3mask[8]; static int nl3=0;
static inline uint64_t slice(int id){ long per=NV/T,k0=per*id,n=(id==T-1)?NV-k0:per; return x512_micro_v3(A+8*k0,B+8*k0,C+8*k0,n,0); }
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
static inline void waitf(int i,long g){ while(atomic_load_explicit(&sl[i].flag,memory_order_acquire)!=g)_mm_pause(); }
static inline uint64_t gather(int me,long g){ uint64_t s=0;
  for(int i=0;i<T;i++){ if(i==me) continue;
    if(MODE==0){ if(me==0){ waitf(i,g); s+=sl[i].v; } }
    else if(lead[i]==me || (me==0 && isleader[i] && i!=0)){ waitf(i,g); s+=sl[i].v; } }
  return s; }
static DWORD WINAPI worker(LPVOID p){ int id=(int)(INT_PTR)p; SetThreadAffinityMask(GetCurrentThread(),coremask[id]);
  SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST); long seen=0;
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen){ if(atomic_load_explicit(&stop,memory_order_relaxed))return 0; _mm_pause(); }
    seen=g; uint64_t v=slice(id); if(MODE==1 && isleader[id]) v+=gather(id,g);
    sl[id].v=v; atomic_store_explicit(&sl[id].flag,g,memory_order_release); } }
static uint64_t pass(void){ long g=atomic_fetch_add_explicit(&gen,1,memory_order_release)+1;
  uint64_t s=slice(0); return s+gather(0,g); }
static double now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart)QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart*1e6/(double)f.QuadPart; }
int main(int argc,char**argv){
  int ccd=argc>2?atoi(argv[2]):0; MODE=argc>3?atoi(argv[3]):0; topo(ccd);
  T=argc>1?atoi(argv[1]):8; if(T<1)T=1; if(T>ncore)T=ncore;
  for(int i=0;i<T;i++){ int j=0; while(coreccd[j]!=coreccd[i]) j++; lead[i]=(i==j)?-1:j; isleader[i]=(i==j); } lead[0]=-1;
  printf("PHYS_CORES=%d L3_GROUPS=%d ORDER=",ncore,nl3); for(int i=0;i<T;i++)printf("%d:LP%d/L3_%d ",i,lowbit(coremask[i]),coreccd[i]); printf("\n");
  A=_aligned_malloc(WORDS*8,64);B=_aligned_malloc(WORDS*8,64);C=_aligned_malloc(WORDS*8,64);
  uint64_t x=0x9E3779B97F4A7C15ULL;
  for(long i=0;i<WORDS;i++){x^=x<<13;x^=x>>7;x^=x<<17;A[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;B[i]=x;x^=x<<13;x^=x>>7;x^=x<<17;C[i]=x;}
  SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
  SetThreadAffinityMask(GetCurrentThread(),coremask[0]); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST);
  uint64_t ref=x512_micro_v3(A,B,C,NV,0);
  HANDLE th[64]; for(int i=1;i<T;i++) th[i]=CreateThread(NULL,0,worker,(LPVOID)(INT_PTR)i,0,NULL);
  for(int w=0;w<2000;w++){pass();} uint64_t got=pass();
  double best=1e30,sum=0; uint64_t sink=0;
  for(int r=0;r<5;r++){double t0=now();for(int i=0;i<SAMPLES;i++)sink+=pass();double us=(now()-t0)/SAMPLES;sum+=us;if(us<best)best=us;}
  printf("T=%-2d CCD_FIRST=%d MODE=%d SLICE_KB=%-4ld VALIDATION=%s BEST_US=%.6f MEAN_US=%.6f GAIN_V3=%.3fx TARGET=%s SINK=%04llx\n",
    T,ccd,MODE,(3*WORDS*8/T)/1024,ref==got?"PASS":"FAIL",best,sum/5,9.232866/best,best<=TARGET_US?"PASS":"FAIL",(unsigned long long)(sink&0xffff));
  fflush(stdout); ExitProcess(ref==got?0:1);
  return ref==got?0:1;
}
