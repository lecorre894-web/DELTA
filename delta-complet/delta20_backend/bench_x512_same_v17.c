// DELTA X512 V17 — BENCH LOURD "MEME TRAVAIL POUR TOUS" (Windows natif)
// Travail identique : R requetes, chacune = graine neuve -> A,B,C de NW mots -> popcount(C ^ (A & B)).
// Methode F (fusion v16) : vecteurs nes en registres, jamais ecrits.
// Methode M (memoire)    : chaque thread ecrit ses A,B,C en RAM/cache puis les relit avec le noyau v3 (x512_micro_v3).
// Meme generateur (32 flux fixes) pour F et M -> memes resultats, valides contre un calcul mono-thread de reference.
// usage : bench_x512_same_v17.exe   (balaye NW = 64K, 256K, 1M, 4M mots ; T = 1, 8 CCD1, 8 CCD0, 16)
#include <windows.h>
#include <immintrin.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#define CHUNKS 32
typedef uint64_t v8 __attribute__((vector_size(64)));
uint64_t x512_micro_v3(const uint64_t*__restrict,const uint64_t*__restrict,const uint64_t*__restrict,long,long*);
static inline uint64_t smix(uint64_t *s){ uint64_t z=(*s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline v8 xs(v8 x){ x^=x<<13; x^=x>>7; x^=x<<17; return x; }
static inline v8 pc(v8 x){ return (v8)_mm512_popcnt_epi64((__m512i)x); }
static void seed12(uint64_t seed,int ch,v8 st[12]){ uint64_t s=seed*0xD1B54A32D192ED03ULL+(uint64_t)ch*0x8CB92BA72F3D8DD7ULL+1; for(int g=0;g<12;g++) for(int l=0;l<8;l++){ uint64_t v=smix(&s); st[g][l]=v?v:1; } }
static long NWD, CNV;
static uint64_t *A,*B,*C;
static uint64_t fused_chunk(uint64_t seed,int ch){
  v8 st[12]; seed12(seed,ch,st);
  v8 a0=st[0],b0=st[1],c0=st[2],a1=st[3],b1=st[4],c1=st[5],a2=st[6],b2=st[7],c2=st[8],a3=st[9],b3=st[10],c3=st[11];
  v8 s0={0},s1={0},s2={0},s3={0};
  for(long k=0;k<CNV;k+=4){
    a0=xs(a0);b0=xs(b0);c0=xs(c0); a1=xs(a1);b1=xs(b1);c1=xs(c1);
    a2=xs(a2);b2=xs(b2);c2=xs(c2); a3=xs(a3);b3=xs(b3);c3=xs(c3);
    s0+=pc(c0^(a0&b0)); s1+=pc(c1^(a1&b1)); s2+=pc(c2^(a2&b2)); s3+=pc(c3^(a3&b3)); }
  v8 s=s0+s1+s2+s3; uint64_t t=0; for(int i=0;i<8;i++) t+=s[i]; return t; }
static void mat_chunk(uint64_t seed,int ch){
  v8 st[12]; seed12(seed,ch,st); long base=(long)ch*CNV;
  v8 *pa=(v8*)A+base,*pb=(v8*)B+base,*pcc=(v8*)C+base;
  for(long k=0;k<CNV;k+=4) for(int g=0;g<4;g++){ st[3*g]=xs(st[3*g]); st[3*g+1]=xs(st[3*g+1]); st[3*g+2]=xs(st[3*g+2]);
    pa[k+g]=st[3*g]; pb[k+g]=st[3*g+1]; pcc[k+g]=st[3*g+2]; } }
static uint64_t mem_chunk(uint64_t seed,int ch){ mat_chunk(seed,ch); long base=(long)ch*CNV; return x512_micro_v3(A+8*base,B+8*base,C+8*base,CNV,0); }
static int T, METH; static _Alignas(64) atomic_long gen; static _Alignas(64) volatile uint64_t SEED;
typedef struct { _Alignas(64) atomic_long flag; uint64_t v; } slot_t; static slot_t sl[64];
static KAFFINITY coremask[64]; static int coreccd[64], ncore=0; static KAFFINITY l3mask[8]; static int nl3=0;
static inline uint64_t work(int id,uint64_t sd){ uint64_t s=0; for(int c=id*CHUNKS/T;c<(id+1)*CHUNKS/T;c++) s+=METH?mem_chunk(sd,c):fused_chunk(sd,c); return s; }
static int lowbit(KAFFINITY m){ for(int i=0;i<64;i++) if(m&((KAFFINITY)1<<i)) return i; return -1; }
static void topo(int ccdfirst){
  ncore=0; nl3=0; DWORD len=0; GetLogicalProcessorInformationEx(RelationAll,NULL,&len);
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
  SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST); long seen=atomic_load(&gen);
  for(;;){ long g; while((g=atomic_load_explicit(&gen,memory_order_acquire))==seen) _mm_pause();
    if(g<0){ return 0; } seen=g; sl[id].v=work(id,SEED); atomic_store_explicit(&sl[id].flag,g,memory_order_release); } }
static uint64_t pass(uint64_t sd){ SEED=sd; long g=atomic_fetch_add_explicit(&gen,1,memory_order_release)+1;
  uint64_t s=work(0,sd); for(int i=1;i<T;i++){ while(atomic_load_explicit(&sl[i].flag,memory_order_acquire)!=g)_mm_pause(); s+=sl[i].v; } return s; }
static double now(void){ static LARGE_INTEGER f; LARGE_INTEGER c; if(!f.QuadPart)QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart*1e6/(double)f.QuadPart; }
static uint64_t reference(uint64_t sd){ for(int c=0;c<CHUNKS;c++) mat_chunk(sd,c); return x512_micro_v3(A,B,C,NWD/8,0); }
int main(void){
  SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
  long sizes[4]={65536,262144,1048576,4194304};
  struct { int T, ccd; const char *nom; } cfg[4]={{1,1,"1 coeur CCD1"},{8,1,"8 coeurs CCD1"},{8,0,"8 coeurs CCD0 (V-cache)"},{16,1,"16 coeurs"}};
  A=_aligned_malloc(sizes[3]*8,64); B=_aligned_malloc(sizes[3]*8,64); C=_aligned_malloc(sizes[3]*8,64);
  printf("DELTA X512 V17 BENCH LOURD MEME TRAVAIL | requete = graine neuve -> 3 x NW mots -> popcount(C^(A&B))\n");
  printf("%-10s %-8s %-25s %-12s %12s %12s %10s %10s %10s\n","NW_mots","Mo/req","CONFIG","METHODE","US/REQ","REQ/S","Gbit/s","VALID","REQUETES");
  long gbase=1;
  for(int zi=0;zi<4;zi++){ NWD=sizes[zi]; CNV=NWD/8/CHUNKS;
    uint64_t refs[3]; for(int v=0;v<3;v++) refs[v]=reference(777+v);
    for(int ci=0;ci<4;ci++) for(METH=0;METH<2;METH++){
      topo(cfg[ci].ccd); T=cfg[ci].T; if(T>ncore)T=ncore;
      for(int i=0;i<T;i++){ atomic_store(&sl[i].flag,0); }
      atomic_store(&gen,gbase);
      SetThreadAffinityMask(GetCurrentThread(),coremask[0]); SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_HIGHEST);
      HANDLE th[64]; for(int i=1;i<T;i++) th[i]=CreateThread(NULL,0,worker,(LPVOID)(INT_PTR)i,0,NULL);
      Sleep(5);
      int ok=1; for(int v=0;v<3;v++) ok&=(pass(777+v)==refs[v]);
      double tp=now(); for(int w=0;w<5;w++) pass(900000+w); tp=(now()-tp)/5; long R=(long)(300000.0/tp); if(R<50)R=50; if(R>400000)R=400000;
      for(int w=0;w<R/10+1;w++) pass(1000000+w);
      double best=1e30; uint64_t sink=0;
      for(int r=0;r<3;r++){ double t0=now(); for(long i=0;i<R;i++) sink+=pass(5000000+(uint64_t)r*R+i); double us=(now()-t0)/R; if(us<best)best=us; }
      double bits=3.0*NWD*64;
      printf("%-10ld %-8.1f %-25s %-12s %12.3f %12.0f %10.1f %10s %10ld%s\n",NWD,3.0*NWD*8/1048576.0,cfg[ci].nom,METH?"M memoire":"F fusion",best,1e6/best,bits/(best*1e3),ok?"PASS":"FAIL",R,(sink==1)?"!":"");
      fflush(stdout);
      gbase=atomic_load(&gen)+1000; atomic_store(&gen,-5); for(int i=1;i<T;i++){ WaitForSingleObject(th[i],INFINITE); CloseHandle(th[i]); }
      atomic_store(&gen,gbase);
    }
    printf("\n");
  }
  fflush(stdout); ExitProcess(0);
}
