#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define NUM_INVOCATIONS (256 * 4096 * 4)
#define ITERS 200000
#define FMA_FLOPS 2
static double now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1000.0+ts.tv_nsec/1e6;}
int main(void){
  float *data=aligned_alloc(64,NUM_INVOCATIONS*sizeof(float));
  for(int i=0;i<NUM_INVOCATIONS;i++)data[i]=1.0f;
  const __m512 b=_mm512_set1_ps(1.000001f),c=_mm512_set1_ps(0.0000001f);
  double t0=now_ms();
  #pragma omp parallel for schedule(static)
  for(int v=0;v<NUM_INVOCATIONS;v+=16*8){
    __m512 a[8];for(int k=0;k<8;k++)a[k]=_mm512_load_ps(data+v+16*k);
    for(int i=0;i<ITERS;++i)for(int k=0;k<8;k++)a[k]=_mm512_fmadd_ps(a[k],b,c);
    for(int k=0;k<8;k++)_mm512_store_ps(data+v+16*k,a[k]);}
  double t1=now_ms(),el=t1-t0,fl=(double)NUM_INVOCATIONS*ITERS*FMA_FLOPS;
  double s=0;for(int i=0;i<NUM_INVOCATIONS;i+=4099)s+=data[i];
  printf("Using device: AVX-512 x8 chaines independantes (meme calcul, meme resultat)\n");
  printf("Invocations (scalar elements): %d, iters/invocation: %d\n",NUM_INVOCATIONS,ITERS);
  printf("Elapsed: %.3f ms\nEstimated: %.2f GFLOPS\nChecksum: %.6f\n",el,fl/(el/1000.0)/1e9,s);
  free(data);return 0;}
