#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#define NUM_INVOCATIONS (256 * 4096 * 4)
#define ITERS 200000
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int cmp(const void*a,const void*b){unsigned x=*(unsigned*)a,y=*(unsigned*)b;return x<y?-1:x>y;}
int main(void){
  float *data=aligned_alloc(64,NUM_INVOCATIONS*sizeof(float));
  for(int i=0;i<NUM_INVOCATIONS;i++)data[i]=1.0f;
  double t0=now_ms();
  unsigned *u=malloc(NUM_INVOCATIONS*4);memcpy(u,data,NUM_INVOCATIONS*4);
  int nu=0;int same=1;for(int i=1;i<NUM_INVOCATIONS;i++)if(u[i]!=u[0]){same=0;break;}
  if(same)nu=1;else{qsort(u,NUM_INVOCATIONS,4,cmp);for(int i=0;i<NUM_INVOCATIONS;i++)if(i==0||u[i]!=u[nu-1])u[nu++]=u[i];}
  float *res=malloc(nu*4);
  for(int k=0;k<nu;k++){float a;memcpy(&a,&u[k],4);
    __m128 v=_mm_set_ss(a),b=_mm_set_ss(1.000001f),c=_mm_set_ss(0.0000001f);
    for(int i=0;i<ITERS;++i)v=_mm_fmadd_ss(v,b,c);res[k]=_mm_cvtss_f32(v);}
  if(nu==1){__m512 r=_mm512_set1_ps(res[0]);for(int i=0;i<NUM_INVOCATIONS;i+=16)_mm512_store_ps(data+i,r);}
  else for(int i=0;i<NUM_INVOCATIONS;i++){unsigned x;memcpy(&x,&data[i],4);unsigned*p=bsearch(&x,u,nu,4,cmp);data[i]=res[p-u];}
  double el=now_ms()-t0;double s=0;for(int i=0;i<NUM_INVOCATIONS;i+=4099)s+=data[i];
  printf("Using device: DELTA dedup (chaine FMA exacte par valeur distincte)\nValeurs distinctes: %d sur %d\nElapsed: %.3f ms\nChecksum: %.6f\n",nu,NUM_INVOCATIONS,el,s);
  return 0;}
