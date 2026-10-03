#include <immintrin.h>
#include <math.h>
void fma_scalar(float*d,int n,int it,float b,float c){for(int v=0;v<n;v++){float a=d[v];for(int i=0;i<it;i++)a=fmaf(a,b,c);d[v]=a;}}
void fma_avx512(float*d,int n,int it,float b,float c){
#ifdef __AVX512F__
 __m512 B=_mm512_set1_ps(b),C=_mm512_set1_ps(c);int v=0;
 for(;v+256<=n;v+=256){__m512 a[16];for(int k=0;k<16;k++)a[k]=_mm512_loadu_ps(d+v+16*k);
  for(int i=0;i<it;i++)for(int k=0;k<16;k++)a[k]=_mm512_fmadd_ps(a[k],B,C);
  for(int k=0;k<16;k++)_mm512_storeu_ps(d+v+16*k,a[k]);}
 fma_scalar(d+v,n-v,it,b,c);
#else
 fma_scalar(d,n,it,b,c);
#endif
}
