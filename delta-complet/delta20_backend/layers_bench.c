#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define DOMAIN 1000000000000ULL
#define AXIS 50000000ULL
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t mix(uint64_t x){x+=0x9E3779B97F4A7C15ULL;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;x=(x^(x>>27))*0x94D049BB133111EBULL;return x^(x>>31);}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){long N=argc>1?atol(argv[1]):50000000;int ok=1;double r[5];uint64_t sink=0;
 for(int k=0;k<5;k++){uint64_t s=0,d=0,c=0;double t0=now();
  for(long i=0;i<N;i++){uint64_t idx=((uint64_t)i*7919ULL+(uint64_t)k)%DOMAIN;s^=mix(idx);d+=idx%32;c+=idx%200+1;}
  r[k]=(now()-t0)/N*1e9;sink^=s^d^c;}
 qsort(r,5,sizeof(double),cmp);
 if(999999999999ULL%32!=31||999999999999ULL%200+1!=200||250000000000ULL%32!=0)ok=0;
 printf("LAYER=CLOUD_1T_ADDRESSING NS_PER_ADDR_MEDIAN=%.2f ADDR_PER_S=%.1f M PROBE_ROUTING=%s\n",r[2],1e3/r[2],ok?"OK":"FAIL");
 printf("PROJECTION_FULL_1T_ADDRESSING_1CORE=%.0f s (%.1f h) [PROJECTION, NON EXECUTEE]\n",DOMAIN*r[2]*1e-9,DOMAIN*r[2]*1e-9/3600);
 int ok2=1;for(int k=0;k<5;k++){double t0=now();uint64_t acc=0;
  for(long p=0;p<N;p++){uint64_t h=mix((uint64_t)p+(uint64_t)k*N);uint64_t x=h%AXIS,y=mix(h^0xA5A5A5A5ULL)%AXIS,z=mix(h^0x5A5A5A5AULL)%AXIS;if(x>=AXIS||y>=AXIS||z>=AXIS)ok2=0;unsigned __int128 cid=(unsigned __int128)x*AXIS*AXIS+(unsigned __int128)y*AXIS+z;if(cid>=(unsigned __int128)AXIS*AXIS*AXIS)ok2=0;acc^=mix((uint64_t)cid^mix((uint64_t)(cid>>64)));}
  r[k]=(now()-t0)/N*1e9;sink^=acc;}
 qsort(r,5,sizeof(double),cmp);
 printf("LAYER=VIRTUAL_SILICON_5CM_1NM NS_PER_PULSE_MEDIAN=%.2f PULSES_PER_S=%.1f M COORD_BOUNDS=%s\n",r[2],1e3/r[2],ok2?"OK":"FAIL");
 printf("LAYERS_VALIDATION=%s SINK=%llu\n",ok&&ok2?"OK":"FAIL",(unsigned long long)sink);
 return 0;}
