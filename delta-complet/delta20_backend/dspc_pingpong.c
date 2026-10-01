#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <time.h>
#define ROUNDS 1000000
#define REPS 5
static _Alignas(64) atomic_int ball;
static _Alignas(64) long pong_count;
static int cpuA=0,cpuB=1;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pin(int c){cpu_set_t s;CPU_ZERO(&s);CPU_SET(c,&s);pthread_setaffinity_np(pthread_self(),sizeof s,&s);}
static void*pong(void*a){(void)a;pin(cpuB);for(long i=0;i<ROUNDS;i++){while(atomic_load_explicit(&ball,memory_order_acquire)!=1);pong_count++;atomic_store_explicit(&ball,0,memory_order_release);}return 0;}
int main(int argc,char**argv){if(argc>2){cpuA=atoi(argv[1]);cpuB=atoi(argv[2]);}
 double r[REPS];long ping_total=0,pong_total=0;pin(cpuA);
 for(int k=0;k<REPS;k++){atomic_store(&ball,0);pong_count=0;pthread_t th;pthread_create(&th,0,pong,0);
  double t0=now();for(long i=0;i<ROUNDS;i++){atomic_store_explicit(&ball,1,memory_order_release);while(atomic_load_explicit(&ball,memory_order_acquire)!=0);ping_total++;}
  double t1=now();pthread_join(th,0);pong_total+=pong_count;r[k]=(t1-t0)/ROUNDS*1e9;}
 for(int i=0;i<REPS;i++)for(int j=i+1;j<REPS;j++)if(r[j]<r[i]){double t=r[i];r[i]=r[j];r[j]=t;}
 printf("=== DSPC PING-PONG CPU%d <-> CPU%d ===\n",cpuA,cpuB);
 printf("ROUNDS=%d x %d REPS\n",ROUNDS,REPS);
 printf("RTT_NS MIN=%.1f MEDIAN=%.1f MAX=%.1f\n",r[0],r[REPS/2],r[REPS-1]);
 printf("ONE_WAY_NS MEDIAN=%.1f\n",r[REPS/2]/2);
 printf("EXCHANGES_PER_S=%.2f M\n",1e3/r[REPS/2]);
 printf("DSPC_PINGPONG_VALIDATION=%s (ping=%ld pong=%ld)\n",ping_total==pong_total&&ping_total==(long)ROUNDS*REPS?"OK":"FAIL",ping_total,pong_total);
 return 0;}
