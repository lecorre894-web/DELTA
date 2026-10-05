#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>
#include <math.h>

#define NW       (16384*4)
#define NV       (NW/8)
#define TESTS    512
#define WARMUP   2000
#define SAMPLES  100000

typedef void *(*ctx_new_t)(void);
typedef void (*ctx_free_t)(void *);

typedef uint64_t (*old_t)(
    void*,int,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    long,long*
);

typedef uint64_t (*micro_t)(
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    long,long*
);

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;
}

static inline uint64_t rng64(uint64_t *x)
{
    uint64_t z=(*x+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}

static double runbench(
    micro_t f,
    uint64_t *A,
    uint64_t *B,
    uint64_t *C,
    long *ni,
    volatile uint64_t *sink)
{
    for(int i=0;i<WARMUP;i++){
        C[0]^=(uint64_t)i+0x1234ULL;
        *sink^=f(A,B,C,NV,ni);
    }

    uint64_t t0=now_ns();

    for(int i=0;i<SAMPLES;i++){
        C[0]^=(uint64_t)i+0x10001ULL;
        *sink^=f(A,B,C,NV,ni);
    }

    uint64_t t1=now_ns();

    return
        ((double)(t1-t0)/(double)SAMPLES)/1000.0;
}

int main(void)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(30,&set);

    int pinok=
        sched_setaffinity(0,sizeof(set),&set)==0;

    void *L0=dlopen("./libdelta_x512.so",RTLD_NOW);
    void *L2=dlopen("./libdelta_x512_internal_v2.so",RTLD_NOW);
    void *L3=dlopen("./libdelta_x512_internal_v3.so",RTLD_NOW);

    if(!L0 || !L2 || !L3){
        fprintf(stderr,"DLOPEN=FAIL\n");
        return 1;
    }

    ctx_new_t xnew=
        (ctx_new_t)dlsym(L0,"x512_new");

    ctx_free_t xfree=
        (ctx_free_t)dlsym(L0,"x512_free");

    old_t original=
        (old_t)dlsym(L0,"x512_run");

    micro_t v2=
        (micro_t)dlsym(L2,"x512_micro_v2");

    micro_t v3=
        (micro_t)dlsym(L3,"x512_micro_v3");

    if(!xnew || !xfree || !original || !v2 || !v3){
        fprintf(stderr,"DLSYM=FAIL\n");
        return 1;
    }

    uint64_t *A=NULL,*B=NULL,*C=NULL;

    if(posix_memalign((void**)&A,64,NW*sizeof(uint64_t)) ||
       posix_memalign((void**)&B,64,NW*sizeof(uint64_t)) ||
       posix_memalign((void**)&C,64,NW*sizeof(uint64_t))){
        fprintf(stderr,"ALLOC=FAIL\n");
        return 1;
    }

    uint64_t seed=894;

    for(long i=0;i<NW;i++){
        A[i]=rng64(&seed);
        B[i]=rng64(&seed);
        C[i]=rng64(&seed);
    }

    void *ctx=xnew();

    if(!ctx){
        fprintf(stderr,"CTX=FAIL\n");
        return 1;
    }

    long ni0=0,ni2=0,ni3=0;

    /*
     * Validation sur 512 mutations différentes.
     */
    for(int t=0;t<TESTS;t++)
    {
        long p=(long)(
            rng64(&seed) % (uint64_t)NW);

        C[p]^=rng64(&seed);

        uint64_t r0=
            original(ctx,1,A,B,C,NV,&ni0);

        uint64_t r2=
            v2(A,B,C,NV,&ni2);

        uint64_t r3=
            v3(A,B,C,NV,&ni3);

        if(r0!=r2 || r0!=r3){
            printf(
                "VALIDATION=FAIL t=%d "
                "OLD=%llu V2=%llu V3=%llu\n",
                t,
                (unsigned long long)r0,
                (unsigned long long)r2,
                (unsigned long long)r3);
            return 2;
        }
    }

    printf("VALIDATION_512=PASS\n");
    printf(
        "NI OLD=%ld V2=%ld V3=%ld\n",
        ni0,ni2,ni3);

    if(ni0!=ni2 || ni0!=ni3){
        printf("NI_VALIDATION=FAIL\n");
        return 3;
    }

    printf("NI_VALIDATION=PASS\n\n");

    volatile uint64_t sink=0;

    /*
     * Deux passages, ordre inversé.
     */
    double v2a=
        runbench(v2,A,B,C,&ni2,&sink);

    double v3a=
        runbench(v3,A,B,C,&ni3,&sink);

    double v3b=
        runbench(v3,A,B,C,&ni3,&sink);

    double v2b=
        runbench(v2,A,B,C,&ni2,&sink);

    double u2=(v2a+v2b)/2.0;
    double u3=(v3a+v3b)/2.0;

    printf("CPU_PIN=30\n");
    printf("PINNING=%s\n",pinok?"PASS":"FAIL");
    printf("WORDS=%d\n",NW);
    printf("NV=%d\n",NV);
    printf("SAMPLES_PER_PASS=%d\n\n",SAMPLES);

    printf("V2_PASS_A_US=%.6f\n",v2a);
    printf("V2_PASS_B_US=%.6f\n",v2b);
    printf("V2_MEAN_US=%.6f\n\n",u2);

    printf("V3_PASS_A_US=%.6f\n",v3a);
    printf("V3_PASS_B_US=%.6f\n",v3b);
    printf("V3_MEAN_US=%.6f\n\n",u3);

    printf(
        "V3_GAIN_VS_V2=%.6fx\n",
        u2/u3);

    printf(
        "V3_GAIN_VS_NATIVE_11_144617=%.6fx\n",
        11.144617/u3);

    printf(
        "REMAINING_TO_0_455790=%.6fx\n",
        u3/0.455790);

    printf(
        "WINNER=%s\n",
        u3<u2 ? "INTERNAL_V3":"INTERNAL_V2");

    printf(
        "TARGET_0_455790_US=%s\n",
        u3<=0.455790 ? "PASS":"FAIL");

    printf(
        "SINK=%016llx\n",
        (unsigned long long)sink);

    xfree(ctx);
    free(A);
    free(B);
    free(C);

    dlclose(L0);
    dlclose(L2);
    dlclose(L3);

    return 0;
}
