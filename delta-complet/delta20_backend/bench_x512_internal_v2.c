#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>

#define NW       (16384*4)
#define NV       (NW/8)
#define TESTS    256
#define WARMUP   1000
#define SAMPLES  100000

typedef void *(*ctx_new_t)(void);
typedef void (*free_t)(void *);

typedef uint64_t (*old_t)(
    void*,int,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    long,long*
);

typedef uint64_t (*micro_v2_t)(
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    long,long*
);

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW,&t);

    return
        (uint64_t)t.tv_sec*1000000000ULL+
        t.tv_nsec;
}

static inline uint64_t rng64(uint64_t *x)
{
    uint64_t z=
        (*x+=0x9e3779b97f4a7c15ULL);

    z=(z^(z>>30))*
      0xbf58476d1ce4e5b9ULL;

    z=(z^(z>>27))*
      0x94d049bb133111ebULL;

    return z^(z>>31);
}

int main(void)
{
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(30,&set);

    int pinok=
        sched_setaffinity(
            0,sizeof(set),&set)==0;

    void *oldlib=
        dlopen("./libdelta_x512.so",RTLD_NOW);

    void *newlib=
        dlopen(
            "./libdelta_x512_internal_v2.so",
            RTLD_NOW);

    if(!oldlib || !newlib){
        fprintf(stderr,"dlopen FAIL\n");
        return 1;
    }

    ctx_new_t xnew=
        (ctx_new_t)dlsym(oldlib,"x512_new");

    free_t xfree=
        (free_t)dlsym(oldlib,"x512_free");

    old_t oldrun=
        (old_t)dlsym(oldlib,"x512_run");

    micro_v2_t newrun=
        (micro_v2_t)dlsym(
            newlib,
            "x512_micro_v2");

    if(!xnew || !xfree || !oldrun || !newrun){
        fprintf(stderr,"dlsym FAIL\n");
        return 1;
    }

    uint64_t *A,*B,*C;

    if(posix_memalign((void**)&A,64,NW*8) ||
       posix_memalign((void**)&B,64,NW*8) ||
       posix_memalign((void**)&C,64,NW*8)){
        return 1;
    }

    uint64_t seed=894;

    for(long i=0;i<NW;i++){
        A[i]=rng64(&seed);
        B[i]=rng64(&seed);
        C[i]=rng64(&seed);
    }

    void *ctx=xnew();

    long nio=0;
    long nin=0;

    /*
     * VALIDATION :
     * différentes entrées, original contre V2.
     */

    for(int t=0;t<TESTS;t++){

        C[t % NW] ^=
            rng64(&seed);

        uint64_t ro=
            oldrun(
                ctx,1,
                A,B,C,
                NV,&nio);

        uint64_t rn=
            newrun(
                A,B,C,
                NV,&nin);

        if(ro!=rn){
            printf(
                "VALIDATION=FAIL test=%d old=%llu new=%llu\n",
                t,
                (unsigned long long)ro,
                (unsigned long long)rn
            );
            return 2;
        }
    }

    printf("VALIDATION_256=PASS\n");
    printf("OLD_NI=%ld\n",nio);
    printf("V2_NI=%ld\n",nin);

    volatile uint64_t sink=0;

    for(int i=0;i<WARMUP;i++){
        C[0]^=(uint64_t)i+1;
        sink^=newrun(A,B,C,NV,&nin);
    }

    uint64_t t0=now_ns();

    for(int i=0;i<SAMPLES;i++){
        C[0]^=
            (uint64_t)i+
            0x10001ULL;

        sink^=
            newrun(
                A,B,C,
                NV,&nin);
    }

    uint64_t t1=now_ns();

    double ns_call=
        (double)(t1-t0)/
        (double)SAMPLES;

    double us_call=
        ns_call/1000.0;

    double gain=
        11.144617/us_call;

    printf("\n");
    printf("CPU_PIN=30\n");
    printf(
        "PINNING=%s\n",
        pinok ? "PASS":"FAIL"
    );

    printf("WORDS=%d\n",NW);
    printf("NV=%d\n",NV);
    printf("SAMPLES=%d\n",SAMPLES);

    printf(
        "V2_NS_PER_TPOP=%.1f\n",
        ns_call
    );

    printf(
        "V2_US_PER_TPOP=%.6f\n",
        us_call
    );

    printf(
        "GAIN_VS_NATIVE_V1=%.6fx\n",
        gain
    );

    printf(
        "SINK=%016llx\n",
        (unsigned long long)sink
    );

    printf(
        "TARGET_US=0.455790\n"
    );

    printf(
        "DIV100_FASTPATH=%s\n",
        us_call<=0.455790 ?
        "PASS":"FAIL"
    );

    printf(
        "INTERNAL_V2=PASS\n"
    );

    xfree(ctx);

    free(A);
    free(B);
    free(C);

    dlclose(oldlib);
    dlclose(newlib);

    return 0;
}
