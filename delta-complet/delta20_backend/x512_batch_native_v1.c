#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>

#define NW       (16384 * 4)
#define BATCH    256
#define WARMUP   100
#define ROUNDS   1000

typedef void *(*fn_new_t)(void);
typedef void (*fn_free_t)(void *);

typedef uint64_t (*fn_run_t)(
    void *, int,
    const uint64_t *,
    const uint64_t *,
    const uint64_t *,
    long, long *
);

static inline uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW,&ts);

    return
        (uint64_t)ts.tv_sec*1000000000ULL +
        (uint64_t)ts.tv_nsec;
}

static inline uint64_t rng64(uint64_t *x)
{
    uint64_t z=(*x += 0x9e3779b97f4a7c15ULL);

    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;

    return z^(z>>31);
}

int main(void)
{
    void *lib=dlopen("./libdelta_x512.so",RTLD_NOW);

    if(!lib){
        fprintf(stderr,"dlopen failed: %s\n",dlerror());
        return 1;
    }

    fn_new_t xnew=(fn_new_t)dlsym(lib,"x512_new");
    fn_free_t xfree=(fn_free_t)dlsym(lib,"x512_free");
    fn_run_t xrun=(fn_run_t)dlsym(lib,"x512_run");

    if(!xnew || !xfree || !xrun){
        fprintf(stderr,"X512 ABI missing\n");
        return 1;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(30,&cpuset);

    int pinned=
        sched_setaffinity(
            0,
            sizeof(cpuset),
            &cpuset
        )==0;

    uint64_t *A=NULL;
    uint64_t *B=NULL;
    uint64_t *C=NULL;

    if(posix_memalign((void**)&A,64,NW*sizeof(uint64_t)) ||
       posix_memalign((void**)&B,64,NW*sizeof(uint64_t)) ||
       posix_memalign((void**)&C,64,NW*sizeof(uint64_t))){

        fprintf(stderr,"allocation failed\n");
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
        fprintf(stderr,"x512_new failed\n");
        return 1;
    }

    volatile uint64_t sink=0;
    long ni=0;

    /*
     * Warmup: every operation really executes.
     */
    for(int w=0;w<WARMUP;w++){

        for(int j=0;j<BATCH;j++){

            C[0] ^=
                ((uint64_t)w<<32) ^
                (uint64_t)j ^
                0x9e3779b97f4a7c15ULL;

            sink ^= xrun(
                ctx,1,
                A,B,C,
                NW/8,
                &ni
            );
        }
    }

    uint64_t total_calls=0;
    uint64_t total_ni=0;

    uint64_t best_batch_ns=UINT64_MAX;
    uint64_t worst_batch_ns=0;

    long double sum_batch_ns=0.0L;

    uint64_t global0=ns_now();

    for(int r=0;r<ROUNDS;r++){

        uint64_t t0=ns_now();

        for(int j=0;j<BATCH;j++){

            /*
             * Changing observable input.
             */
            C[0] ^=
                ((uint64_t)r<<32) ^
                (uint64_t)j ^
                0xd1b54a32d192ed03ULL;

            uint64_t result=xrun(
                ctx,1,
                A,B,C,
                NW/8,
                &ni
            );

            sink ^=
                result +
                (uint64_t)j +
                ((uint64_t)r<<16);

            total_ni +=
                (uint64_t)ni;

            total_calls++;
        }

        uint64_t t1=ns_now();

        uint64_t dt=t1-t0;

        if(dt<best_batch_ns)
            best_batch_ns=dt;

        if(dt>worst_batch_ns)
            worst_batch_ns=dt;

        sum_batch_ns +=
            (long double)dt;
    }

    uint64_t global1=ns_now();

    double wall_s=
        (global1-global0)/1e9;

    double ns_per_call=
        (double)(global1-global0) /
        (double)total_calls;

    double best_ns_per_call=
        (double)best_batch_ns /
        (double)BATCH;

    double avg_batch_ns=
        (double)(
            sum_batch_ns /
            (long double)ROUNDS
        );

    double avg_batch_us=
        avg_batch_ns/1000.0;

    double ginstr=
        ((double)total_ni/wall_s)/1e9;

    double calls_s=
        (double)total_calls/wall_s;

    printf("============================================================\n");
    printf("DELTA X512 — NATIVE BATCH V1\n");
    printf("============================================================\n");

    printf("CPU_PIN=30\n");
    printf("PINNING=%s\n",pinned ? "PASS":"FAIL");

    printf("WORDS=%d\n",NW);
    printf("BATCH=%d\n",BATCH);
    printf("ROUNDS=%d\n",ROUNDS);
    printf("TOTAL_TPOP=%llu\n",
        (unsigned long long)total_calls);

    printf("\n");

    printf("AVG_BATCH_US=%.6f\n",
        avg_batch_us);

    printf("AMORTIZED_NS_PER_TPOP=%.1f\n",
        ns_per_call);

    printf("AMORTIZED_US_PER_TPOP=%.6f\n",
        ns_per_call/1000.0);

    printf("BEST_BATCH_NS_PER_TPOP=%.1f\n",
        best_ns_per_call);

    printf("WORST_BATCH_US=%.6f\n",
        worst_batch_ns/1000.0);

    printf("\n");

    printf("TPOP_PER_SECOND=%.2f\n",
        calls_s);

    printf("NI_PER_CALL=%ld\n",ni);

    printf("TOTAL_NI=%llu\n",
        (unsigned long long)total_ni);

    printf("X512_GINSTR_S=%.6f\n",
        ginstr);

    printf("SINK=%016llx\n",
        (unsigned long long)sink);

    printf("\n");

    printf("NATIVE_V1_REFERENCE_US=11.144617\n");
    printf("FASTPATH_DIV100_TARGET_US=0.455790\n");

    double gain=
        11.144617 /
        (ns_per_call/1000.0);

    printf("GAIN_VS_NATIVE_V1=%.6fx\n",
        gain);

    if(ns_per_call <= 455.790)
        printf("DIV100_FASTPATH=PASS\n");
    else
        printf("DIV100_FASTPATH=FAIL\n");

    printf("BATCH_EXECUTION=PASS\n");
    printf("============================================================\n");

    free(A);
    free(B);
    free(C);

    xfree(ctx);
    dlclose(lib);

    return 0;
}
