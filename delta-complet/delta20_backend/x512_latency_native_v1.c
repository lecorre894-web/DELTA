#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <dlfcn.h>
#include <sched.h>
#include <string.h>

#define NW      (16384 * 4)
#define WARMUP  1000
#define LOOPS   100000

typedef void *(*fn_new_t)(void);
typedef void  (*fn_free_t)(void *);
typedef uint64_t (*fn_run_t)(
    void *,
    int,
    const uint64_t *,
    const uint64_t *,
    const uint64_t *,
    long,
    long *
);

static inline uint64_t rng64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    return
        (uint64_t)ts.tv_sec * 1000000000ULL +
        (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;

    return (x > y) - (x < y);
}

int main(void)
{
    void *lib = dlopen("./libdelta_x512.so", RTLD_NOW);

    if (!lib) {
        fprintf(stderr, "dlopen: %s\n", dlerror());
        return 1;
    }

    fn_new_t  xnew  = (fn_new_t)dlsym(lib, "x512_new");
    fn_free_t xfree = (fn_free_t)dlsym(lib, "x512_free");
    fn_run_t  xrun  = (fn_run_t)dlsym(lib, "x512_run");

    if (!xnew || !xfree || !xrun) {
        fprintf(stderr, "X512 ABI symbols missing\n");
        return 1;
    }

    uint64_t *A = NULL;
    uint64_t *B = NULL;
    uint64_t *C = NULL;

    if (posix_memalign((void **)&A, 64, NW*sizeof(uint64_t)) ||
        posix_memalign((void **)&B, 64, NW*sizeof(uint64_t)) ||
        posix_memalign((void **)&C, 64, NW*sizeof(uint64_t))) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    uint64_t seed = 894;

    for (long i=0; i<NW; i++) {
        A[i]=rng64(&seed);
        B[i]=rng64(&seed);
        C[i]=rng64(&seed);
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(30, &set);

    int pinned =
        sched_setaffinity(0, sizeof(set), &set) == 0;

    void *ctx=xnew();

    if (!ctx) {
        fprintf(stderr, "x512_new failed\n");
        return 1;
    }

    long ni=0;

    volatile uint64_t sink=0;

    /*
     * Warm-up outside measurement.
     * Change one input word so each invocation has observable
     * changing input/output state.
     */
    for (int i=0; i<WARMUP; i++) {
        C[0] ^= (uint64_t)i + 1ULL;

        sink ^= xrun(
            ctx,
            1,              /* micro */
            A,B,C,
            NW/8,
            &ni
        );
    }

    uint64_t *lat =
        malloc((size_t)LOOPS*sizeof(uint64_t));

    if (!lat) {
        fprintf(stderr, "latency allocation failed\n");
        return 1;
    }

    uint64_t total_ni=0;

    uint64_t global_start=ns_now();

    for (int i=0; i<LOOPS; i++) {

        /*
         * Prevent a constant-input benchmark while retaining
         * exactly the same amount of X512 work.
         */
        C[0] ^= (uint64_t)i + 0x10001ULL;

        uint64_t t0=ns_now();

        uint64_t r=xrun(
            ctx,
            1,
            A,B,C,
            NW/8,
            &ni
        );

        uint64_t t1=ns_now();

        lat[i]=t1-t0;

        total_ni += (uint64_t)ni;

        /*
         * Observable accumulation prevents dead-result
         * elimination at benchmark level.
         */
        sink ^= r + (uint64_t)i;
    }

    uint64_t global_end=ns_now();

    qsort(
        lat,
        LOOPS,
        sizeof(uint64_t),
        cmp_u64
    );

    long double sum=0.0L;

    for (int i=0; i<LOOPS; i++)
        sum += (long double)lat[i];

    double mean_ns =
        (double)(sum/(long double)LOOPS);

    double min_ns =
        (double)lat[0];

    double median_ns =
        (double)lat[LOOPS/2];

    double p95_ns =
        (double)lat[(LOOPS*95)/100];

    double p99_ns =
        (double)lat[(LOOPS*99)/100];

    double max_ns =
        (double)lat[LOOPS-1];

    double wall_s =
        (global_end-global_start)/1e9;

    double amortized_ns =
        (double)(global_end-global_start) /
        (double)LOOPS;

    double ginstr =
        ((double)total_ni / wall_s) / 1e9;

    printf("============================================================\n");
    printf("DELTA X512 — NATIVE LATENCY V1\n");
    printf("============================================================\n");

    printf("CPU_PIN=30\n");
    printf("PINNING=%s\n", pinned ? "PASS" : "FAIL");

    printf("WORDS=%d\n", NW);
    printf("WARMUP=%d\n", WARMUP);
    printf("SAMPLES=%d\n", LOOPS);

    printf("\n");

    printf("MIN_NS=%.1f\n", min_ns);
    printf("MEDIAN_NS=%.1f\n", median_ns);
    printf("MEAN_NS=%.1f\n", mean_ns);
    printf("P95_NS=%.1f\n", p95_ns);
    printf("P99_NS=%.1f\n", p99_ns);
    printf("MAX_NS=%.1f\n", max_ns);

    printf("\n");

    printf("MEAN_US=%.6f\n", mean_ns/1000.0);
    printf("AMORTIZED_NS_PER_TPOP=%.1f\n", amortized_ns);
    printf("AMORTIZED_US_PER_TPOP=%.6f\n", amortized_ns/1000.0);

    printf("NI_PER_CALL=%ld\n", ni);
    printf("TOTAL_NI=%llu\n",
        (unsigned long long)total_ni);

    printf("X512_GINSTR_S=%.6f\n", ginstr);

    printf("SINK=%016llx\n",
        (unsigned long long)sink);

    printf("\n");

    printf("FASTPATH_PREVIOUS_US=45.579\n");
    printf("DIV100_TARGET_US=0.455790\n");
    printf("ISOLATED_PREVIOUS_US=15.572\n");
    printf("ISOLATED_DIV100_TARGET_US=0.155720\n");

    printf("\n");

    if (amortized_ns <= 455.790)
        printf("FASTPATH_DIV100_THRESHOLD=PASS\n");
    else
        printf("FASTPATH_DIV100_THRESHOLD=FAIL\n");

    if (amortized_ns <= 155.720)
        printf("ISOLATED_DIV100_THRESHOLD=PASS\n");
    else
        printf("ISOLATED_DIV100_THRESHOLD=FAIL\n");

    printf("NATIVE_EXECUTION=PASS\n");
    printf("============================================================\n");

    free(lat);
    free(A);
    free(B);
    free(C);

    xfree(ctx);
    dlclose(lib);

    return 0;
}
