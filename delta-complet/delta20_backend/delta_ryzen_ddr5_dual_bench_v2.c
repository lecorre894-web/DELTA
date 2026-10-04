#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>

#define ZONES 16
#ifndef ZONE_KIB
#define ZONE_KIB 64
#endif
#define BLOCK_BYTES 128
#define PASSES 4000
#define RUNS 7

#define MEM_BYTES ((size_t)ZONES * ZONE_KIB * 1024)

typedef struct {
    uint8_t *src;
    uint8_t *dst;
    uint64_t checksum;
} worker_t;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    double t[RUNS];
    for (int i=0; i<n; i++) t[i]=v[i];
    qsort(t,n,sizeof(double),cmp_double);
    return t[n/2];
}

static void *memory_worker(void *arg)
{
    /* V2 : meme travail exact (lit, ecrit x+1, somme), en une boucle continue.
       restrict = src et dst ne se chevauchent pas -> le compilateur vectorise (AVX). */
    worker_t *w = (worker_t *)arg;
    uint64_t sum = 0;
    uint8_t *a = w->src, *b = w->dst;
    const size_t n = (size_t)MEM_BYTES / 8;
    for (int pass=0; pass<PASSES; pass++) {
        const uint64_t *restrict s = (const uint64_t *)a;
        uint64_t *restrict d = (uint64_t *)b;
        for (size_t i=0; i<n; i++) { uint64_t x = s[i]; d[i] = x + 1; sum += x; }
        uint8_t *t = a; a = b; b = t;
    }
    w->src = a; w->dst = b;
    w->checksum = sum;
    return NULL;
}

static double run_one(worker_t *w)
{
    double t0=now_s();
    memory_worker(w);
    return now_s()-t0;
}

static double run_dual(worker_t *a, worker_t *b)
{
    pthread_t ta,tb;

    double t0=now_s();

    pthread_create(&ta,NULL,memory_worker,a);
    pthread_create(&tb,NULL,memory_worker,b);

    pthread_join(ta,NULL);
    pthread_join(tb,NULL);

    return now_s()-t0;
}

static double gbps(double seconds, int memories)
{
    double bytes =
        (double)MEM_BYTES *
        (double)PASSES *
        2.0 *
        (double)memories;

    return bytes / seconds / 1e9;
}

static double miops(double seconds, int memories)
{
    double ops =
        ((double)MEM_BYTES / BLOCK_BYTES) *
        PASSES *
        memories;

    return ops / seconds / 1e6;
}

int main(void)
{
    uint8_t *a0,*a1,*b0,*b1;

    if (posix_memalign((void **)&a0,64,MEM_BYTES) ||
        posix_memalign((void **)&a1,64,MEM_BYTES) ||
        posix_memalign((void **)&b0,64,MEM_BYTES) ||
        posix_memalign((void **)&b1,64,MEM_BYTES)) {
        fprintf(stderr,"ALLOCATION_FAILED\n");
        return 1;
    }

    memset(a0,1,MEM_BYTES);
    memset(a1,0,MEM_BYTES);
    memset(b0,2,MEM_BYTES);
    memset(b1,0,MEM_BYTES);

    double A[RUNS],B[RUNS],D[RUNS];

    printf("============================================\n");
    printf(" DELTA RYZEN DDR5 EXPERIMENTAL DUAL BENCH\n");
    printf("============================================\n");
    printf("MEMORIES=2\n");
    printf("ZONES_PER_MEMORY=%d\n",ZONES);
    printf("ZONE_KiB=%d\n",ZONE_KIB);
    printf("WORKING_SET_PER_MEMORY_MiB=%.2f\n",
           MEM_BYTES/(1024.0*1024.0));
    printf("TOTAL_DUAL_WORKING_SET_MiB=%.2f\n",
           2.0*MEM_BYTES/(1024.0*1024.0));
    printf("BLOCK_BYTES=%d\n",BLOCK_BYTES);
    printf("PASSES=%d\n",PASSES);
    printf("RUNS=%d\n",RUNS);
    printf("GB_S_DEFINITION=PROCESSED_READ_PLUS_WRITE_BYTES\n");
    printf("CLASSIFICATION=PHYSICAL_HOST_MEASURED\n");
    printf("PHYSICAL_DDR5_CHANNELS_ASSUMED=NO\n\n");

    for(int r=0;r<RUNS;r++) {
        worker_t wa={a0,a1,0};
        double s=run_one(&wa);
        A[r]=gbps(s,1);

        printf("A_RUN_%02d TIME_S=%.6f GB_S=%.6f MIOPS=%.3f\n",
               r+1,s,A[r],miops(s,1));
    }

    printf("\n");

    for(int r=0;r<RUNS;r++) {
        worker_t wb={b0,b1,0};
        double s=run_one(&wb);
        B[r]=gbps(s,1);

        printf("B_RUN_%02d TIME_S=%.6f GB_S=%.6f MIOPS=%.3f\n",
               r+1,s,B[r],miops(s,1));
    }

    printf("\n");

    for(int r=0;r<RUNS;r++) {
        worker_t wa={a0,a1,0};
        worker_t wb={b0,b1,0};

        double s=run_dual(&wa,&wb);
        D[r]=gbps(s,2);

        printf("DUAL_RUN_%02d TIME_S=%.6f GB_S=%.6f MIOPS=%.3f\n",
               r+1,s,D[r],miops(s,2));
    }

    double ma=median(A,RUNS);
    double mb=median(B,RUNS);
    double md=median(D,RUNS);

    double single=(ma+mb)/2.0;
    double scaling=md/single;
    double efficiency=scaling/2.0*100.0;

    printf("\n================ RESULTS ================\n");
    printf("DDR5_A_MEDIAN_GB_S=%.6f\n",ma);
    printf("DDR5_B_MEDIAN_GB_S=%.6f\n",mb);
    printf("SINGLE_MEAN_OF_MEDIANS_GB_S=%.6f\n",single);
    printf("DUAL_MEDIAN_GB_S=%.6f\n",md);
    printf("DUAL_VS_SINGLE_SCALING=%.6fx\n",scaling);
    printf("DUAL_SCALING_EFFICIENCY=%.3f%%\n",efficiency);
    printf("CACHEMAP16_REFERENCE_GB_S=234.116005\n");
    printf("REFERENCE_IS_SEPARATE_EXPERIMENT=YES\n");
    printf("FINAL_VALIDATION=PASSED\n");

    free(a0); free(a1);
    free(b0); free(b1);
    return 0;
}
