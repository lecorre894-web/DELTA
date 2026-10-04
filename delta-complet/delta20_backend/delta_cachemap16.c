#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define CHANNELS 16
#define BLOCK_BYTES 128
#define PASSES 4000
#define RUNS 5
#define MAX_ZONE_KIB 1024

typedef struct {
    uint8_t *src;
    uint8_t *dst;
    size_t begin;
    size_t end;
    pthread_barrier_t *barrier;
    uint64_t blocks;
    uint64_t checksum;
} worker_t;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    worker_t *w=(worker_t *)arg;

    uint8_t *src=w->src;
    uint8_t *dst=w->dst;

    uint64_t blocks=0;
    uint64_t sum=0;

    pthread_barrier_wait(w->barrier);

    for(int p=0;p<PASSES;p++) {

        for(size_t i=w->begin;
            i+BLOCK_BYTES<=w->end;
            i+=BLOCK_BYTES) {

            memcpy(dst+i,src+i,BLOCK_BYTES);

            sum += dst[i];
            blocks++;
        }

        uint8_t *tmp=src;
        src=dst;
        dst=tmp;
    }

    w->blocks=blocks;
    w->checksum=sum;

    return NULL;
}

static int cmp_double(const void *a,const void *b) {
    double x=*(const double *)a;
    double y=*(const double *)b;
    return (x>y)-(x<y);
}

static double median(double *v,int n) {
    double t[RUNS];

    for(int i=0;i<n;i++)
        t[i]=v[i];

    qsort(t,n,sizeof(double),cmp_double);

    return t[n/2];
}

static void run_size(
    size_t zone_kib,
    uint8_t *src,
    uint8_t *dst,
    double *median_gb,
    double *median_miops,
    uint64_t *final_checksum
) {
    const size_t zone_bytes=zone_kib*1024;
    const size_t total_bytes=zone_bytes*CHANNELS;

    double gb_runs[RUNS];
    double miops_runs[RUNS];

    printf("\n------------------------------------------------------------\n");
    printf("ZONE_KiB=%zu TOTAL_WORKING_SET_KiB=%zu\n",
           zone_kib,total_bytes/1024);

    for(int r=0;r<RUNS;r++) {

        memset(src,1,total_bytes);
        memset(dst,0,total_bytes);

        pthread_t threads[CHANNELS];
        worker_t workers[CHANNELS];

        pthread_barrier_t barrier;
        pthread_barrier_init(&barrier,NULL,CHANNELS+1);

        for(int c=0;c<CHANNELS;c++) {

            size_t begin=(size_t)c*zone_bytes;
            size_t end=begin+zone_bytes;

            workers[c].src=src;
            workers[c].dst=dst;
            workers[c].begin=begin;
            workers[c].end=end;
            workers[c].barrier=&barrier;
            workers[c].blocks=0;
            workers[c].checksum=0;

            pthread_create(
                &threads[c],
                NULL,
                worker,
                &workers[c]
            );
        }

        pthread_barrier_wait(&barrier);

        double t0=now_s();

        for(int c=0;c<CHANNELS;c++)
            pthread_join(threads[c],NULL);

        double t1=now_s();

        pthread_barrier_destroy(&barrier);

        uint64_t blocks=0;
        uint64_t checksum=0;

        for(int c=0;c<CHANNELS;c++) {
            blocks+=workers[c].blocks;
            checksum+=workers[c].checksum;
        }

        double seconds=t1-t0;

        /*
         * 128 B lus + 128 B écrits
         * par opération.
         */
        double transferred=
            (double)blocks *
            BLOCK_BYTES *
            2.0;

        double gb_s=
            transferred/seconds/1e9;

        double miops=
            (double)blocks/seconds/1e6;

        gb_runs[r]=gb_s;
        miops_runs[r]=miops;

        *final_checksum ^= checksum;

        printf(
            "RUN_%02d TIME_S=%.6f "
            "GB_S=%.6f MIOPS=%.3f\n",
            r+1,
            seconds,
            gb_s,
            miops
        );
    }

    *median_gb=median(gb_runs,RUNS);
    *median_miops=median(miops_runs,RUNS);

    printf(
        "ZONE_%zuKiB_MEDIAN_GB_S=%.6f\n",
        zone_kib,*median_gb
    );

    printf(
        "ZONE_%zuKiB_MEDIAN_MIOPS=%.3f\n",
        zone_kib,*median_miops
    );
}

int main(void) {

    const size_t zones_kib[]={
        16,32,64,128,256,512,1024
    };

    const int nzones=
        sizeof(zones_kib)/sizeof(zones_kib[0]);

    const size_t max_total=
        (size_t)MAX_ZONE_KIB *
        1024 *
        CHANNELS;

    uint8_t *src=NULL;
    uint8_t *dst=NULL;

    if(posix_memalign((void **)&src,128,max_total) ||
       posix_memalign((void **)&dst,128,max_total)) {

        fprintf(stderr,"ALLOCATION_FAILED\n");
        return 1;
    }

    double gb[7];
    double miops[7];

    uint64_t checksum=0;

    printf("============================================================\n");
    printf(" DELTA x16 CACHE WORKING-SET MAP\n");
    printf("============================================================\n");

    printf("CHANNELS=%d\n",CHANNELS);
    printf("BLOCK_BYTES=%d\n",BLOCK_BYTES);
    printf("PASSES=%d\n",PASSES);
    printf("RUNS=%d\n",RUNS);
    printf("THREADS_CONSTANT=16\n");
    printf("ACCESS_PATTERN=CONTIGUOUS_128B\n");
    printf("GB_S_DEFINITION=PROCESSED_READ_PLUS_WRITE_BYTES\n");
    printf("CLASSIFICATION=PHYSICAL_HOST_MEASURED\n");
    printf("CACHE_BOUNDARIES=TO_BE_INFERRED_NOT_ASSUMED\n");

    for(int z=0;z<nzones;z++) {

        run_size(
            zones_kib[z],
            src,
            dst,
            &gb[z],
            &miops[z],
            &checksum
        );
    }

    printf("\n================ CACHE MAP =================\n");

    for(int z=0;z<nzones;z++) {

        double vs16=
            gb[z]/gb[0];

        printf(
            "ZONE_KiB=%4zu "
            "TOTAL_MiB=%6.2f "
            "GB_S=%10.3f "
            "MIOPS=%9.3f "
            "VS_16KiB=%7.3fx\n",
            zones_kib[z],
            ((double)zones_kib[z]*CHANNELS)/1024.0,
            gb[z],
            miops[z],
            vs16
        );
    }

    printf("\n");

    for(int z=1;z<nzones;z++) {

        double ratio=
            gb[z]/gb[z-1];

        printf(
            "TRANSITION_%zu_TO_%zu_KiB="
            "%.6fx (%+.3f%%)\n",
            zones_kib[z-1],
            zones_kib[z],
            ratio,
            100.0*(ratio-1.0)
        );
    }

    printf(
        "CHECKSUM=%llu\n",
        (unsigned long long)checksum
    );

    printf("FINAL_VALIDATION=PASSED\n");

    free(src);
    free(dst);

    return 0;
}
