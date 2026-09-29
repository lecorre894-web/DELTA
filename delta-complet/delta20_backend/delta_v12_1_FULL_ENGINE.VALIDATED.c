#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NQ 10000000u
#define CPU_WORKERS 32u
#define DSPC_CAP 1024u
#define BATCH 2048u
#define CACHE_LEVELS 200u
#define CACHE_LINES 256u
#define PASSES 5u

/* Start with the last validated persistent split.
   The FULL engine will profile all layers before any new ratio sweep. */
#define GPU_EVENTS 1200000u
#define CPU_FIRST GPU_EVENTS
#define CPU_EVENTS (NQ-GPU_EVENTS)

typedef struct {
    uint32_t id, version;
    float a, b;
    uint32_t first_edge;
} node_t;

typedef struct {
    uint32_t first, count;
} context_t;

typedef struct {
    uint64_t contexts, events;
} worker_stat_t;

typedef struct {
    uint32_t tag;
    uint32_t stamp;
} cache_line_t;

typedef struct {
    cache_line_t line[CACHE_LINES];
} cache_level_t;

typedef struct {
    uint64_t read, write, hit, miss, fill, evict;
    uint64_t contexts, events;
} cache_stat_t;

typedef struct {
    context_t ring[DSPC_CAP];
    uint32_t head, tail, count;
    int closed;
    pthread_mutex_t mu;
    pthread_cond_t not_empty, not_full;
} dspc_t;

static node_t *nodes;
static dspc_t dspc;
static worker_stat_t wstat[CPU_WORKERS];
static cache_level_t cache_level[CACHE_LEVELS];
/* Thread-local telemetry: one private counter bank per CPU worker and cache level.
   Reduced deterministically after pthread_join; no shared counter increments
   occur in the hot path. */
static cache_stat_t cache_stat[CPU_WORKERS][CACHE_LEVELS];
static atomic_uint ready_workers;
static atomic_int start_gate;
static atomic_ullong dspc_routed, dspc_returned;

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec/1e9;
}
static void check(cl_int e, const char *m) {
    if(e != CL_SUCCESS) { fprintf(stderr,"%s FAIL: %d\n",m,e); exit(1); }
}

/* Selective software-cache policy:
   every CPU context is routed through exactly one L1..L200 level.
   This keeps all 200 real software cache structures active without
   repeating the rejected 200-lookups-per-event path. */
static inline unsigned cache_for_context(uint64_t context_id) {
    if(context_id < CACHE_LEVELS) return (unsigned)context_id;
    return (unsigned)(context_id % CACHE_LEVELS);
}
static inline void cache_touch(unsigned wid, unsigned level, uint32_t q, int write) {
    cache_level_t *c=&cache_level[level];
    cache_stat_t *st=&cache_stat[wid][level];
    uint32_t idx=q & (CACHE_LINES-1u);
    uint32_t tag=q / CACHE_LINES;
    cache_line_t *l=&c->line[idx];

    st->read++;
    if(write) st->write++;

    /* Cache data remain shared by design; telemetry is private.
       The cache is advisory and never determines EVENT-STATE correctness. */
    uint32_t old_stamp=__atomic_load_n(&l->stamp,__ATOMIC_RELAXED);
    uint32_t old_tag=__atomic_load_n(&l->tag,__ATOMIC_RELAXED);
    if(old_stamp && old_tag==tag) {
        st->hit++;
    } else {
        st->miss++;
        if(old_stamp) st->evict++;
        st->fill++;
        __atomic_store_n(&l->tag,tag,__ATOMIC_RELAXED);
        __atomic_store_n(&l->stamp,1u,__ATOMIC_RELAXED);
    }
}
static void dspc_init(void) {
    memset(&dspc,0,sizeof(dspc));
    pthread_mutex_init(&dspc.mu,NULL);
    pthread_cond_init(&dspc.not_empty,NULL);
    pthread_cond_init(&dspc.not_full,NULL);
}
static void dspc_push(context_t c) {
    pthread_mutex_lock(&dspc.mu);
    while(dspc.count==DSPC_CAP) pthread_cond_wait(&dspc.not_full,&dspc.mu);
    dspc.ring[dspc.tail]=c;
    dspc.tail=(dspc.tail+1u)%DSPC_CAP;
    dspc.count++;
    pthread_cond_signal(&dspc.not_empty);
    pthread_mutex_unlock(&dspc.mu);
    atomic_fetch_add_explicit(&dspc_routed,1,memory_order_relaxed);
}
static int dspc_pop(context_t *c) {
    pthread_mutex_lock(&dspc.mu);
    while(!dspc.count && !dspc.closed) pthread_cond_wait(&dspc.not_empty,&dspc.mu);
    if(!dspc.count && dspc.closed) { pthread_mutex_unlock(&dspc.mu); return 0; }
    *c=dspc.ring[dspc.head];
    dspc.head=(dspc.head+1u)%DSPC_CAP;
    dspc.count--;
    pthread_cond_signal(&dspc.not_full);
    pthread_mutex_unlock(&dspc.mu);
    return 1;
}
static void dspc_close(void) {
    pthread_mutex_lock(&dspc.mu);
    dspc.closed=1;
    pthread_cond_broadcast(&dspc.not_empty);
    pthread_mutex_unlock(&dspc.mu);
}

static void *cpu_worker(void *arg) {
    unsigned wid=(unsigned)(uintptr_t)arg;
    atomic_fetch_add_explicit(&ready_workers,1,memory_order_release);
    while(!atomic_load_explicit(&start_gate,memory_order_acquire))
        __asm__ volatile("yield");
    context_t c;
    while(dspc_pop(&c)) {
        uint64_t cid=atomic_load_explicit(&dspc_returned,memory_order_relaxed)
                   + wstat[wid].contexts;
        unsigned lev=cache_for_context(cid);
        cache_stat[wid][lev].contexts++;
        cache_stat[wid][lev].events += c.count;
        for(uint32_t k=0;k<c.count;k++) {
            uint32_t q=c.first+k;
            cache_touch(wid,lev,q,1);
            nodes[q].a += 0.000001f;
            nodes[q].b -= 0.000001f;
            nodes[q].version++;
            wstat[wid].events++;
        }
        wstat[wid].contexts++;
        atomic_fetch_add_explicit(&dspc_returned,1,memory_order_relaxed);
    }
    return NULL;
}

static const char *kernel_src =
"typedef struct { uint id; uint version; float a; float b; uint first_edge; } node_t;\n"
"__kernel void v10(__global node_t *nodes) {\n"
" size_t q=get_global_id(0);\n"
" if(q >= 1200000UL) return;\n"
" nodes[q].a += 0.000001f;\n"
" nodes[q].b -= 0.000001f;\n"
" nodes[q].version++;\n"
"}\n";

int main(void) {
    if(sizeof(node_t)!=20) { fprintf(stderr,"node_t=%zu expected=20\n",sizeof(node_t)); return 1; }
    printf("=== DELTA V12.1 FULL ENGINE TEST ===\n");
    printf("10M EVENT-STATE + DSPC + L1-L200 selective + C1-C32 + persistent OpenCL\n");

    size_t bytes=(size_t)NQ*sizeof(node_t);
    size_t gpu_bytes=(size_t)GPU_EVENTS*sizeof(node_t);
    nodes=calloc(NQ,sizeof(*nodes));
    if(!nodes) return 1;

    cl_int e; cl_uint np=0;
    check(clGetPlatformIDs(0,NULL,&np),"clGetPlatformIDs");
    cl_platform_id *plats=calloc(np,sizeof(*plats));
    check(clGetPlatformIDs(np,plats,NULL),"clGetPlatformIDs list");
    cl_device_id dev=NULL;
    for(cl_uint p=0;p<np && !dev;p++) {
        cl_uint nd=0;
        if(clGetDeviceIDs(plats[p],CL_DEVICE_TYPE_GPU,0,NULL,&nd)==CL_SUCCESS && nd) {
            cl_device_id *ds=calloc(nd,sizeof(*ds));
            check(clGetDeviceIDs(plats[p],CL_DEVICE_TYPE_GPU,nd,ds,NULL),"clGetDeviceIDs");
            dev=ds[0]; free(ds);
        }
    }
    if(!dev) { fprintf(stderr,"No OpenCL GPU\n"); return 1; }
    char name[256]={0}; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,NULL);

    double setup0=now_s();
    cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); check(e,"clCreateContext");
    cl_command_queue queue=clCreateCommandQueueWithProperties(ctx,dev,NULL,&e); check(e,"clCreateCommandQueue");
    cl_program prog=clCreateProgramWithSource(ctx,1,&kernel_src,NULL,&e); check(e,"clCreateProgram");
    e=clBuildProgram(prog,1,&dev,"",NULL,NULL);
    if(e!=CL_SUCCESS) {
        size_t n=0; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&n);
        char *log=calloc(n+1,1); clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,n,log,NULL);
        fprintf(stderr,"%s\n",log); return 1;
    }
    cl_kernel kernel=clCreateKernel(prog,"v10",&e); check(e,"clCreateKernel");
    cl_mem gpu_buf=clCreateBuffer(ctx,CL_MEM_READ_WRITE,gpu_bytes,NULL,&e); check(e,"clCreateBuffer");
    check(clSetKernelArg(kernel,0,sizeof(gpu_buf),&gpu_buf),"clSetKernelArg");
    clFinish(queue);
    double setup_wall=now_s()-setup0;

    printf("GPU               : %s\n",name);
    printf("node state        : %.2f MiB\n",bytes/1048576.0);
    printf("CPU/GPU split     : %u / %u events\n",CPU_EVENTS,GPU_EVENTS);
    printf("CPU workers       : %u persistent per pass\n",CPU_WORKERS);
    printf("DSPC ring         : %u contexts\n",DSPC_CAP);
    printf("CPU batch         : %u events/context\n",BATCH);
    printf("software caches   : L1 -> L%u, %u lines/level, selective\n",CACHE_LEVELS,CACHE_LINES);
    printf("OpenCL setup wall : %.6f s\n",setup_wall);

    int all_ok=1;
    double sum_wall=0.0, best_wall=1e99;

    for(unsigned pass=0;pass<PASSES;pass++) {
        for(uint32_t i=0;i<NQ;i++) {
            nodes[i].id=i; nodes[i].version=0; nodes[i].a=1.0f; nodes[i].b=0.0f;
            nodes[i].first_edge=UINT32_MAX;
        }
        memset(wstat,0,sizeof(wstat));
        memset(cache_level,0,sizeof(cache_level));
        memset(cache_stat,0,sizeof(cache_stat));
        atomic_store(&ready_workers,0); atomic_store(&start_gate,0);
        atomic_store(&dspc_routed,0); atomic_store(&dspc_returned,0);
        dspc_init();

        check(clEnqueueWriteBuffer(queue,gpu_buf,CL_TRUE,0,gpu_bytes,nodes,0,NULL,NULL),
              "clEnqueueWriteBuffer reset");

        pthread_t th[CPU_WORKERS];
        for(unsigned w=0;w<CPU_WORKERS;w++) {
            int r=pthread_create(&th[w],NULL,cpu_worker,(void*)(uintptr_t)w);
            if(r) { fprintf(stderr,"pthread_create C%u FAIL=%d\n",w+1,r); return 1; }
        }
        while(atomic_load_explicit(&ready_workers,memory_order_acquire)!=CPU_WORKERS)
            __asm__ volatile("yield");

        size_t global=GPU_EVENTS;
        double t0=now_s();
        atomic_store_explicit(&start_gate,1,memory_order_release);

        double gpu0=now_s();
        check(clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global,NULL,0,NULL,NULL),
              "clEnqueueNDRangeKernel");

        uint64_t contexts_sent=0;
        for(uint32_t first=CPU_FIRST; first<NQ; first+=BATCH) {
            uint32_t count=(NQ-first<BATCH)?(NQ-first):BATCH;
            dspc_push((context_t){first,count});
            contexts_sent++;
        }
        dspc_close();

        clFinish(queue);
        double gpu_wall=now_s()-gpu0;

        for(unsigned w=0;w<CPU_WORKERS;w++) pthread_join(th[w],NULL);
        double hybrid_wall=now_s()-t0;

        double read0=now_s();
        check(clEnqueueReadBuffer(queue,gpu_buf,CL_TRUE,0,gpu_bytes,nodes,0,NULL,NULL),
              "clEnqueueReadBuffer");
        double read_wall=now_s()-read0;

        const uint32_t probe[3]={7u,900000u,9999999u};
        for(int p=0;p<3;p++) {
            node_t *n=&nodes[probe[p]];
            n->a+=0.000001f; n->b-=0.000001f; n->version++;
        }

        uint64_t active=0,mismatch=0,checksum=0;
        for(uint32_t q=0;q<NQ;q++) {
            uint32_t expect=(q==7u || q==900000u || q==9999999u)?2u:1u;
            if(nodes[q].version) active++;
            if(nodes[q].version!=expect) mismatch++;
            checksum+=(uint64_t)q+nodes[q].version;
        }

        uint64_t cpu_events=0,cpu_contexts=0,active_workers=0;
        for(unsigned w=0;w<CPU_WORKERS;w++) {
            cpu_events+=wstat[w].events; cpu_contexts+=wstat[w].contexts;
            if(wstat[w].events) active_workers++;
        }
        uint64_t levels_active=0,cache_events=0,hits=0,miss=0,fills=0,evict=0;
        uint64_t cache_reads=0,cache_writes=0,cache_contexts=0;
        for(unsigned l=0;l<CACHE_LEVELS;l++) {
            uint64_t lev_events=0;
            for(unsigned w=0;w<CPU_WORKERS;w++) {
                cache_stat_t *st=&cache_stat[w][l];
                cache_contexts+=st->contexts;
                lev_events+=st->events;
                cache_reads+=st->read;
                cache_writes+=st->write;
                hits+=st->hit;
                miss+=st->miss;
                fills+=st->fill;
                evict+=st->evict;
            }
            if(lev_events) levels_active++;
            cache_events+=lev_events;
        }
        uint64_t routed=atomic_load(&dspc_routed), returned=atomic_load(&dspc_returned);
        uint64_t epochs=(uint64_t)NQ+3ULL;

        int ok=
            cpu_events==CPU_EVENTS &&
            routed==contexts_sent && returned==contexts_sent &&
            cpu_contexts==contexts_sent &&
            active_workers==CPU_WORKERS &&
            levels_active==CACHE_LEVELS &&
            cache_contexts==contexts_sent &&
            cache_events==CPU_EVENTS &&
            cache_reads==CPU_EVENTS && cache_writes==CPU_EVENTS &&
            active==NQ && mismatch==0 &&
            epochs==10000003ULL &&
            checksum==50000005000003ULL &&
            nodes[7].version==2 && nodes[900000].version==2 && nodes[9999999].version==2;

        printf("\n=== FULL PASS %u/%u ===\n",pass+1,PASSES);
        printf("DSPC routed/return: %llu / %llu\n",(unsigned long long)routed,(unsigned long long)returned);
        printf("CPU contexts      : %llu\n",(unsigned long long)cpu_contexts);
        printf("C1-C32 active     : %llu / %u\n",(unsigned long long)active_workers,CPU_WORKERS);
        printf("L1-L200 active    : %llu / %u\n",(unsigned long long)levels_active,CACHE_LEVELS);
        printf("cache contexts    : %llu\n",(unsigned long long)cache_contexts);
        printf("cache events      : %llu\n",(unsigned long long)cache_events);
        printf("cache read/write  : %llu / %llu\n",(unsigned long long)cache_reads,(unsigned long long)cache_writes);
        printf("cache hit/miss    : %llu / %llu\n",(unsigned long long)hits,(unsigned long long)miss);
        printf("cache fill/evict  : %llu / %llu\n",(unsigned long long)fills,(unsigned long long)evict);
        printf("CPU events        : %llu\n",(unsigned long long)cpu_events);
        printf("GPU events        : %u\n",GPU_EVENTS);
        printf("GPU compute wall  : %.6f s\n",gpu_wall);
        printf("HYBRID wall       : %.6f s\n",hybrid_wall);
        printf("HYBRID rate       : %.3f M events/s\n",NQ/hybrid_wall/1e6);
        printf("readback wall     : %.6f s\n",read_wall);
        printf("nodes active      : %llu / %u\n",(unsigned long long)active,NQ);
        printf("version mismatches: %llu\n",(unsigned long long)mismatch);
        printf("epochs            : %llu\n",(unsigned long long)epochs);
        printf("checksum          : %llu\n",(unsigned long long)checksum);
        printf("probe versions    : %u / %u / %u\n",nodes[7].version,nodes[900000].version,nodes[9999999].version);
        printf("PASS VALIDATION   : %s\n",ok?"OK":"FAIL");

        sum_wall+=hybrid_wall;
        if(hybrid_wall<best_wall) best_wall=hybrid_wall;
        if(!ok) all_ok=0;

        pthread_mutex_destroy(&dspc.mu);
        pthread_cond_destroy(&dspc.not_empty);
        pthread_cond_destroy(&dspc.not_full);
    }

    printf("\n=== V12.1 FULL ENGINE SUMMARY ===\n");
    printf("passes            : %u\n",PASSES);
    printf("mean HYBRID wall  : %.6f s\n",sum_wall/PASSES);
    printf("mean rate         : %.3f M events/s\n",NQ/(sum_wall/PASSES)/1e6);
    printf("best HYBRID wall  : %.6f s\n",best_wall);
    printf("best rate         : %.3f M events/s\n",NQ/best_wall/1e6);
    printf("DELTA V12.1 FULL ENGINE VALIDATION: %s\n",all_ok?"OK":"FAIL");

    clReleaseMemObject(gpu_buf); clReleaseKernel(kernel); clReleaseProgram(prog);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    free(plats); free(nodes);
    return all_ok?0:1;
}
