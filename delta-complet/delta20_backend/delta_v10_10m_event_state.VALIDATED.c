#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <time.h>

/*
 DELTA V10-10M — registre distribue evenementiel, prototype structurel.
 10 000 000 identifiants logiques; aucun vecteur d'etat 2^N.
 Ce programme ne simule pas un etat quantique arbitraire complet.
*/
#define NQ 10000000u
#define EVENTS 1000000u
#define SHARDS 1024u
#define EMPTY 0xffffffffu

typedef struct {
    uint32_t id;
    uint32_t version;
    float a, b;              /* etat local compact experimental */
    uint32_t first_edge;
} node_t;

typedef struct {
    uint32_t peer;
    uint32_t next;
    float weight;
} edge_t;

typedef struct {
    uint32_t q;
    float da, db;
    uint32_t expected_version;
} event_t;

static node_t *nodes;
static _Atomic uint64_t shard_epoch[SHARDS];

static uint64_t ns_now(void){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;
}

static inline uint32_t shard_of(uint32_t q){
    return (uint32_t)(((uint64_t)q * 11400714819323198485ull) >> 54) & (SHARDS-1);
}

static inline void apply_event(event_t e){
    node_t *n=&nodes[e.q];
    n->a += e.da;
    n->b += e.db;
    n->version++;
    atomic_fetch_add_explicit(&shard_epoch[shard_of(e.q)],1,memory_order_relaxed);
}

int main(void){
    const size_t bytes=(size_t)NQ*sizeof(node_t);
    nodes=calloc(NQ,sizeof(node_t));
    if(!nodes){ perror("calloc"); return 1; }

    for(uint32_t i=0;i<NQ;i++){
        nodes[i].id=i;
        nodes[i].version=0;
        nodes[i].a=1.0f;
        nodes[i].b=0.0f;
        nodes[i].first_edge=EMPTY;
    }

    printf("DELTA V10-10M EVENT-STATE\n");
    printf("logical nodes       : %u\n",NQ);
    printf("central 2^N vector  : NONE\n");
    printf("shards              : %u\n",SHARDS);
    printf("node bytes          : %zu\n",sizeof(node_t));
    printf("allocated state     : %.2f MiB\n",bytes/1048576.0);

    /* Trois positions tres eloignees: preuve d'adressage sans registre 2^N. */
    uint32_t probe[3]={7u,900000u,9999999u};
    for(int i=0;i<3;i++){
        event_t e={probe[i],0.125f,0.25f,nodes[probe[i]].version};
        apply_event(e);
    }

    uint64_t t0=ns_now();
    uint64_t checksum=0;
    for(uint32_t i=0;i<EVENTS;i++){
        uint32_t q=(uint32_t)(((uint64_t)i*2654435761u)%NQ);
        event_t e={q,0.000001f,-0.000001f,nodes[q].version};
        apply_event(e);
        checksum += (uint64_t)q + nodes[q].version;
    }
    uint64_t t1=ns_now();

    uint64_t epochs=0;
    for(uint32_t s=0;s<SHARDS;s++)
        epochs += atomic_load_explicit(&shard_epoch[s],memory_order_relaxed);

    printf("events              : %u\n",EVENTS+3u);
    printf("event wall          : %.6f s\n",(t1-t0)/1e9);
    printf("event rate          : %.3f M/s\n",EVENTS/((t1-t0)/1e9)/1e6);
    printf("shard epoch total   : %llu\n",(unsigned long long)epochs);
    printf("probe 7             : v=%u a=%.6f b=%.6f shard=%u\n",
           nodes[7].version,nodes[7].a,nodes[7].b,shard_of(7));
    printf("probe 900000        : v=%u a=%.6f b=%.6f shard=%u\n",
           nodes[900000].version,nodes[900000].a,nodes[900000].b,shard_of(900000));
    printf("probe 9999999       : v=%u a=%.6f b=%.6f shard=%u\n",
           nodes[9999999].version,nodes[9999999].a,nodes[9999999].b,shard_of(9999999));
    printf("checksum            : %llu\n",(unsigned long long)checksum);

    int ok = epochs==(uint64_t)EVENTS+3u &&
             nodes[7].version>=1 &&
             nodes[900000].version>=1 &&
             nodes[9999999].version>=1;
    printf("V10 structural test : %s\n",ok?"OK":"FAIL");

    free(nodes);
    return ok?0:2;
}
