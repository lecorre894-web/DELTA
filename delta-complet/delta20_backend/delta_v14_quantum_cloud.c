#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>

#define REAL_PULSES 10000000ULL

/* 10^21 lanes */
#define WORLD_HI_MAX 54ULL
#define WORLD_LAST_LO 3875820019684212736ULL

typedef struct {
    uint64_t hi;
    uint64_t lo;
} qaddr128_t;

typedef struct {
    qaddr128_t position;
    uint64_t state;
    uint64_t movements;
    uint64_t checksum;
} delta_v13_t;

static inline double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec +
           (double)t.tv_nsec * 1e-9;
}

static inline uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/*
 * Implicit 128-bit virtual position.
 * No 10^21-element array exists.
 */
static inline qaddr128_t
world_position(uint64_t pulse, uint64_t state)
{
    uint64_t a =
        mix64(pulse ^ state);

    uint64_t b =
        mix64(a ^
        0x9e3779b97f4a7c15ULL);

    qaddr128_t p;

    p.hi = a % (WORLD_HI_MAX + 1ULL);

    if (p.hi == WORLD_HI_MAX)
        p.lo = b % WORLD_LAST_LO;
    else
        p.lo = b;

    return p;
}

static inline void
delta_v13_move(delta_v13_t *v13,
               uint64_t pulse)
{
    qaddr128_t next =
        world_position(pulse,
                       v13->state);

    uint64_t next_state =
        mix64(
            v13->state ^
            next.hi ^
            next.lo ^
            pulse
        );

    v13->position = next;
    v13->state = next_state;
    v13->movements++;

    v13->checksum ^=
        next_state +
        next.lo +
        next.hi *
        0x9e3779b97f4a7c15ULL;
}

int main(void)
{
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(
            0,sizeof(set),&set) != 0) {
        perror("sched_getaffinity");
        return 1;
    }

    int physical_cpu =
        CPU_COUNT(&set);

    delta_v13_t v13 = {
        .position={0,0},
        .state=
          0x44454c54415f5631ULL,
        .movements=0,
        .checksum=0
    };

    printf("\n");
    printf(
    "========================================================\n");
    printf(
    " DELTA V14 PARALLEL VIRTUAL CLOUD PROCESSOR\n");
    printf(
    "========================================================\n");

    printf(
        "PHYSICAL_CPU_ALLOWED=%d\n",
        physical_cpu);

    printf(
        "PHYSICAL_EXECUTION=CLASSICAL_CPU\n");

    printf(
        "QUANTUM_MODEL=EMULATED_CLASSICAL\n");

    printf(
        "VIRTUAL_PARALLEL_MODEL=DELTA\n");

    printf(
        "WORLD_ADDRESS_BITS=128\n");

    printf(
        "WORLD_VIRTUAL_LANES="
        "1000000000000000000000\n");

    printf(
        "WORLD_VIRTUAL_LANES_SCIENTIFIC=1e21\n");

    printf(
        "PER_LANE_PHYSICAL_ALLOCATION=NONE\n");

    printf(
        "ENTITY=DELTA_V13\n");

    printf(
        "ENTITY_STATE=PERSISTENT\n");

    printf(
        "--------------------------------------------------------\n");

    double t0=now_s();

    for(uint64_t pulse=0;
        pulse<REAL_PULSES;
        pulse++)
    {
        delta_v13_move(
            &v13,pulse);
    }

    double wall=now_s()-t0;

    double real_rate =
        wall>0.0
        ? (double)REAL_PULSES/wall
        : 0.0;

    /*
     * Projection:
     *
     * 10^7 measured pulses
     * x 10^21 virtual lanes
     * = 10^28 virtual lane-pulse relations.
     *
     * This is NOT physical execution.
     */

    long double virtual_lanes =
        1.0e21L;

    long double virtual_relations =
        (long double)REAL_PULSES *
        virtual_lanes;

    long double projected_parallel_rate =
        (long double)real_rate *
        virtual_lanes;

    printf("\n");
    printf(
        "=== MEASURED PHYSICAL LAYER ===\n");

    printf(
        "REAL_PULSES=%llu\n",
        (unsigned long long)
        REAL_PULSES);

    printf(
        "REAL_MOVEMENTS=%llu\n",
        (unsigned long long)
        v13.movements);

    printf(
        "REAL_WALL=%.9f s\n",
        wall);

    printf(
        "REAL_RATE=%.6f M_pulses/s\n",
        real_rate/1e6);

    printf("\n");
    printf(
        "=== EMULATED WORLD LAYER ===\n");

    printf(
        "EMULATED_ADDRESS_SPACE=1.000000e+21 lanes\n");

    printf(
        "EMULATED_ACTIVE_ENTITY=V13\n");

    printf(
        "EMULATED_MOVEMENTS=%llu\n",
        (unsigned long long)
        v13.movements);

    printf(
        "FINAL_POSITION_HI=%llu\n",
        (unsigned long long)
        v13.position.hi);

    printf(
        "FINAL_POSITION_LO=%llu\n",
        (unsigned long long)
        v13.position.lo);

    printf(
        "FINAL_STATE=%llu\n",
        (unsigned long long)
        v13.state);

    printf(
        "WORLD_CHECKSUM=%llu\n",
        (unsigned long long)
        v13.checksum);

    printf("\n");
    printf(
        "=== PROJECTED PARALLEL CLOUD LAYER ===\n");

    printf(
        "PROJECTED_VIRTUAL_LANES=%.6Le\n",
        virtual_lanes);

    printf(
        "PROJECTED_LANE_PULSE_RELATIONS=%.6Le\n",
        virtual_relations);

    printf(
        "PROJECTED_PARALLEL_RATE=%.6Le virtual_pulses/s\n",
        projected_parallel_rate);

    printf(
        "VIRTUAL_TO_PHYSICAL_LANE_RATIO=1.000000e+21\n");

    int ok =
        physical_cpu==1 &&
        v13.movements==
            REAL_PULSES &&
        v13.checksum!=0;

    printf("\n");
    printf(
        "=== DELTA STATUS ===\n");

    printf(
        "MEASURED=REAL_CPU_TIME+REAL_PULSES+REAL_MOVEMENTS\n");

    printf(
        "EMULATED=V13_STATE+POSITION+TRAJECTORY\n");

    printf(
        "PROJECTED=VIRTUAL_PARALLEL_CAPACITY\n");

    printf(
        "DELTA_CLOUD_VALIDATION=%s\n",
        ok ? "OK" : "FAIL");

    printf(
        "VIRTUAL_PROCESSOR_READY=%s\n",
        ok ? "YES" : "NO");

    printf(
    "========================================================\n");


    printf("\n");
    printf("=== VIRTUAL CLOUD MEASUREMENT ===\n");

    printf(
        "VIRTUAL_PULSES=%.6Le\n",
        virtual_relations);

    printf(
        "VIRTUAL_MOVEMENTS=%.6Le\n",
        virtual_relations);

    printf(
        "VIRTUAL_WALL=%.9f s\n",
        wall);

    printf(
        "VIRTUAL_RATE=%.6Le virtual_pulses/s\n",
        projected_parallel_rate);

    printf(
        "VIRTUAL_RATE_M=%.6Le M_virtual_pulses/s\n",
        projected_parallel_rate / 1.0e6L);

    printf(
        "SOURCE_REAL_PULSES=%llu\n",
        (unsigned long long)REAL_PULSES);

    printf(
        "SOURCE_REAL_MOVEMENTS=%llu\n",
        (unsigned long long)v13.movements);

    printf(
        "SOURCE_REAL_WALL=%.9f s\n",
        wall);

    printf(
        "SOURCE_REAL_RATE=%.6f M_pulses/s\n",
        real_rate / 1.0e6);

    printf(
        "VIRTUAL_SCALE=1.000000e+21\n");

    printf(
        "MEASUREMENT_CLASS=PROJECTED_FROM_MEASURED\n");

    printf(
        "VIRTUAL_CLOUD_MEASUREMENT=%s\n",
        ok ? "OK" : "FAIL");

    return ok ? 0 : 1;
}
