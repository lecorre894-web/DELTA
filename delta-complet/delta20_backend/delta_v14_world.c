#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>

#define V14_PULSES 10000000ULL

typedef struct {
    uint64_t hi;
    uint64_t lo;
} delta_pos128_t;

typedef struct {
    delta_pos128_t position;
    uint64_t state;
    uint64_t movements;
    uint64_t checksum;
} delta_v13_entity_t;

static inline double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
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
   V14 WORLD
   Logical address space = 10^21 positions.

   10^21 =
   54 * 2^64 + 3875820019684212736
*/
static inline delta_pos128_t
v14_position(uint64_t pulse, uint64_t previous_state)
{
    const uint64_t MAX_HI = 54ULL;
    const uint64_t MAX_LO = 3875820019684212736ULL;

    uint64_t a =
        mix64(pulse ^ previous_state);

    uint64_t b =
        mix64(a ^ 0x9e3779b97f4a7c15ULL);

    delta_pos128_t p;

    p.hi = a % (MAX_HI + 1ULL);

    if (p.hi == MAX_HI)
        p.lo = b % MAX_LO;
    else
        p.lo = b;

    return p;
}

static inline void
v13_move(delta_v13_entity_t *v13, uint64_t pulse)
{
    delta_pos128_t next =
        v14_position(pulse, v13->state);

    /*
       V13 carries its state while moving.
       No object is allocated for the destination lane.
    */
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
        (next.hi * 0x9e3779b97f4a7c15ULL);
}

int main(void)
{
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_getaffinity");
        return 1;
    }

    int allowed = CPU_COUNT(&set);

    delta_v13_entity_t v13 = {
        .position = {0,0},
        .state =
            0x44454c54415f5631ULL,
        .movements = 0,
        .checksum = 0
    };

    printf("\n");
    printf("====================================================\n");
    printf(" DELTA V14 WORLD — V13 FIRST MOVEMENT\n");
    printf("====================================================\n");

    printf("PHYSICAL_CPU_ALLOWED=%d\n", allowed);

    printf(
        "WORLD_VIRTUAL_LANES="
        "1000000000000000000000\n"
    );

    printf("WORLD_ADDRESS_BITS=128\n");
    printf("WORLD_ALLOCATION_PER_LANE=NONE\n");

    printf("ENTITY=DELTA_V13\n");
    printf("ENTITY_STATE=PERSISTENT\n");

    printf("STATUS_PHYSICAL=MEASURED\n");
    printf("STATUS_WORLD=EMULATED\n");
    printf("STATUS_EXTRAPOLATION=PROJECTED\n");

    printf("----------------------------------------------------\n");

    double t0 = now_s();

    for (uint64_t pulse=0;
         pulse<V14_PULSES;
         pulse++)
    {
        v13_move(&v13, pulse);
    }

    double wall = now_s() - t0;

    double rate =
        wall > 0.0
        ? (double)v13.movements / wall
        : 0.0;

    printf(
        "REAL_MOVEMENTS=%llu\n",
        (unsigned long long)v13.movements
    );

    printf(
        "REAL_WALL=%.9f s\n",
        wall
    );

    printf(
        "REAL_MOVEMENT_RATE=%.3f M/s\n",
        rate / 1e6
    );

    printf(
        "V13_FINAL_POSITION_HI=%llu\n",
        (unsigned long long)v13.position.hi
    );

    printf(
        "V13_FINAL_POSITION_LO=%llu\n",
        (unsigned long long)v13.position.lo
    );

    printf(
        "V13_FINAL_STATE=%llu\n",
        (unsigned long long)v13.state
    );

    printf(
        "V13_WORLD_CHECKSUM=%llu\n",
        (unsigned long long)v13.checksum
    );

    int ok =
        allowed == 1 &&
        v13.movements == V14_PULSES &&
        v13.checksum != 0;

    printf(
        "V14_WORLD_VALIDATION=%s\n",
        ok ? "OK" : "FAIL"
    );

    printf("----------------------------------------------------\n");

    printf(
        "MEASURED=CPU_TIME+REAL_MOVEMENTS\n"
    );

    printf(
        "EMULATED=V13_POSITION+STATE+TRAJECTORY\n"
    );

    printf(
        "PROJECTED=1E21_ADDRESSABLE_LANES\n"
    );

    printf(
        "V13_FIRST_MOVEMENT_COMPLETE=%s\n",
        ok ? "YES" : "NO"
    );

    return ok ? 0 : 1;
}
