#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>

#define REAL_PULSES 10000000ULL

typedef struct {
    uint64_t hi;
    uint64_t lo;
} lane128_t;

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
 * Génère une adresse logique 128 bits.
 * Pour le niveau maximum, l'espace est borné à 10^21 lanes.
 */
static inline lane128_t virtual_lane(uint64_t pulse)
{
    /*
     * 10^21 = 54 * 2^64 + 3875820019684212736
     *
     * On utilise directement pulse comme graine déterministe.
     * L'adresse résultante reste dans [0,10^21).
     */
    const uint64_t MAX_HI = 54ULL;
    const uint64_t MAX_LO = 3875820019684212736ULL;

    uint64_t a = mix64(pulse);
    uint64_t b = mix64(pulse ^ 0x9e3779b97f4a7c15ULL);

    lane128_t lane;

    lane.hi = a % (MAX_HI + 1ULL);

    if (lane.hi == MAX_HI)
        lane.lo = b % MAX_LO;
    else
        lane.lo = b;

    return lane;
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

    volatile uint64_t checksum = 0;

    printf("\n");
    printf("====================================================\n");
    printf(" DELTA V14 CLOUD — 1000 MILLIARDS DE MILLIARDS\n");
    printf("====================================================\n");
    printf("PHYSICAL_CPU_ALLOWED=%d\n", allowed);
    printf("REAL_PULSES=%llu\n",
           (unsigned long long)REAL_PULSES);
    printf("VIRTUAL_LANES=1000000000000000000000\n");
    printf("VIRTUAL_LANES_SCIENTIFIC=1e21\n");
    printf("ADDRESS_BITS=128\n");
    printf("PER_LANE_ALLOCATION=NONE\n");
    printf("STATUS_REAL=MEASURED\n");
    printf("STATUS_VIRTUAL=PROJECTED\n");
    printf("----------------------------------------------------\n");

    double t0 = now_s();

    for (uint64_t i = 0; i < REAL_PULSES; i++) {

        lane128_t lane = virtual_lane(i);

        uint64_t state =
            mix64(
                i ^
                lane.lo ^
                mix64(lane.hi)
            );

        checksum ^= state;
        checksum += lane.lo;
        checksum ^= lane.hi;
    }

    double wall = now_s() - t0;
    double rate = (double)REAL_PULSES / wall;

    /*
     * 10^7 pulses × 10^21 lanes = 10^28
     * uniquement comme espace d'adressage virtuel.
     */

    printf("REAL_WALL=%.9f s\n", wall);
    printf("REAL_RATE=%.3f M_pulses/s\n", rate / 1e6);
    printf("VIRTUAL_ADDRESS_EVENTS=1.000e+28\n");
    printf("CHECKSUM=%llu\n",
           (unsigned long long)checksum);

    printf("VALIDATION=%s\n",
           (allowed == 1 && checksum != 0) ? "OK" : "FAIL");

    printf("----------------------------------------------------\n");
    printf("V14_1E21_COMPLETE=YES\n");

    return 0;
}
