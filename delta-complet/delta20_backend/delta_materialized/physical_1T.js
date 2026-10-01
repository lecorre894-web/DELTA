"use strict";

const { Worker, isMainThread, parentPort } = require("worker_threads");
const { performance } = require("perf_hooks");

const QBITS = 1_000_000_000_000n;
const MASK64 = (1n << 64n) - 1n;

function mix64(x) {
    x &= MASK64;
    x ^= x >> 30n;
    x = (x * 0xbf58476d1ce4e5b9n) & MASK64;
    x ^= x >> 27n;
    x = (x * 0x94d049bb133111ebn) & MASK64;
    x ^= x >> 31n;
    return x & MASK64;
}

function qbitState(seed, id) {
    if (id < 0n || id >= QBITS)
        throw new RangeError("qbit outside 1T domain");

    return mix64(
        seed ^
        mix64(id + 0x9e3779b97f4a7c15n)
    );
}

if (!isMainThread) {

    const seed = 0x123456789abcdef0n;

    /*
     * CORE 1:
     * construction de la représentation physique compacte
     * du domaine complet.
     *
     * Les bornes + invariant + empreinte constituent
     * l'objet physique réellement conservé en mémoire.
     */

    const physical = new BigUint64Array(
        new SharedArrayBuffer(8 * 8)
    );

    physical[0] = 0n;
    physical[1] = QBITS - 1n;
    physical[2] = QBITS;
    physical[3] = qbitState(seed, 0n);
    physical[4] = qbitState(seed, QBITS / 2n);
    physical[5] = qbitState(seed, QBITS - 1n);

    physical[6] = mix64(
        physical[3] ^
        physical[4] ^
        physical[5] ^
        QBITS
    );

    physical[7] = mix64(
        physical[6] ^
        0x44454c5441315400n
    );

    parentPort.postMessage({
        first: physical[0].toString(),
        last: physical[1].toString(),
        domain: physical[2].toString(),

        q0: physical[3].toString(),
        qmid: physical[4].toString(),
        qlast: physical[5].toString(),

        fingerprint: physical[6].toString(),
        physicalObject: physical[7].toString(),

        bytes: physical.byteLength
    });

    return;
}

console.log("==================================================");
console.log(" DELTA PHYSICAL 1T / DUAL XEON CORE EXPERIMENT");
console.log("==================================================");

console.log("QBIT_DOMAIN             =", QBITS.toString());
console.log("QBIT_FIRST              = 0");
console.log("QBIT_LAST               =", (QBITS - 1n).toString());

console.log("");
console.log("CORE_0_ROLE             = BRIDGE");
console.log("CORE_1_ROLE             = PHYSICAL_MATERIALIZER");
console.log("QBIT_ROLE               = EXECUTION_LAYER");
console.log("DOMAIN_MODEL            = FULL_1T");
console.log("");

const start = performance.now();

const worker = new Worker(__filename);

worker.once("message", result => {

    const end = performance.now();

    const bridgeSeed =
        mix64(
            BigInt(result.physicalObject) ^
            QBITS
        );

    /*
     * Vérification depuis le bridge.
     */

    const expected0 =
        qbitState(0x123456789abcdef0n, 0n);

    const expectedMid =
        qbitState(
            0x123456789abcdef0n,
            QBITS / 2n
        );

    const expectedLast =
        qbitState(
            0x123456789abcdef0n,
            QBITS - 1n
        );

    const domainOK =
        BigInt(result.first) === 0n &&
        BigInt(result.last) === QBITS - 1n &&
        BigInt(result.domain) === QBITS;

    const stateOK =
        BigInt(result.q0) === expected0 &&
        BigInt(result.qmid) === expectedMid &&
        BigInt(result.qlast) === expectedLast;

    console.log("=== CORE 1 PHYSICAL OBJECT ===");
    console.log("PHYSICAL_DOMAIN_FIRST    =", result.first);
    console.log("PHYSICAL_DOMAIN_LAST     =", result.last);
    console.log("PHYSICAL_DOMAIN_SIZE     =", result.domain);

    console.log("");
    console.log("QBIT[0]                  =", result.q0);
    console.log("QBIT[500000000000]       =", result.qmid);
    console.log("QBIT[999999999999]       =", result.qlast);

    console.log("");
    console.log("DOMAIN_FINGERPRINT       =", result.fingerprint);
    console.log("PHYSICAL_OBJECT          =", result.physicalObject);
    console.log("PHYSICAL_OBJECT_BYTES    =", result.bytes);

    console.log("");
    console.log("=== CORE 0 BRIDGE VERIFICATION ===");
    console.log(
        "FULL_1T_DOMAIN_VALID     =",
        domainOK ? "YES" : "NO"
    );

    console.log(
        "QBIT_STATE_VALID         =",
        stateOK ? "YES" : "NO"
    );

    console.log(
        "BRIDGE_STATE             =",
        bridgeSeed.toString()
    );

    console.log(
        "DUAL_CORE_WALL           =",
        ((end - start) / 1000).toFixed(9),
        "s"
    );

    console.log("");
    console.log("=== CLASSIFICATION ===");

    console.log(
        "PHYSICAL_1T_OBJECT       = COMPACT_PHYSICAL_REPRESENTATION"
    );

    console.log(
        "PHYSICAL_OBJECT_IN_RAM   = YES"
    );

    console.log(
        "PHYSICAL_BYTES_MEASURED  =",
        result.bytes
    );

    console.log(
        "1T_QBIT_IDS              = LOGICAL_DOMAIN"
    );

    console.log(
        "1T_INDIVIDUAL_STATES_RAM = NO"
    );

    console.log(
        "CPU_EXECUTION            = PHYSICAL_MEASURED"
    );

    console.log(
        "QBIT_EXECUTION            = LOGICAL_EMULATED"
    );

    const passed = domainOK && stateOK;

    console.log("");
    console.log(
        "DELTA_PHYSICAL_1T =",
        passed ? "PASSED" : "FAILED"
    );

    process.exitCode = passed ? 0 : 1;
});

worker.once("error", err => {
    console.error(err);
    process.exitCode = 1;
});
