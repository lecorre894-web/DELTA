"use strict";

const fs = require("fs");
const { performance } = require("perf_hooks");

const QBIT_DOMAIN = 1_000_000_000_000n;
const MATERIALIZED_QBITS = 1_000_000;
const STATE_BYTES = 8;

const DATA_FILE = "delta_materialized/DELTA_QBITS.bin";
const META_FILE = "delta_materialized/DELTA_QBITS_META.json";

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

console.log("==============================================");
console.log(" DELTA PERSISTENT QBIT MATERIALIZATION");
console.log("==============================================");
console.log("");
console.log("QBIT_DOMAIN                 =", QBIT_DOMAIN.toString());
console.log("CPU_CORE_ROLE               = BRIDGE");
console.log("QBIT_ROLE                   = EXECUTION_LAYER");
console.log("LOGICAL_DOMAIN              = IMPLICIT");
console.log("PHYSICAL_MATERIALIZATION    = PERSISTENT_FILE");
console.log("TARGET_MATERIALIZED_QBITS   =", MATERIALIZED_QBITS);
console.log("STATE_BYTES                 =", STATE_BYTES);
console.log("");

const bytes = MATERIALIZED_QBITS * STATE_BYTES;

console.log("ALLOCATING_BUFFER...");
const buffer = Buffer.allocUnsafe(bytes);

let checksum = 0n;

const start = performance.now();

for (let i = 0; i < MATERIALIZED_QBITS; i++) {
    const qbit = BigInt(i);

    const state = mix64(
        qbit ^
        0x123456789abcdef0n
    );

    buffer.writeBigUInt64LE(state, i * STATE_BYTES);

    checksum ^= state;
}

const materialized = performance.now();

fs.writeFileSync(DATA_FILE, buffer);

const persisted = performance.now();

const first = buffer.readBigUInt64LE(0);
const middle =
    buffer.readBigUInt64LE(
        Math.floor(MATERIALIZED_QBITS / 2) * STATE_BYTES
    );
const last =
    buffer.readBigUInt64LE(
        (MATERIALIZED_QBITS - 1) * STATE_BYTES
    );

const validation =
    first === mix64(0n ^ 0x123456789abcdef0n) &&
    middle ===
        mix64(
            BigInt(Math.floor(MATERIALIZED_QBITS / 2)) ^
            0x123456789abcdef0n
        ) &&
    last ===
        mix64(
            BigInt(MATERIALIZED_QBITS - 1) ^
            0x123456789abcdef0n
        );

const materializeWall = (materialized - start) / 1000;
const persistWall = (persisted - materialized) / 1000;
const totalWall = (persisted - start) / 1000;

const metadata = {
    qbit_domain: QBIT_DOMAIN.toString(),
    physically_materialized_qbits: MATERIALIZED_QBITS,
    state_bytes: STATE_BYTES,
    physical_bytes: bytes,
    checksum: checksum.toString(),
    first_state: first.toString(),
    middle_state: middle.toString(),
    last_state: last.toString(),
    validation
};

fs.writeFileSync(
    META_FILE,
    JSON.stringify(metadata, null, 2)
);

console.log("");
console.log("=== PHYSICAL MATERIALIZATION ===");
console.log("PHYSICALLY_MATERIALIZED_QBITS =", MATERIALIZED_QBITS);
console.log("PHYSICAL_BYTES                =", bytes);
console.log("DATA_FILE                     =", DATA_FILE);
console.log("PERSISTENT_STORAGE            = YES");
console.log("");

console.log("=== PROBES FROM MATERIALIZED MEMORY ===");
console.log("QBIT[0]                       =", first.toString());
console.log(
    "QBIT[" + Math.floor(MATERIALIZED_QBITS / 2) + "]                 =",
    middle.toString()
);
console.log(
    "QBIT[" + (MATERIALIZED_QBITS - 1) + "]                 =",
    last.toString()
);
console.log("");

console.log("CHECKSUM                      =", checksum.toString());
console.log("MATERIALIZE_WALL              =", materializeWall.toFixed(9), "s");
console.log("PERSIST_WALL                  =", persistWall.toFixed(9), "s");
console.log("TOTAL_WALL                    =", totalWall.toFixed(9), "s");
console.log(
    "MATERIALIZATION_RATE         =",
    (MATERIALIZED_QBITS / totalWall).toFixed(3),
    "qbit-states/s"
);

console.log("");
console.log("=== CLASSIFICATION ===");
console.log("1T_QBIT_DOMAIN                = LOGICAL_EMULATED");
console.log("CPU_BRIDGE_EXECUTION          = PHYSICAL_MEASURED");
console.log("MATERIALIZED_SUBSET           = PHYSICAL_MEMORY_BYTES");
console.log("PER_MATERIALIZED_QBIT_BYTES   =", STATE_BYTES);
console.log("ALL_1T_MATERIALIZED           = NO");
console.log(
    "PERSISTENT_MATERIALIZATION   =",
    validation ? "PASSED" : "FAILED"
);

process.exitCode = validation ? 0 : 1;
