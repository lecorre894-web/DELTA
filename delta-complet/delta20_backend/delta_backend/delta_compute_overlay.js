"use strict";

const os = require("os");
const { execFileSync } = require("child_process");

/*
 * DELTA COMPUTE OVERLAY
 *
 * Physical host:
 *   GitHub Codespaces / Xeon
 *
 * Virtual execution topology:
 *   AMD Ryzen 9 9950X3D
 *   NVIDIA GeForce RTX 4080
 *
 * The overlay describes DELTA's virtual execution topology.
 * It does NOT claim PCIe/CUDA passthrough to the Codespaces VM.
 */

const OVERLAY = Object.freeze({
    architecture: "DELTA_RYZEN_RTX_OVERLAY_V1",

    cpu: Object.freeze({
        vendor: "AMD",
        model: "AMD Ryzen 9 9950X3D",
        cores: 16,
        threads: 32,
        role: "VIRTUAL_EXECUTION_LAYER",
        executionClass: "VIRTUALIZED"
    }),

    gpu: Object.freeze({
        vendor: "NVIDIA",
        model: "NVIDIA GeForce RTX 4080",
        vramMiB: 16376,
        role: "VIRTUAL_ACCELERATION_LAYER",
        executionClass: "VIRTUALIZED"
    })
});

function command(cmd, args) {
    try {
        return execFileSync(cmd, args, {
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"]
        }).trim();
    } catch {
        return null;
    }
}

function physicalHost() {
    const cpu =
        command("bash", [
            "-lc",
            "lscpu | awk -F: '/Model name/{gsub(/^[ \\t]+/,\"\",$2); print $2; exit}'"
        ]) || os.cpus()[0]?.model || "UNKNOWN";

    const gpu = command("bash", [
        "-lc",
        "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader | head -1"
    ]);

    return {
        cpu,
        logicalCPUs: os.cpus().length,
        gpu: gpu || "NONE",
        gpuPhysicallyVisible: Boolean(gpu)
    };
}

function state() {
    const host = physicalHost();

    return {
        architecture: OVERLAY.architecture,

        physicalHost: {
            node: "CODESPACES",
            cpu: host.cpu,
            logicalCPUs: host.logicalCPUs,
            gpu: host.gpu,
            gpuExecution:
                host.gpuPhysicallyVisible
                    ? "PHYSICAL_VISIBLE"
                    : "NOT_VISIBLE"
        },

        overlay: OVERLAY,

        executionView: "RYZEN_RTX_OVER_XEON",

        physicalGpuPassthrough:
            host.gpuPhysicallyVisible,

        overlayActive: true
    };
}

function print() {
    const r = state();

    console.log("=== DELTA COMPUTE OVERLAY ===");

    console.log("");
    console.log("--- PHYSICAL HOST ---");
    console.log("HOST_NODE               =", r.physicalHost.node);
    console.log("HOST_CPU                =", r.physicalHost.cpu);
    console.log("HOST_LOGICAL_CPUS       =", r.physicalHost.logicalCPUs);
    console.log("HOST_GPU                =", r.physicalHost.gpu);
    console.log("HOST_GPU_EXECUTION      =", r.physicalHost.gpuExecution);

    console.log("");
    console.log("--- VIRTUAL EXECUTION LAYER ---");
    console.log("OVERLAY_CPU             =", r.overlay.cpu.model);
    console.log("OVERLAY_CPU_CORES       =", r.overlay.cpu.cores);
    console.log("OVERLAY_CPU_THREADS     =", r.overlay.cpu.threads);
    console.log("OVERLAY_CPU_EXECUTION   =", r.overlay.cpu.executionClass);

    console.log("OVERLAY_GPU             =", r.overlay.gpu.model);
    console.log("OVERLAY_GPU_VRAM_MIB    =", r.overlay.gpu.vramMiB);
    console.log("OVERLAY_GPU_EXECUTION   =", r.overlay.gpu.executionClass);

    console.log("");
    console.log("DELTA_EXECUTION_VIEW    =", r.executionView);
    console.log(
        "GPU_PASSTHROUGH        =",
        r.physicalGpuPassthrough ? "YES" : "NO"
    );
    console.log("DELTA_COMPUTE_OVERLAY   = ACTIVE");

    return r;
}

module.exports = {
    OVERLAY,
    state,
    print
};

if (require.main === module) {
    print();
}
