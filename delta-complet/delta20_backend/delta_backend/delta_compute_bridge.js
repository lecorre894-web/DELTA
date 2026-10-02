"use strict";

const { execFileSync } = require("child_process");

function run(cmd, args = []) {
    try {
        return execFileSync(cmd, args, {
            encoding: "utf8",
            timeout: 15000,
            stdio: ["ignore", "pipe", "pipe"]
        }).trim();
    } catch {
        return null;
    }
}

function localHardware() {
    const cpu = run("bash", [
        "-lc",
        "lscpu | grep 'Model name:' | sed 's/.*: *//'"
    ]);

    const gpu = run("bash", [
        "-lc",
        "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    ]);

    return {
        node: "CODESPACES",
        cpu: cpu || "UNKNOWN",
        gpu: gpu || "NONE",
        gpuExecution: gpu ? "PHYSICAL_VISIBLE" : "NOT_VISIBLE"
    };
}

function remoteHardware() {
    const host = process.env.DELTA_COMPUTE_HOST;

    if (!host) {
        return {
            connected: false,
            state: "REMOTE_NODE_NOT_CONFIGURED"
        };
    }

    const result = run("ssh", [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        host,
        "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
    ]);

    if (!result) {
        return {
            connected: false,
            state: "REMOTE_NODE_UNREACHABLE"
        };
    }

    return {
        connected: true,
        state: "REMOTE_NODE_CONNECTED",
        result
    };
}

function status() {
    return {
        bridge: "DELTA_COMPUTE_BRIDGE",
        local: localHardware(),
        remote: remoteHardware()
    };
}

function print() {
    const r = status();

    console.log("=== DELTA COMPUTE BRIDGE ===");
    console.log("LOCAL_NODE =", r.local.node);
    console.log("LOCAL_CPU =", r.local.cpu);
    console.log("LOCAL_GPU =", r.local.gpu);
    console.log("LOCAL_GPU_EXECUTION =", r.local.gpuExecution);

    console.log("");
    console.log("REMOTE_STATE =", r.remote.state);

    if (r.remote.connected) {
        console.log("REMOTE_HARDWARE =");
        console.log(r.remote.result);
    }

    console.log("");
    console.log(
        "DELTA_REMOTE_COMPUTE =",
        r.remote.connected ? "READY" : "WAITING_FOR_REMOTE_NODE"
    );

    return r;
}

module.exports = {
    status,
    print
};

if (require.main === module) {
    print();
}
