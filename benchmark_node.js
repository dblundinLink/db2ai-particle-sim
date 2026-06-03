#!/usr/bin/env node

// Benchmark JS engine throughput in node (no browser) to estimate max-interactive-N
// Usage: node web/benchmark_node.js

const fs = require('fs');
const ParticleEngine = require('./particle_engine.js');

// Test parameters
const N_VALUES = [10, 50, 100, 200, 400];
const STEPS_PER_TRIAL = 1000;  // 10x longer to get measurable timing
const TRIALS = 3;

// Physics constants (same as Python baseline)
const G = 1.0;
const k_e = 1.0;
const epsilon = 0.01;
const dt = 0.001;

function randomIC(N) {
    const positions = [];
    const velocities = [];
    const masses = [];
    const charges = [];

    for (let i = 0; i < N; i++) {
        // JS engine expects 2D arrays [[x,y], [x,y], ...]
        positions.push([Math.random() * 2 - 1, Math.random() * 2 - 1]);
        velocities.push([0, 0]);
        masses.push(1.0);
        charges.push(i % 2 === 0 ? 1.0 : -1.0);
    }

    return { positions, velocities, masses, charges };
}

function benchmark(N) {
    const times = [];

    for (let trial = 0; trial < TRIALS; trial++) {
        const ic = randomIC(N);
        const engine = new ParticleEngine({ G, k_e, epsilon, dt, dim: 2 });
        engine.init(ic.positions, ic.velocities, ic.masses, ic.charges);

        const start = process.hrtime.bigint();  // Use high-res timer
        for (let step = 0; step < STEPS_PER_TRIAL; step++) {
            engine.step();
        }
        const end = process.hrtime.bigint();
        const elapsed = Number(end - start) / 1e9;  // Convert nanoseconds to seconds

        times.push(elapsed);

        // Debug: print per-trial timing
        if (trial === 0) {
            console.log(`  (N=${N}, trial ${trial}: ${elapsed.toFixed(3)}s)`);
        }
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const particleStepsPerSec = Math.round(N * STEPS_PER_TRIAL / avgTime);

    return {
        N: N,
        avg_time_sec: avgTime,
        particle_steps_per_sec: particleStepsPerSec,
        min_time: Math.min(...times),
        max_time: Math.max(...times)
    };
}

console.log("Benchmarking JS engine in node...\n");
console.log("N      Time(s)    p·s/s");
console.log("----- --------- --------");

const results = [];
for (const N of N_VALUES) {
    const result = benchmark(N);
    console.log(`${String(N).padStart(5)} ${result.avg_time_sec.toFixed(3).padStart(9)} ${String(result.particle_steps_per_sec).padStart(8)}`);
    results.push(result);
}

// Estimate max-interactive-N assuming 30 fps target
// At 30 fps, we have 33ms per frame. For smooth interaction, assume ~10ms budget for physics (rest for rendering)
// 10ms = 0.01s per frame → need 100 frames/sec physics throughput
// particle_steps_per_sec / N = steps_per_sec achievable at that N
// For 100 steps/sec (smooth animation), need: particle_steps_per_sec >= 100 * N
console.log("\n--- Estimating max-interactive-N ---");
console.log("Target: 30 fps (100 physics steps/sec for smooth animation)");

let maxInteractiveN = 0;
for (const result of results) {
    const stepsPerSec = result.particle_steps_per_sec / result.N;
    const fps = stepsPerSec;  // 1 step per frame
    if (fps >= 100) {
        maxInteractiveN = result.N;
    }
    console.log(`N=${result.N}: ${Math.round(stepsPerSec)} steps/sec → ${fps >= 100 ? '✓' : '✗'} interactive`);
}

console.log(`\nEstimated max-interactive-N: ${maxInteractiveN} (conservative, node-based)`);

// Save results
const output = {
    benchmark_type: "JS engine (node)",
    timestamp: new Date().toISOString(),
    steps_per_trial: STEPS_PER_TRIAL,
    trials: TRIALS,
    results: results,
    estimated_max_interactive_N: maxInteractiveN
};

fs.writeFileSync('results/benchmarks_js.json', JSON.stringify(output, null, 2));
console.log("\nResults saved to results/benchmarks_js.json");
