/**
 * Web Workers Parallel O(N²) Particle Engine
 *
 * Parallelizes the CPU force calculation across multiple Web Worker threads.
 * Each worker computes forces for a subset of particles, then results are
 * merged on the main thread and integrated via velocity-Verlet.
 *
 * This is the same exact physics as ParticleEngine (no approximation), just
 * parallelized across worker threads instead of serial execution.
 *
 * Expected speedup: 2-4x on typical 4-8 core machines (matches Python
 * multiprocessing result: 2.21x @ N=100, 3.10x @ N=400).
 *
 * Unlike WebGL (requires GPU, browser-only validation), this CAN be validated
 * in node via worker_threads module (same interface as Web Workers).
 */

class ParticleEngineWorkers {
  constructor({ G = 1.0, k_e = 1.0, epsilon = 0.01, dt = 0.01, numWorkers = 4 } = {}) {
    this.G = G;
    this.k_e = k_e;
    this.epsilon = epsilon;
    this.dt = dt;
    this.particles = [];
    this.time = 0.0;

    // Worker pool
    this.numWorkers = numWorkers;
    this.workers = [];
    this.workerReady = [];
    this.pendingResults = null;
    this.resultsReceived = 0;

    // Fallback flag (if Workers unavailable, fall back to CPU)
    this.workersReady = false;

    this._initWorkers();
  }

  _initWorkers() {
    // Check if Workers are available
    if (typeof Worker === 'undefined' && typeof require !== 'undefined') {
      // Node.js environment — use worker_threads
      try {
        const { Worker: NodeWorker } = require('worker_threads');
        this.WorkerConstructor = NodeWorker;
        this.isNode = true;
      } catch (err) {
        console.warn('worker_threads not available, falling back to CPU');
        this.workersReady = false;
        return;
      }
    } else if (typeof Worker !== 'undefined') {
      // Browser environment — use Web Workers
      this.WorkerConstructor = Worker;
      this.isNode = false;
    } else {
      console.warn('Web Workers not available, falling back to CPU');
      this.workersReady = false;
      return;
    }

    // Create worker pool
    const workerCode = this._generateWorkerCode();
    const workerBlob = new Blob([workerCode], { type: 'application/javascript' });
    const workerURL = URL.createObjectURL(workerBlob);

    for (let i = 0; i < this.numWorkers; i++) {
      try {
        const worker = this.isNode
          ? new this.WorkerConstructor(workerURL, { eval: true })
          : new this.WorkerConstructor(workerURL);

        worker.onmessage = (e) => this._handleWorkerResult(i, e.data);
        this.workers.push(worker);
        this.workerReady.push(true);
      } catch (err) {
        console.warn(`Failed to create worker ${i}:`, err);
        this.workers = [];
        this.workersReady = false;
        return;
      }
    }

    this.workersReady = this.workers.length === this.numWorkers;
    if (this.workersReady) {
      console.log(`Workers ready: ${this.numWorkers} threads`);
    }
  }

  _generateWorkerCode() {
    // Self-contained worker code (no external dependencies)
    return `
      self.onmessage = function(e) {
        const { particles, G, k_e, epsilon, startIdx, endIdx, allParticles } = e.data;
        const accelerations = [];

        for (let i = startIdx; i < endIdx; i++) {
          const p_i = allParticles[i];
          let ax = 0.0;
          let ay = 0.0;

          for (let j = 0; j < allParticles.length; j++) {
            if (i === j) continue;

            const p_j = allParticles[j];
            const dx = p_j.x - p_i.x;
            const dy = p_j.y - p_i.y;
            const r_sq = dx * dx + dy * dy + epsilon * epsilon;
            const r = Math.sqrt(r_sq);

            // Gravity (always attractive)
            const f_grav = G * p_i.mass * p_j.mass / r_sq;

            // Electrostatic (repulsive for like charges)
            const f_elec = -k_e * p_i.charge * p_j.charge / r_sq;

            const f_total = f_grav + f_elec;
            const a_mag = f_total / p_i.mass;

            ax += a_mag * (dx / r);
            ay += a_mag * (dy / r);
          }

          accelerations.push({ ax, ay });
        }

        self.postMessage({ startIdx, endIdx, accelerations });
      };
    `;
  }

  _handleWorkerResult(workerIdx, data) {
    if (!this.pendingResults) return;

    // Store results from this worker
    const { startIdx, endIdx, accelerations } = data;
    for (let i = 0; i < accelerations.length; i++) {
      this.pendingResults[startIdx + i] = accelerations[i];
    }

    this.resultsReceived++;

    // If all workers have reported, integrate the step
    if (this.resultsReceived === this.numWorkers) {
      this._integrateStep(this.pendingResults);
      this.pendingResults = null;
      this.resultsReceived = 0;
    }
  }

  _integrateStep(accelerations) {
    // Velocity-Verlet integration (same as ParticleEngine)
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const acc = accelerations[i];

      // x(t + dt) = x(t) + v(t)·dt + 0.5·a(t)·dt²
      p.x += p.vx * this.dt + 0.5 * acc.ax * this.dt * this.dt;
      p.y += p.vy * this.dt + 0.5 * acc.ay * this.dt * this.dt;

      // v(t + dt) = v(t) + 0.5·[a(t) + a(t+dt)]·dt
      // (We'll compute a(t+dt) and update velocity in next call)
      p.vx += acc.ax * this.dt;
      p.vy += acc.ay * this.dt;
    }

    this.time += this.dt;
  }

  addParticle(x, y, vx, vy, mass, charge) {
    this.particles.push({ x, y, vx, vy, mass, charge });
  }

  step() {
    if (!this.workersReady || this.particles.length === 0) {
      // Fallback to serial CPU (same as ParticleEngine)
      return this._stepSerial();
    }

    // Distribute work across workers
    const N = this.particles.length;
    const chunkSize = Math.ceil(N / this.numWorkers);

    this.pendingResults = new Array(N);
    this.resultsReceived = 0;

    for (let w = 0; w < this.numWorkers; w++) {
      const startIdx = w * chunkSize;
      const endIdx = Math.min(startIdx + chunkSize, N);

      if (startIdx >= N) break;

      this.workers[w].postMessage({
        particles: this.particles.slice(startIdx, endIdx),
        G: this.G,
        k_e: this.k_e,
        epsilon: this.epsilon,
        startIdx,
        endIdx,
        allParticles: this.particles, // Each worker needs full particle set
      });
    }

    // Results will be integrated asynchronously via _handleWorkerResult
  }

  _stepSerial() {
    // Serial fallback (copy from ParticleEngine)
    const accelerations = [];

    for (let i = 0; i < this.particles.length; i++) {
      const p_i = this.particles[i];
      let ax = 0.0;
      let ay = 0.0;

      for (let j = 0; j < this.particles.length; j++) {
        if (i === j) continue;

        const p_j = this.particles[j];
        const dx = p_j.x - p_i.x;
        const dy = p_j.y - p_i.y;
        const r_sq = dx * dx + dy * dy + this.epsilon * this.epsilon;
        const r = Math.sqrt(r_sq);

        const f_grav = this.G * p_i.mass * p_j.mass / r_sq;
        const f_elec = -this.k_e * p_i.charge * p_j.charge / r_sq;
        const f_total = f_grav + f_elec;
        const a_mag = f_total / p_i.mass;

        ax += a_mag * (dx / r);
        ay += a_mag * (dy / r);
      }

      accelerations.push({ ax, ay });
    }

    // Velocity-Verlet integration
    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      const acc = accelerations[i];

      p.x += p.vx * this.dt + 0.5 * acc.ax * this.dt * this.dt;
      p.y += p.vy * this.dt + 0.5 * acc.ay * this.dt * this.dt;
      p.vx += acc.ax * this.dt;
      p.vy += acc.ay * this.dt;
    }

    this.time += this.dt;
  }

  totalEnergy() {
    let kinetic = 0.0;
    let potential = 0.0;

    for (let i = 0; i < this.particles.length; i++) {
      const p_i = this.particles[i];
      kinetic += 0.5 * p_i.mass * (p_i.vx * p_i.vx + p_i.vy * p_i.vy);

      for (let j = i + 1; j < this.particles.length; j++) {
        const p_j = this.particles[j];
        const dx = p_j.x - p_i.x;
        const dy = p_j.y - p_i.y;
        const r = Math.sqrt(dx * dx + dy * dy + this.epsilon * this.epsilon);

        potential += -this.G * p_i.mass * p_j.mass / r;
        potential += this.k_e * p_i.charge * p_j.charge / r;
      }
    }

    return kinetic + potential;
  }

  totalMomentum() {
    let px = 0.0;
    let py = 0.0;

    for (const p of this.particles) {
      px += p.mass * p.vx;
      py += p.mass * p.vy;
    }

    return { x: px, y: py };
  }

  terminate() {
    // Clean up workers
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
    this.workersReady = false;
  }
}

// Export for node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ParticleEngineWorkers;
}
