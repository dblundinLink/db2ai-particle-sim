// particle_engine.js — JavaScript N-body simulator matching Python reference
// Implements gravity + electrostatic forces with velocity-Verlet integration

class ParticleEngine {
  constructor(config = {}) {
    // Physics constants (match Python defaults)
    this.G = config.G !== undefined ? config.G : 1.0;
    this.k_e = config.k_e !== undefined ? config.k_e : 1.0;
    this.epsilon = config.epsilon !== undefined ? config.epsilon : 0.01;
    this.dt = config.dt !== undefined ? config.dt : 0.01;
    this.dim = config.dim !== undefined ? config.dim : 2;

    // Particle state (all arrays of length N)
    this.N = 0;
    this.x = null;  // positions [N, dim]
    this.v = null;  // velocities [N, dim]
    this.m = null;  // masses [N]
    this.q = null;  // charges [N]
    this.a = null;  // accelerations [N, dim] (cached)
  }

  // Initialize particles from arrays
  init(x, v, m, q) {
    this.N = x.length;
    this.x = x.map(row => [...row]);  // deep copy
    this.v = v.map(row => [...row]);
    this.m = [...m];
    this.q = [...q];
    this.a = Array(this.N).fill(0).map(() => Array(this.dim).fill(0));
    this._computeAccelerations();
  }

  // Compute net accelerations on all particles
  _computeAccelerations() {
    // Zero out accelerations
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.a[i][d] = 0;
      }
    }

    // O(N²) pairwise forces
    for (let i = 0; i < this.N; i++) {
      for (let j = i + 1; j < this.N; j++) {
        // r_ij = x[j] - x[i]
        const r_ij = Array(this.dim);
        let r_sq = this.epsilon * this.epsilon;
        for (let d = 0; d < this.dim; d++) {
          r_ij[d] = this.x[j][d] - this.x[i][d];
          r_sq += r_ij[d] * r_ij[d];
        }
        const r = Math.sqrt(r_sq);

        // Gravity: F_grav = +G * m_i * m_j / (r² + ε²) toward j
        const f_grav = this.G * this.m[i] * this.m[j] / r_sq;

        // Electrostatic: F_elec = -k_e * q_i * q_j / (r² + ε²)
        // (negative sign: like charges repel, opposite attract)
        const f_elec = -this.k_e * this.q[i] * this.q[j] / r_sq;

        const f_total = f_grav + f_elec;

        // Unit vector r̂_ij
        const r_hat = r_ij.map(c => c / r);

        // Apply force to both particles (Newton's 3rd law)
        for (let d = 0; d < this.dim; d++) {
          const force_d = f_total * r_hat[d];
          this.a[i][d] += force_d / this.m[i];   // on i
          this.a[j][d] -= force_d / this.m[j];   // on j (opposite)
        }
      }
    }
  }

  // Velocity-Verlet integration step
  step() {
    // v(t + dt/2) = v(t) + a(t) * dt/2
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.v[i][d] += this.a[i][d] * this.dt / 2;
      }
    }

    // x(t + dt) = x(t) + v(t + dt/2) * dt
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.x[i][d] += this.v[i][d] * this.dt;
      }
    }

    // Recompute accelerations at new positions
    this._computeAccelerations();

    // v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.v[i][d] += this.a[i][d] * this.dt / 2;
      }
    }
  }

  // Compute total kinetic energy
  kineticEnergy() {
    let ke = 0;
    for (let i = 0; i < this.N; i++) {
      let v_sq = 0;
      for (let d = 0; d < this.dim; d++) {
        v_sq += this.v[i][d] * this.v[i][d];
      }
      ke += 0.5 * this.m[i] * v_sq;
    }
    return ke;
  }

  // Compute total potential energy (gravity + electrostatic)
  potentialEnergy() {
    let pe = 0;
    for (let i = 0; i < this.N; i++) {
      for (let j = i + 1; j < this.N; j++) {
        let r_sq = this.epsilon * this.epsilon;
        for (let d = 0; d < this.dim; d++) {
          const dx = this.x[j][d] - this.x[i][d];
          r_sq += dx * dx;
        }
        const r = Math.sqrt(r_sq);

        // Gravity potential: -G * m_i * m_j / r
        pe -= this.G * this.m[i] * this.m[j] / r;

        // Electrostatic potential: +k_e * q_i * q_j / r
        pe += this.k_e * this.q[i] * this.q[j] / r;
      }
    }
    return pe;
  }

  // Total energy
  totalEnergy() {
    return this.kineticEnergy() + this.potentialEnergy();
  }

  // Total momentum (vector)
  totalMomentum() {
    const p = Array(this.dim).fill(0);
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        p[d] += this.m[i] * this.v[i][d];
      }
    }
    return p;
  }

  // Get current state as plain object (for serialization/testing)
  getState() {
    return {
      x: this.x.map(row => [...row]),
      v: this.v.map(row => [...row]),
      m: [...this.m],
      q: [...this.q],
      energy: this.totalEnergy(),
      momentum: this.totalMomentum()
    };
  }
}

// Export for node testing (if running in node) or browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ParticleEngine;
}
