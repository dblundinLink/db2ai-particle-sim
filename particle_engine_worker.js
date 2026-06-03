/**
 * Web Worker for parallel CPU force calculation
 *
 * Each worker computes forces for a subset of particles (chunk_start to chunk_end).
 * The main thread distributes particles across N workers, collects results, and
 * performs velocity-Verlet integration.
 *
 * This parallelizes the O(N²) force loop across CPU cores in the browser — same
 * exact physics as single-threaded engine, just distributed execution.
 */

self.onmessage = function(e) {
  const {
    type,
    particles,          // Full particle array (all N particles)
    chunk_start,        // Start index for this worker's chunk
    chunk_end,          // End index (exclusive)
    G,
    k_e,
    epsilon,
    worker_id
  } = e.data;

  if (type === 'compute_forces') {
    const accelerations = [];

    // Compute acceleration for particles [chunk_start, chunk_end)
    for (let i = chunk_start; i < chunk_end; i++) {
      const p_i = particles[i];
      let ax = 0;
      let ay = 0;

      // Sum forces from all other particles (O(N) inner loop)
      for (let j = 0; j < particles.length; j++) {
        if (i === j) continue;

        const p_j = particles[j];

        // Vector from i to j
        const dx = p_j.x - p_i.x;
        const dy = p_j.y - p_i.y;
        const r_sq = dx * dx + dy * dy + epsilon * epsilon;
        const r = Math.sqrt(r_sq);

        // Gravitational force (always attractive)
        const f_grav = G * p_i.mass * p_j.mass / r_sq;

        // Electrostatic force (repulsive for like charges, attractive for opposite)
        const f_elec = -k_e * p_i.charge * p_j.charge / r_sq;

        // Total force magnitude
        const f_total = f_grav + f_elec;

        // Acceleration contribution (F = ma, so a = F/m)
        const a_mag = f_total / p_i.mass;
        ax += a_mag * (dx / r);  // Unit vector toward j
        ay += a_mag * (dy / r);
      }

      accelerations.push({ ax, ay });
    }

    // Send results back to main thread
    self.postMessage({
      type: 'forces_computed',
      worker_id,
      chunk_start,
      chunk_end,
      accelerations
    });
  }
};
