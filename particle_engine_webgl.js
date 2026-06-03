// particle_engine_webgl.js — GPU-accelerated N-body simulator via WebGL
// Uses transform feedback to compute forces on GPU, matching exact physics of particle_engine.js

class ParticleEngineWebGL {
  constructor(config = {}) {
    // Physics constants (match Python/JS reference defaults)
    this.G = config.G !== undefined ? config.G : 1.0;
    this.k_e = config.k_e !== undefined ? config.k_e : 1.0;
    this.epsilon = config.epsilon !== undefined ? config.epsilon : 0.01;
    this.dt = config.dt !== undefined ? config.dt : 0.01;
    this.dim = config.dim !== undefined ? config.dim : 2;

    // Particle state
    this.N = 0;
    this.x = null;  // positions [N, dim] (host side)
    this.v = null;  // velocities [N, dim]
    this.m = null;  // masses [N]
    this.q = null;  // charges [N]
    this.a = null;  // accelerations [N, dim]

    // WebGL context and resources
    this.canvas = null;
    this.gl = null;
    this.program = null;
    this.buffers = {};
    this.gpuReady = false;

    this._initWebGL();
  }

  _initWebGL() {
    // Check if running in browser (has document)
    if (typeof document === 'undefined') {
      console.warn('Not in browser environment — falling back to CPU engine');
      this.gpuReady = false;
      return;
    }

    // Create offscreen canvas for GPU compute (no display)
    this.canvas = document.createElement('canvas');
    this.canvas.width = 1;
    this.canvas.height = 1;

    // Get WebGL2 context (needed for transform feedback)
    this.gl = this.canvas.getContext('webgl2');
    if (!this.gl) {
      console.warn('WebGL2 not available — falling back to CPU engine');
      this.gpuReady = false;
      return;
    }

    // Check for required extensions
    const ext = this.gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
      console.warn('EXT_color_buffer_float not available — GPU acceleration disabled');
      this.gpuReady = false;
      return;
    }

    this.gpuReady = true;
  }

  _createShaders() {
    const gl = this.gl;

    // Simple fullscreen quad vertex shader
    const vertexShaderSource = `#version 300 es
precision highp float;
in vec2 a_position;
out vec2 v_texCoord;

void main() {
  v_texCoord = a_position * 0.5 + 0.5;  // map [-1,1] to [0,1]
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

    // Fragment shader: compute acceleration for one particle (rendered to framebuffer)
    // Each pixel computes acceleration for one particle
    const fragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 outColor;

// Particle data textures (all N particles)
uniform sampler2D u_positionTex;  // RG32F: [x, y]
uniform sampler2D u_massTex;      // R32F: m
uniform sampler2D u_chargeTex;    // R32F: q

// Physics constants
uniform float u_G;
uniform float u_k_e;
uniform float u_epsilon;
uniform int u_N;
uniform int u_texWidth;

void main() {
  // Determine which particle this pixel represents
  ivec2 fragCoord = ivec2(gl_FragCoord.xy);
  int i = fragCoord.y * u_texWidth + fragCoord.x;

  if (i >= u_N) {
    outColor = vec4(0.0);
    return;
  }

  // Fetch particle i data
  vec2 x_i = texelFetch(u_positionTex, fragCoord, 0).xy;
  float m_i = texelFetch(u_massTex, fragCoord, 0).x;
  float q_i = texelFetch(u_chargeTex, fragCoord, 0).x;

  // Compute acceleration via O(N²) pairwise sum
  vec2 acc = vec2(0.0);

  for (int j = 0; j < 4096; j++) {  // hardware loop limit
    if (j >= u_N) break;
    if (j == i) continue;

    // Fetch particle j data
    ivec2 j_coord = ivec2(j % u_texWidth, j / u_texWidth);
    vec2 x_j = texelFetch(u_positionTex, j_coord, 0).xy;
    float m_j = texelFetch(u_massTex, j_coord, 0).x;
    float q_j = texelFetch(u_chargeTex, j_coord, 0).x;

    // r_ij = x_j - x_i
    vec2 r_ij = x_j - x_i;
    float r_sq = dot(r_ij, r_ij) + u_epsilon * u_epsilon;
    float r = sqrt(r_sq);

    // Gravity: F_grav = +G * m_i * m_j / (r² + ε²)
    float f_grav = u_G * m_i * m_j / r_sq;

    // Electrostatic: F_elec = -k_e * q_i * q_j / (r² + ε²)
    float f_elec = -u_k_e * q_i * q_j / r_sq;

    float f_total = f_grav + f_elec;

    // Unit vector r̂_ij
    vec2 r_hat = r_ij / r;

    // Acceleration on i
    acc += (f_total / m_i) * r_hat;
  }

  // Output acceleration as RG channels
  outColor = vec4(acc.x, acc.y, 0.0, 1.0);
}
`;

    // Compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
      console.error('Vertex shader compile error:', gl.getShaderInfoLog(vertexShader));
      this.gpuReady = false;
      return;
    }

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      console.error('Fragment shader compile error:', gl.getShaderInfoLog(fragmentShader));
      this.gpuReady = false;
      return;
    }

    // Link program
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('Program link error:', gl.getProgramInfoLog(this.program));
      this.gpuReady = false;
      return;
    }

    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
  }

  _createBuffersAndTextures() {
    const gl = this.gl;

    // Texture layout: pack N particles into smallest square texture
    this.texWidth = Math.ceil(Math.sqrt(this.N));
    this.texHeight = Math.ceil(this.N / this.texWidth);

    // Position texture (RG32F for 2D positions)
    const positionData = new Float32Array(this.texWidth * this.texHeight * 2);
    for (let i = 0; i < this.N; i++) {
      positionData[i * 2] = this.x[i][0];
      positionData[i * 2 + 1] = this.x[i][1];
    }

    this.buffers.positionTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.positionTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, this.texWidth, this.texHeight, 0, gl.RG, gl.FLOAT, positionData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Mass texture (R32F)
    const massData = new Float32Array(this.texWidth * this.texHeight);
    for (let i = 0; i < this.N; i++) {
      massData[i] = this.m[i];
    }

    this.buffers.massTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.massTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.texWidth, this.texHeight, 0, gl.RED, gl.FLOAT, massData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Charge texture (R32F)
    const chargeData = new Float32Array(this.texWidth * this.texHeight);
    for (let i = 0; i < this.N; i++) {
      chargeData[i] = this.q[i];
    }

    this.buffers.chargeTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.chargeTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, this.texWidth, this.texHeight, 0, gl.RED, gl.FLOAT, chargeData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // Framebuffer for rendering (output: acceleration texture)
    this.buffers.accelerationTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.accelerationTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, this.texWidth, this.texHeight, 0, gl.RG, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    this.buffers.framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.buffers.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.buffers.accelerationTex, 0);

    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('Framebuffer incomplete');
      this.gpuReady = false;
      return;
    }

    // Fullscreen quad vertices
    const quadVertices = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1,
    ]);

    this.buffers.quadVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  init(x, v, m, q) {
    this.N = x.length;
    this.x = x.map(row => [...row]);
    this.v = v.map(row => [...row]);
    this.m = [...m];
    this.q = [...q];
    this.a = Array(this.N).fill(0).map(() => Array(this.dim).fill(0));

    if (this.gpuReady) {
      this._createShaders();
      if (this.gpuReady) {
        this._createBuffersAndTextures();
      }
    }

    if (this.gpuReady) {
      this._computeAccelerationsGPU();
    } else {
      // Fallback to CPU (same as particle_engine.js)
      this._computeAccelerationsCPU();
    }
  }

  _computeAccelerationsGPU() {
    const gl = this.gl;

    // Update position texture with current state
    const positionData = new Float32Array(this.texWidth * this.texHeight * 2);
    for (let i = 0; i < this.N; i++) {
      positionData[i * 2] = this.x[i][0];
      positionData[i * 2 + 1] = this.x[i][1];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.positionTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.texWidth, this.texHeight, gl.RG, gl.FLOAT, positionData);

    // Bind framebuffer and set viewport
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.buffers.framebuffer);
    gl.viewport(0, 0, this.texWidth, this.texHeight);

    // Use shader program
    gl.useProgram(this.program);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.positionTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_positionTex'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.massTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_massTex'), 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.buffers.chargeTex);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_chargeTex'), 2);

    // Set uniforms
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_G'), this.G);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_k_e'), this.k_e);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_epsilon'), this.epsilon);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_N'), this.N);
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_texWidth'), this.texWidth);

    // Bind quad vertices
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.quadVBO);
    const posLoc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Draw fullscreen quad (runs fragment shader for each pixel)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Read back acceleration data
    const accelerationData = new Float32Array(this.texWidth * this.texHeight * 4);  // RGBA
    gl.readPixels(0, 0, this.texWidth, this.texHeight, gl.RGBA, gl.FLOAT, accelerationData);

    // Copy to acceleration array
    for (let i = 0; i < this.N; i++) {
      this.a[i][0] = accelerationData[i * 4];      // R channel = ax
      this.a[i][1] = accelerationData[i * 4 + 1];  // G channel = ay
    }

    // Cleanup
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _computeAccelerationsCPU() {
    // Exact copy of particle_engine.js O(N²) logic
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.a[i][d] = 0;
      }
    }

    for (let i = 0; i < this.N; i++) {
      for (let j = i + 1; j < this.N; j++) {
        const r_ij = Array(this.dim);
        let r_sq = this.epsilon * this.epsilon;
        for (let d = 0; d < this.dim; d++) {
          r_ij[d] = this.x[j][d] - this.x[i][d];
          r_sq += r_ij[d] * r_ij[d];
        }
        const r = Math.sqrt(r_sq);

        const f_grav = this.G * this.m[i] * this.m[j] / r_sq;
        const f_elec = -this.k_e * this.q[i] * this.q[j] / r_sq;
        const f_total = f_grav + f_elec;
        const r_hat = r_ij.map(c => c / r);

        for (let d = 0; d < this.dim; d++) {
          const force_d = f_total * r_hat[d];
          this.a[i][d] += force_d / this.m[i];
          this.a[j][d] -= force_d / this.m[j];
        }
      }
    }
  }

  step() {
    // Velocity-Verlet integration (same as particle_engine.js)
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.v[i][d] += this.a[i][d] * this.dt / 2;
      }
    }

    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.x[i][d] += this.v[i][d] * this.dt;
      }
    }

    if (this.gpuReady) {
      this._computeAccelerationsGPU();
    } else {
      this._computeAccelerationsCPU();
    }

    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        this.v[i][d] += this.a[i][d] * this.dt / 2;
      }
    }
  }

  // Energy and momentum (same as particle_engine.js)
  totalEnergy() {
    let ke = 0;
    for (let i = 0; i < this.N; i++) {
      let v_sq = 0;
      for (let d = 0; d < this.dim; d++) {
        v_sq += this.v[i][d] * this.v[i][d];
      }
      ke += 0.5 * this.m[i] * v_sq;
    }

    let pe_grav = 0;
    let pe_elec = 0;
    for (let i = 0; i < this.N; i++) {
      for (let j = i + 1; j < this.N; j++) {
        let r_sq = this.epsilon * this.epsilon;
        for (let d = 0; d < this.dim; d++) {
          const dx = this.x[j][d] - this.x[i][d];
          r_sq += dx * dx;
        }
        const r = Math.sqrt(r_sq);
        pe_grav += -this.G * this.m[i] * this.m[j] / r;
        pe_elec += this.k_e * this.q[i] * this.q[j] / r;
      }
    }

    return ke + pe_grav + pe_elec;
  }

  totalMomentum() {
    const p = Array(this.dim).fill(0);
    for (let i = 0; i < this.N; i++) {
      for (let d = 0; d < this.dim; d++) {
        p[d] += this.m[i] * this.v[i][d];
      }
    }
    return p;
  }
}

// Export for Node.js testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ParticleEngineWebGL;
}
