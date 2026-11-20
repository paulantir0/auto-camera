// The curved-sheet model of auto_camera/pages.py (Matt Zucker's page_dewarp
// model) and its fit to the lines of text of one page.
//
// A page is a sheet, straight from top to bottom and curving as a cubic from
// left to right, seen through a pinhole camera. Each line of text is a line
// of constant height on the sheet. Python fits it with SciPy's least_squares
// (trust region, soft-L1 loss, sparse Jacobian); here the same problem is
// solved by Levenberg-Marquardt with the same loss: per point a sheet
// position x, per line a height y, and eight numbers shared by all (camera
// rotation and position, the two slopes of the curve). The points' x are
// eliminated first (each touches only its own two residuals), which leaves a
// small dense system over the shared numbers and the lines' heights.

export const FOCAL = 1.4;       // pages.FOCAL: focal length as a share of half the frame's width

export interface Pose { rvec: number[]; tvec: number[] }

/** Rotation matrix of a rotation vector (cv2.Rodrigues), row-major. */
export function rodrigues(r: ArrayLike<number>): number[] {
  const theta = Math.hypot(r[0], r[1], r[2]);
  if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = r[0] / theta, y = r[1] / theta, z = r[2] / theta;
  const c = Math.cos(theta), s = Math.sin(theta), C = 1 - c;
  return [
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    y * x * C + z * s, c + y * y * C, y * z * C - x * s,
    z * x * C - y * s, z * y * C + x * s, c + z * z * C,
  ];
}

/** _sheet_z: height of the sheet above its plane at x (0 to width): a cubic, zero at both ends. */
export function sheetZ(x: number, width: number, alpha: number, beta: number): number {
  const u = x / width;
  return width * (((alpha + beta) * u - (2 * alpha + beta)) * u + alpha) * u;
}

/** Project sheet points (x, y, z) into normalised frame coordinates (cv2.projectPoints with K = diag(FOCAL, FOCAL, 1)). */
export function project(xs: ArrayLike<number>, ys: ArrayLike<number>, zs: ArrayLike<number>, rvec: ArrayLike<number>, tvec: ArrayLike<number>, out: Float64Array): Float64Array {
  const R = rodrigues(rvec);
  for (let i = 0; i < xs.length; i++) {
    const X = R[0] * xs[i] + R[1] * ys[i] + R[2] * zs[i] + tvec[0];
    const Y = R[3] * xs[i] + R[4] * ys[i] + R[5] * zs[i] + tvec[1];
    const Z = R[6] * xs[i] + R[7] * ys[i] + R[8] * zs[i] + tvec[2];
    const iz = Z !== 0 ? 1 / Z : 0;
    out[2 * i] = FOCAL * X * iz;
    out[2 * i + 1] = FOCAL * Y * iz;
  }
  return out;
}

export interface SheetProblem {
  observed: Float64Array;    // 2N: the samples, normalised (x, y)
  spanOf: Int32Array;        // N: the line of each sample
  spans: number;
  width: number;             // the sheet's width (for the cubic)
  scale: number;             // f_scale of the soft-L1 loss
}

export interface SheetFit { params: Float64Array; residuals: Float64Array; evaluations: number }

/**
 * Minimise 0.5 * sum C^2 rho((f_i / C)^2), rho(z) = 2 (sqrt(1 + z) - 1)
 * (SciPy's soft_l1), f the 2N residuals, over params = [rvec(3), tvec(3),
 * alpha, beta, y per line, x per point]. At most `maxEvaluations` residual
 * evaluations, as least_squares' max_nfev.
 */
export function fitSheet(problem: SheetProblem, start: Float64Array, maxEvaluations = 200): SheetFit {
  const { observed, spanOf, spans, width, scale: C } = problem;
  const N = spanOf.length, G = 8, S = spans, M = G + S;
  const p = Float64Array.from(start);
  const zs = new Float64Array(N), xs = new Float64Array(N), ys = new Float64Array(N);
  const proj = new Float64Array(2 * N);
  let evaluations = 0;

  const residuals = (q: Float64Array, out: Float64Array) => {
    evaluations++;
    for (let i = 0; i < N; i++) {
      xs[i] = q[G + S + i];
      ys[i] = q[G + spanOf[i]];
      zs[i] = sheetZ(xs[i], width, q[6], q[7]);
    }
    project(xs, ys, zs, q.subarray(0, 3), q.subarray(3, 6), proj);
    for (let k = 0; k < 2 * N; k++) out[k] = proj[k] - observed[k];
    return out;
  };
  const cost = (f: Float64Array) => {
    let s = 0;
    for (let k = 0; k < f.length; k++) { const z = (f[k] / C) ** 2; s += C * C * 2 * (Math.sqrt(1 + z) - 1); }
    return 0.5 * s;
  };

  let f = residuals(p, new Float64Array(2 * N));
  let current = cost(f);
  let lambda = 1e-3;
  const fp = new Float64Array(2 * N);
  const A = new Float64Array(2 * N * G);   // d f / d global
  const B = new Float64Array(2 * N);       // d f / d own line's y
  const Cx = new Float64Array(2 * N);      // d f / d own x
  const trial = new Float64Array(p.length);
  const fTrial = new Float64Array(2 * N);
  while (evaluations < maxEvaluations) {
    // Jacobian by forward differences: one evaluation per shared number, one for all lines' y, one for all points' x
    const step = (v: number) => 1e-7 * Math.max(1, Math.abs(v));
    for (let g = 0; g < G; g++) {
      const h = step(p[g]);
      const save = p[g];
      p[g] = save + h;
      residuals(p, fp);
      p[g] = save;
      for (let k = 0; k < 2 * N; k++) A[k * G + g] = (fp[k] - f[k]) / h;
    }
    {
      const hs = new Float64Array(S);
      for (let s = 0; s < S; s++) { hs[s] = step(p[G + s]); p[G + s] += hs[s]; }
      residuals(p, fp);
      for (let s = 0; s < S; s++) p[G + s] -= hs[s];
      for (let i = 0; i < N; i++) { const h = hs[spanOf[i]]; B[2 * i] = (fp[2 * i] - f[2 * i]) / h; B[2 * i + 1] = (fp[2 * i + 1] - f[2 * i + 1]) / h; }
    }
    {
      const hx = new Float64Array(N);
      for (let i = 0; i < N; i++) { hx[i] = step(p[G + S + i]); p[G + S + i] += hx[i]; }
      residuals(p, fp);
      for (let i = 0; i < N; i++) p[G + S + i] -= hx[i];
      for (let i = 0; i < N; i++) { Cx[2 * i] = (fp[2 * i] - f[2 * i]) / hx[i]; Cx[2 * i + 1] = (fp[2 * i + 1] - f[2 * i + 1]) / hx[i]; }
    }
    if (evaluations >= maxEvaluations) break;
    // soft-L1 weights (iteratively reweighted least squares)
    const w = new Float64Array(2 * N);
    for (let k = 0; k < 2 * N; k++) w[k] = 1 / Math.sqrt(1 + (f[k] / C) ** 2);
    // normal equations over [global, y], with each point's x eliminated
    const H = new Float64Array(M * M), gvec = new Float64Array(M);
    const hxx = new Float64Array(N), gx = new Float64Array(N), hcol = new Float64Array(N * M);
    for (let i = 0; i < N; i++) {
      const s = spanOf[i];
      for (let d = 0; d < 2; d++) {
        const k = 2 * i + d, wk = w[k];
        const a = A.subarray(k * G, k * G + G);
        for (let r = 0; r < G; r++) {
          for (let c = 0; c < G; c++) H[r * M + c] += wk * a[r] * a[c];
          H[r * M + G + s] += wk * a[r] * B[k];
          gvec[r] += wk * a[r] * f[k];
          hcol[i * M + r] += wk * a[r] * Cx[k];
        }
        H[(G + s) * M + G + s] += wk * B[k] * B[k];
        gvec[G + s] += wk * B[k] * f[k];
        hcol[i * M + G + s] += wk * B[k] * Cx[k];
        hxx[i] += wk * Cx[k] * Cx[k];
        gx[i] += wk * Cx[k] * f[k];
      }
    }
    for (let r = 0; r < G; r++) for (let s = 0; s < S; s++) H[(G + s) * M + r] = H[r * M + G + s];
    let accepted = false;
    for (let attempt = 0; attempt < 12 && evaluations < maxEvaluations; attempt++) {
      // damped: Marquardt's scaling of the diagonal
      const K = Float64Array.from(H), rhs = Float64Array.from(gvec);
      for (let r = 0; r < M; r++) K[r * M + r] += lambda * Math.max(H[r * M + r], 1e-12);
      const dx = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const d = hxx[i] * (1 + lambda) + 1e-12;
        dx[i] = d;
        const col = hcol.subarray(i * M, i * M + M);
        const s = spanOf[i];
        // only the shared numbers and this point's line are coupled to its x
        for (let r = 0; r < G; r++) {
          if (!col[r]) continue;
          for (let c = 0; c < G; c++) K[r * M + c] -= col[r] * col[c] / d;
          K[r * M + G + s] -= col[r] * col[G + s] / d;
          K[(G + s) * M + r] -= col[G + s] * col[r] / d;
          rhs[r] -= col[r] * gx[i] / d;
        }
        K[(G + s) * M + G + s] -= col[G + s] * col[G + s] / d;
        rhs[G + s] -= col[G + s] * gx[i] / d;
      }
      const delta = solve(K, rhs, M);
      if (!delta) { lambda *= 10; continue; }
      for (let r = 0; r < M; r++) trial[r] = p[r] - delta[r];
      for (let i = 0; i < N; i++) {
        const col = hcol.subarray(i * M, i * M + M);
        let v = gx[i];
        for (let r = 0; r < G; r++) v -= col[r] * delta[r];
        v -= col[G + spanOf[i]] * delta[G + spanOf[i]];
        trial[G + S + i] = p[G + S + i] - v / dx[i];
      }
      residuals(trial, fTrial);
      const next = cost(fTrial);
      if (Number.isFinite(next) && next < current) {
        const gain = (current - next) / Math.max(current, 1e-300);
        p.set(trial); f.set(fTrial); current = next;
        lambda = Math.max(lambda / 3, 1e-9);
        accepted = true;
        if (gain < 1e-10) return { params: p, residuals: f, evaluations };
        break;
      }
      lambda *= 4;
    }
    if (!accepted) break;
  }
  return { params: p, residuals: f, evaluations };
}

/** Solve the symmetric system K x = b (n x n, row-major) by Gaussian elimination with partial pivoting; null if singular. */
function solve(K: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const a = Float64Array.from(K), x = Float64Array.from(b);
  for (let c = 0; c < n; c++) {
    let pivot = c, best = Math.abs(a[c * n + c]);
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r * n + c]) > best) { best = Math.abs(a[r * n + c]); pivot = r; }
    if (!(best > 1e-300)) return null;
    if (pivot !== c) {
      for (let k = 0; k < n; k++) { const t = a[c * n + k]; a[c * n + k] = a[pivot * n + k]; a[pivot * n + k] = t; }
      const t = x[c]; x[c] = x[pivot]; x[pivot] = t;
    }
    for (let r = c + 1; r < n; r++) {
      const m = a[r * n + c] / a[c * n + c];
      if (!m) continue;
      for (let k = c; k < n; k++) a[r * n + k] -= m * a[c * n + k];
      x[r] -= m * x[c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let k = r + 1; k < n; k++) s -= a[r * n + k] * x[k];
    x[r] = s / a[r * n + r];
  }
  return x;
}
