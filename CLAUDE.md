# Lobatto workspace

Repository `Ananke-Inria/Lobatto` (formerly `OndoMathX/QuadratiX`, renamed 2026-11-25). The OndoMathX crates (mesh, SEM, TetriX, toolbox) depend on these, never the reverse.

Rust workspace for Gauss–Lobatto spectral-element numerics.

| Crate | Role |
|---|---|
| `lobatto` | Gauss–Lobatto points, quadrature, barycentric/Lagrange basis utilities |
| `lobatto-grid` | Grid geometry (`Grid1D`/`GridND<N>` on arbitrary domains, `Extension` folding, DOF indexing; `NodeLayout`: Gauss–Lobatto elements or, for periodic directions, `Grid1D::trigonometric` equidistant points with the trigonometric interpolant (one element of `n` cardinal functions, so the ND machinery is shared); the shared ND indexing helpers `nd_col_major_strides`/`nd_fiber_base`/`nd_multi_indices`/`nd_multi_index` live here) and interpolation (`eval`/`eval_extended`/`convect`, `EvalPlan`, `Scalar`) |
| `lobatto-grid::fft` (feature `fft`; was the crate `lobatto-fft` until lobatto-grid 0.1.9) | HoFFT transforms (`hofft::Engine`/`EngineND`), screened Poisson solvers (`solver::Poisson`/`PoissonND`), equidistant trigonometric directions via `with_layout(s)` (DFT + exact Fourier symbol, as order-1 HoFFT; exact per-fibre translation `shift_in_place(_dir)`); engines/solvers expose their grid via `grid()` |
| `lobatto-grid::spectral` (feature `spectral`; was the crate `lobatto-spectral`) | Screened Poisson solvers by dense fast diagonalisation (same interface as `fft::solver`, real only, small `n·r`); solvers expose their grid via `grid()` |
| `lobatto-lowrank` | Low-rank sums of products `Σ a_k ρ_k ⊗ μ_k` of fields on two grids (`LowRankSum<P, Q>`), weighted L² algebra (Gram, norm, inner product, marginals, factor maps) and QR → SVD compression (`Truncation`), point evaluation via `lobatto-grid` plans; public building blocks `Leaf<P>` (one side: functions on a grid) and `linalg` (TSQR, thin SVD, row scaling on faer) |

Conventions:
- Crates are published to crates.io individually with **version-only dependencies** between them (no path deps): publish the upstream crate first, then bump the requirement downstream.
- Temporary: the root `Cargo.toml` has `[patch.crates-io] lobatto-grid = { path = "lobatto-grid" }` so lobatto-lowrank builds against the unpublished lobatto-grid 0.1.9; remove it once 0.1.9 is on crates.io.
- Zero clippy warnings (`cargo clippy --benches --tests`); tests in release mode (`cargo test --release`).
- Benches (all in lobatto-grid): `cargo bench --bench interp` (interpolation/convection), `--bench fft_solver` and `--bench spectral_solver` (solves; the files were `solver.rs` in the two former crates, group names unchanged). Criterion group names are stable identifiers — renaming them discards baseline history. The saved baselines of the former `lobatto-fft/target/criterion` were deleted during the merge (2026-11-25).

Doc-comment math (KaTeX renders on docs.rs via each crate's `katex.html`; hover/GitHub show raw source):
- LaTeX (`$…$`/`$$…$$`) only for real formulas (eigendecompositions, quadrature weights, operators). Trivia — intervals, identifiers, index arithmetic, `O(…)` — goes in code spans: `` `[0, 1]` ``, `` `n·r` ``, `` `N_dof` ``, `` `0..=r` ``.
- First doc line (the summary) stays LaTeX-free; heavy `$$` blocks go below it.
- Inside `$$` blocks, never start a continuation line with `+`, `-`, `*`, or `1.` (markdown parses it as a list → `clippy::doc_lazy_continuation`).
- Preview locally with `RUSTDOCFLAGS="--html-in-header $(pwd)/katex.html" cargo doc --no-deps` from the workspace root (plain `cargo doc` shows raw LaTeX — the docs.rs header flag is not applied locally).
- The `katex.html` copies (workspace root, `doc/`, one per crate for packaging) must stay identical — a KaTeX version bump touches all of them.

### Merge of lobatto-fft and lobatto-spectral into lobatto-grid — done (2026-11-25, unpublished: lobatto-grid 0.1.9)

- Modules `lobatto_grid::fft` and `lobatto_grid::spectral` behind the features `fft` (`dep:rustfft`, `dep:nalgebra`) and `spectral` (`dep:nalgebra`), both in `default`; with `default-features = false` only grid + interpolation compile. The grid API is unchanged, hence 0.1.9.
- Files moved with `git mv` (history kept): sources to `src/fft/`, `src/spectral/`; tests, benches, examples and `tests/data` prefixed `fft_` / `spectral_`; test files gated with `#![cfg(feature = …)]`, benches and examples with `required-features`; the spectral test `agrees_with_lobatto_fft` needs both features.
- Inside the modules `lobatto_grid::` became `crate::`; in tests, benches, examples and doctests `lobatto_fft::` / `lobatto_spectral::` became `lobatto_grid::fft::` / `lobatto_grid::spectral::`. No formatting change (this workspace is not rustfmt-formatted).
- The crates `lobatto-fft` and `lobatto-spectral` were deleted from crates.io (2026-11-25).
- Checks: 305 tests (all features), 107 (`fft` only), 73 (`spectral` only), clippy and `cargo doc -D warnings` clean.

## TODO

### 2. Efficient solver for real solutions (realfft)

- Scope (agreed): **periodic directions first**. Use `realfft` r2c/c2r on the first periodic direction — half spectrum (n/2+1 modes), remaining directions stay complex, per-mode solves run over the reduced mode set, c2r inverse restores the real field.
- Dirichlet/Neumann directions keep the existing complex path via `solve_in_place_real` (the copy-through wrapper); DST/DCT real-to-real transforms for Odd/Even extensions are deliberately **out of scope** for this item.
- Acceptance: bitwise/1e-15 agreement with `solve_in_place` real part; target ≥1.5× on all-periodic real problems; measure with `hofft_solve_in_place_5d` in `benches/solver.rs` (scale `n` up for big runs; written for 1.9M DOFs to keep iteration fast).

### 3. Solver efficiency study — profile before optimizing

- Hypothesis to verify: `solve_in_place` is **memory-bound / parallel-inefficient**, not compute-bound (~25 Melem/s at 1.9M-DOF 5D on the dev machine).
- Deliverable is a measurement report, not code:
  - Per-stage timings: mass scaling, each direction's forward/inverse sweep, mode solves.
  - Thread-scaling curve (1/2/4/8/all cores) per stage.
  - Achieved memory bandwidth per stage vs machine peak.
  - Allocation counts in the hot path. Read-through count, to be confirmed by `dhat`: `EngineND::forward`/`inverse` allocate **three** buffers per fibre (the gathered fibre `Vec`, the `r·n_fft` `buf` inside `Engine::forward`/`inverse`, and rustfft's internal scratch — `process` is used, never `process_with_scratch`); the lobatto-fft symbol solve allocates a `DVector` per mode tuple plus one temporary per direction per sub-fibre from `ad_mul`; the lobatto-spectral `PoissonND::sweep` allocates two `Vec`s per fibre.
- Tooling: `dhat` (or `jemalloc-ctl`) for allocation bytes and counts on a single `solve_in_place`; Instruments Time Profiler on the macOS dev machine, `perf stat -e task-clock,cycles,cache-misses,dTLB-load-misses` on Linux. Glibc knobs such as `MALLOC_ARENA_MAX` don't apply on macOS. Measure with no other session active on the checkout.
- Configurations: the existing `hofft_solve_in_place_3d` and `_5d` groups with `n` scaled up (no new bench group); the fixed per-allocation cost dominates at small `dim_d`, which is also where the Rayon fan-out is widest, so include a small-`n·r` case.
- Suspects the profile should confirm or kill (optimization items get written from the data, in this order of suspicion):
  - N full-array strided sweeps per transform direction (2N per solve) — candidate for blocked transposes / fused passes.
  - Per-fiber gather→`Vec`→FFT→scatter instead of batched contiguous FFTs with `process_with_scratch`.
  - Per-mode-tuple heap allocations in step 2.

### 4. lobatto-spectral solver

- **Everything configurable**: each per-direction choice is an explicit option, never an automatic decision. In particular:
  - Node layout: Gauss–Lobatto elements vs equidistant points (the latter only for `Periodic`).
  - Diagonalisation: FFT vs dense eigendecomposition (vs banded/blocked eigensolve if it lands).
  - Interpolation: Lagrange (element-local) vs trigonometric.
  - Selection is per direction, mixing allowed, through the constructor (e.g. an enum next to `BoundaryCondition`); current behaviour is the default so existing code keeps building.
- **Fourier directions — superseded (2026-11-03)**: implemented in lobatto-grid 0.1.8 (`Grid1D::trigonometric`, `NodeLayout`) and lobatto-fft 0.5.5 (`Poisson::with_layout`, `PoissonND::with_layouts`, `EngineND::with_layouts`), see "Trigonometric directions" below; deliberately **not** in lobatto-spectral. Original text, kept for reference: when a direction is `Periodic`, replace the Gauss–Lobatto elements by `n·r` equidistant points on `[x_l, x_l + L)` and diagonalise with a plain FFT (`realfft` r2c/c2r, real field) instead of the dense `n·r × n·r` eigendecomposition.
  - Mass is uniform (`L/(n·r)`), stiffness eigenvalues are `k²` with `k = 2πm/L`; the pseudo-inverse rule for the constant mode carries over unchanged.
  - Design constraint: `Grid1D` assumes Lobatto nodes per element and owns interpolation (`eval`/`convect`), so a Fourier direction needs either a `Grid1D` node-layout variant or a trigonometric-interpolation path; `grid()` / `get_x()` must keep reporting the true nodes.
  - Mixed directions (Fourier in some, dense diagonalisation in the others) stay under the same `PoissonND` interface.
  - Acceptance: spectral accuracy on smooth periodic manufactured solutions; `O(n·r·log(n·r))` per DOF per Fourier direction; `apply_in_place`/`exp_in_place`/`euler_*` work unchanged through the per-mode eigenvalues.
- **Cost at larger `n·r`**: the dense eigendecomposition (`O((n·r)³)` setup) and the dense per-direction transforms (`O(n·r)` per DOF) make the crate's sweet spot "up to about a hundred" DOFs per direction.
  - Candidates: a banded or blocked eigensolve exploiting the element-local sparsity of `K`, or a crossover benchmark against `lobatto-fft` on identical problems that documents, per `n·r` and `N`, where each crate wins (recorded in `doc/`).
  - The benchmark comes first: it decides whether the eigensolve work is worth doing at all.


### 5. Allocation-free hot paths — done (2026-10-23)

Per-task scratch (`for_each_init`) in `EngineND` sweeps (via `Engine::forward_with`/`inverse_with`, `process_with_scratch`), the lobatto-fft per-mode step (`ad_mul_to`/`mul_to`), and the lobatto-spectral `sweep`; contiguous `d == 0` path; fibre/mode-tuple disjointness checked by unit tests. No internal grain-size thresholds (`with_min_len` batching was removed on purpose: hardware dependent, and the caller controls threads through the Rayon pool). Bit-exact against `tests/reference.rs` (stored pre-optimisation solutions) in both crates; no new `unsafe`.

Measured on a 5D, 1.1M-DOF mixed-BC solve (14 threads, counting allocator / wall time, outside Criterion because of concurrent sessions):
- lobatto-fft `apply_in_place_dirs`: 6.5M → 4.7k allocations, ~39–49 ms → 14 ms (sweeps 16.5 → 3.3 ms; per-mode step ~32 → 11 ms). Item 3's read-through was 2 buffers per fibre, not 3: rustfft allocates no scratch at these FFT lengths.
- lobatto-spectral: 1.37M → 4.1k allocations, 11.5–14 ms → 6.1 ms.

Follow-up, index arithmetic out of the per-mode step (lobatto-fft 0.5.4, lobatto-spectral 0.2.5, on lobatto-grid 0.1.7 helpers `nd_multi_indices` / `nd_multi_index`; bit-identical to 0.5.3 / 0.2.4):
- lobatto-fft: each mode tuple's compact-buffer offsets, coefficients and canonical entries are built once (`EngineND::mode_indices`, per-direction tables of the independent modes) and shared by the gather and the scatter; mass scaling per fibre along direction 0 (`mass_index` tables); `g` evaluated over `nd_multi_indices`. Gather + scatter alone 2.3 → 0.8 ms; full solve ×1.1–1.24 (5D r=4 mixed 16.9 → 14.9 ms).
- lobatto-spectral: mass scaling and eigenvalue step per fibre along direction 0, ×1.05–1.15.

What remains in the lobatto-fft per-mode step is mostly the small `r_d × r_d` products (~6 ms of ~14 ms, 5D r=4), close to compute-bound (~130 GFLOP/s with plain loops). Rejected: batching mode tuples along `k_0` (shared `V_{k_d}` for d ≥ 1) — too complex to maintain, bit-exactness would need hand-written products reproducing nalgebra's `dotc`/`gemv` summation order. Plain loops instead of nalgebra change rounding. Fewer-operation routes: realfft (item 2), real eigenvectors in Odd/Even directions (unchecked).

### Trigonometric directions — done (2026-11-03, unpublished: lobatto-grid 0.1.8, lobatto-fft 0.5.5)

Requested by Odeon (grid filter of `ode-observers`): periodic directions as trigonometric polynomials on equidistant points, bounded directions kept as Gauss–Lobatto elements. Design: `n ≥ 2` points `x_j = x_l + j·L/n` on `[x_l, x_l + L)`, modes `−⌊(n−1)/2⌋..=⌊n/2⌋`, Nyquist mode of an even `n` as a cosine (real data → real interpolant).
- lobatto-grid: `NodeLayout { GaussLobatto (default), Equidistant }`, `Grid1D::trigonometric(n, l, xl)`, `layout()`, `local_count()`. An equidistant grid is **one element spanning the period** (`n() = 1`, `r() = n_dof() = local_count() = n`, `h() = L`, `Periodic`, weights `L/n`), so `EvalGeometry`, gather, `contract`, `EvalPlan` are shared; only `fill_phi`, `eval_at`, `EvalGeometry` counts and `derivative` branch on the layout.
  - Interpolation: barycentric trigonometric formula (Henrici 1979, Berrut 1984), `κ = cot` (n even) / `csc` (n odd), from the nearest node with a `(cos, sin)(πm/n)` table (no cancellation near nodes); `O(n)` per point per equidistant direction.
  - Derivative: realfft r2c → `ik_m` (Nyquist 0) → c2r, `O(n log n)`; `GridND::derivative` plans once per call with per-task buffers, `Grid1D::derivative` plans on every call (~0.5 µs, slower than the old dense `D` for `n ≤ 32`; a per-thread planner cache would remove it). Replaced the dense antisymmetric `D` (benchmark in `~/Claude/trig-deriv-realfft`: 1.2–330× on 1M-DOF `GridND`, more accurate at large `n`, 5e-13 vs 4e-11 at `n = 2048`). Same operator as the old `D` in exact arithmetic, so `W D + Dᵀ W = 0` (Odeon's skew-symmetric transport stability proof) is unaffected; only the matrix probed column by column is now antisymmetric to rounding instead of entry-for-entry (the computed `uᵀ W D u` was already rounding-level, not 0, with the dense `D`).
- lobatto-fft: `Poisson::with_layout`, `PoissonND::with_layouts`, `Engine::trigonometric`, `EngineND::with_layouts` (`new` = all Gauss–Lobatto). An equidistant direction has `n·r` points and is the order-1 periodic HoFFT (plain DFT), mass `L/N`, the **true symbol** `μ_m = (2πm/L)²` with Nyquist included (not the eigenvalues of `D²`, which would leave Nyquist undamped); every solve/operator works unchanged through it, so `apply_mu_in_place_dir` is the Fourier multiplier. `Equidistant` requires `Periodic` (panics otherwise).
  - Exact translation `u(x) ← u(x − c)`: `Poisson::shift_in_place(f, c)`, `PoissonND::shift_in_place_dir(d, f, c: &[f64])` (one shift per fibre, `nd_fiber_base` order), phase `e^{−ik_m c}`, Nyquist by its real part, equal to `convect` with `x_d ↦ x_d − c`. Shares the private `for_each_fibre` loop with `apply_mu_in_place_dir`.
- Tests: `lobatto-grid/tests/trigonometric.rs` (cardinal property, exactness, DFT interpolant incl. points at/near nodes, derivative vs DFT, spectral convergence of `e^{sin θ}`, mixed 2D/3D plans and derivatives, constant shift = Fourier translation) and `lobatto-fft/tests/trigonometric.rs` (DFT engine, manufactured Fourier modes incl. Nyquist, mixed 3D vs shifted GL solve, directional multiplier, real data, shifts). Existing Gauss–Lobatto results bit-identical (reference tests).
- Not done: lobatto-spectral (by decision). Publish lobatto-grid 0.1.8 first, then lobatto-fft 0.5.5, then drop the temporary `[patch.crates-io]`.

### Deferred / smaller items

- One-shot `eval` grouping: the serial `HashMap` pass remains; agreed candidate designs are hybrid counting-sort (when `n_elems ≲ c·m`) vs parallel sort-by-key. Moot for the plan path (grouping happens at plan build).
- `EvalPlan` φ-caching option (store basis values, ~`Σ(r_d+1)·8` B/point ≈ 1.6 GB at 10M pts r=4 4D): only worth testing if the apply proves tolerant of the extra bandwidth.
- Two-for-one real solve (`solve_pair`: pack `f1 + i·f2`): cheap 2× when applications solve two real fields per step; superseded for all-periodic problems if TODO 2 lands.
