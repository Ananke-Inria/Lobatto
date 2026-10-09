# QuadratiX workspace

Rust workspace for Gauss–Lobatto spectral-element numerics.

| Crate | Role |
|---|---|
| `lobatto` | Gauss–Lobatto points, quadrature, barycentric/Lagrange basis utilities |
| `lobatto-grid` | Grid geometry (`Grid1D`/`GridND<N>` on arbitrary domains, `Extension` folding, DOF indexing) and interpolation (`eval`/`eval_extended`/`convect`, `EvalPlan`, `Scalar`) |
| `lobatto-fft` | HoFFT transforms (`hofft::Engine`/`EngineND`), screened Poisson solvers (`solver::Poisson`/`PoissonND`); engines/solvers expose their grid via `grid()` |
| `lobatto-spectral` | Screened Poisson solvers by dense fast diagonalisation (same interface as `lobatto-fft::solver`, real only, small `n·r`); solvers expose their grid via `grid()` |
| `lobatto-lowrank` | Low-rank sums of products `Σ a_k ρ_k ⊗ μ_k` of fields on two grids (`LowRankSum<P, Q>`), weighted L² norms and QR → SVD compression (`Truncation`); structure only, algorithms are `todo!()` |

Conventions:
- Crates are published to crates.io individually with **version-only dependencies** between them (no path deps): publish the upstream crate first, then bump the requirement downstream.
- Zero clippy warnings (`cargo clippy --benches --tests`); tests in release mode (`cargo test --release`).
- Benches: `cargo bench --bench interp` in lobatto-grid (interpolation/convection), `cargo bench --bench solver` in lobatto-fft (solves). Criterion group names are stable identifiers — renaming them discards baseline history.

Doc-comment math (KaTeX renders on docs.rs via each crate's `katex.html`; hover/GitHub show raw source):
- LaTeX (`$…$`/`$$…$$`) only for real formulas (eigendecompositions, quadrature weights, operators). Trivia — intervals, identifiers, index arithmetic, `O(…)` — goes in code spans: `` `[0, 1]` ``, `` `n·r` ``, `` `N_dof` ``, `` `0..=r` ``.
- First doc line (the summary) stays LaTeX-free; heavy `$$` blocks go below it.
- Inside `$$` blocks, never start a continuation line with `+`, `-`, `*`, or `1.` (markdown parses it as a list → `clippy::doc_lazy_continuation`).
- Preview locally with `RUSTDOCFLAGS="--html-in-header $(pwd)/katex.html" cargo doc --no-deps` from the workspace root (plain `cargo doc` shows raw LaTeX — the docs.rs header flag is not applied locally).
- The `katex.html` copies (workspace root, `doc/`, one per crate for packaging) must stay identical — a KaTeX version bump touches all of them.

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
- **Fourier directions**: when a direction is `Periodic`, replace the Gauss–Lobatto elements by `n·r` equidistant points on `[x_l, x_l + L)` and diagonalise with a plain FFT (`realfft` r2c/c2r, real field) instead of the dense `n·r × n·r` eigendecomposition.
  - Mass is uniform (`L/(n·r)`), stiffness eigenvalues are `k²` with `k = 2πm/L`; the pseudo-inverse rule for the constant mode carries over unchanged.
  - Design constraint: `Grid1D` assumes Lobatto nodes per element and owns interpolation (`eval`/`convect`), so a Fourier direction needs either a `Grid1D` node-layout variant or a trigonometric-interpolation path; `grid()` / `get_x()` must keep reporting the true nodes.
  - Mixed directions (Fourier in some, dense diagonalisation in the others) stay under the same `PoissonND` interface.
  - Acceptance: spectral accuracy on smooth periodic manufactured solutions; `O(n·r·log(n·r))` per DOF per Fourier direction; `apply_in_place`/`exp_in_place`/`euler_*` work unchanged through the per-mode eigenvalues.
- **Cost at larger `n·r`**: the dense eigendecomposition (`O((n·r)³)` setup) and the dense per-direction transforms (`O(n·r)` per DOF) make the crate's sweet spot "up to about a hundred" DOFs per direction.
  - Candidates: a banded or blocked eigensolve exploiting the element-local sparsity of `K`, or a crossover benchmark against `lobatto-fft` on identical problems that documents, per `n·r` and `N`, where each crate wins (recorded in `doc/`).
  - The benchmark comes first: it decides whether the eigensolve work is worth doing at all.


### 5. Allocation-free hot paths (candidate fixes — gated on item 3's data, apply in this order)

Sites: `EngineND::forward`/`inverse` and `Engine::forward`/`inverse` in `lobatto-fft/src/hofft.rs`; the symbol solve in `lobatto-fft/src/solver.rs` (`PoissonND::solve_in_place`); `PoissonND::sweep` in `lobatto-spectral/src/solver.rs`. Same gather → `Vec` → transform → scatter pattern everywhere.

- Thread-local scratch via `rayon::for_each_init` at every per-fibre / per-mode closure; fill with a plain index loop (no `collect`/`from_fn`). Extra memory `O(n_threads · dim_d)`.
- rustfft scratch: `Engine::forward`/`inverse` take a scratch argument (their `buf` plus `get_inplace_scratch_len()`) and call `process_with_scratch`. `Engine` is shared by `&self` across threads, so scratch is passed in, not stored.
- Contiguous fast path for `d == 0` (`stride == 1` in the column-major layout): `par_chunks_mut(dim_d)`, no gather/scatter, no `unsafe`. One of the `N` directions per sweep in both crates.
- Batch several fibres per Rayon task when `dim_d` is small (amortises task dispatch).
- Symbol solve: hoist the `ad_mul` temporary; one per-thread scratch for the whole mode tuple.
- Real-input transform: already item 2.
- Guard: a `debug_assert!` (once per shape, not per task) that the `nd_fiber_base` index sets are disjoint across `s`, so the `unsafe` invariant is checkable.
- Document the scratch strategy in the module docs of `hofft.rs` and both `solver.rs` so it isn't reverted by accident.
- Acceptance: allocator traffic per `solve_in_place` down ≥ 90 % (`dhat`); wall time down ≥ 30 % on the item-3 configurations; bit-exact or ≤ 1 ULP vs the pre-optimisation output on the convergence tests (store a reference solution under `tests/`); no new `unsafe`; `cargo test --release --workspace` and clippy clean.

### Deferred / smaller items

- One-shot `eval` grouping: the serial `HashMap` pass remains; agreed candidate designs are hybrid counting-sort (when `n_elems ≲ c·m`) vs parallel sort-by-key. Moot for the plan path (grouping happens at plan build).
- `EvalPlan` φ-caching option (store basis values, ~`Σ(r_d+1)·8` B/point ≈ 1.6 GB at 10M pts r=4 4D): only worth testing if the apply proves tolerant of the extra bandwidth.
- Two-for-one real solve (`solve_pair`: pack `f1 + i·f2`): cheap 2× when applications solve two real fields per step; superseded for all-periodic problems if TODO 2 lands.
