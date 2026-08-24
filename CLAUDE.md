# QuadratiX workspace

Rust workspace for Gauss–Lobatto spectral-element numerics.

| Crate | Role |
|---|---|
| `lobatto` | Gauss–Lobatto points, quadrature, barycentric/Lagrange basis utilities |
| `lobatto-grid` | Grid geometry (`Grid1D`/`GridND<N>` on arbitrary domains, `Extension` folding, DOF indexing) and interpolation (`eval`/`eval_extended`/`convect`, `EvalPlan`, `Scalar`) |
| `lobatto-fft` | HoFFT transforms (`hofft::Engine`/`EngineND`), screened Poisson solvers (`solver::Poisson`/`PoissonND`); engines/solvers expose their grid via `grid()` |
| `lobatto-spectral` | High-order screened Poisson solvers on Gauss–Lobatto grids |

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
  - Allocation counts in the hot path (per-fiber `Vec`s in `forward`/`inverse`, per-mode-tuple `DVector` in the symbol solve).
- Suspects the profile should confirm or kill (optimization items get written from the data, in this order of suspicion):
  - N full-array strided sweeps per transform direction (2N per solve) — candidate for blocked transposes / fused passes.
  - Per-fiber gather→`Vec`→FFT→scatter instead of batched contiguous FFTs with `process_with_scratch`.
  - Per-mode-tuple heap allocations in step 2.

### Deferred / smaller items

- One-shot `eval` grouping: the serial `HashMap` pass remains; agreed candidate designs are hybrid counting-sort (when `n_elems ≲ c·m`) vs parallel sort-by-key. Moot for the plan path (grouping happens at plan build).
- `EvalPlan` φ-caching option (store basis values, ~`Σ(r_d+1)·8` B/point ≈ 1.6 GB at 10M pts r=4 4D): only worth testing if the apply proves tolerant of the extra bandwidth.
- Two-for-one real solve (`solve_pair`: pack `f1 + i·f2`): cheap 2× when applications solve two real fields per step; superseded for all-periodic problems if TODO 2 lands.
