# lobatto-lowrank

Low-rank sums of products of fields on two Gauss-Lobatto grids, with weighted
QR/SVD compression.

A field on an `N`-dimensional grid is split between a left grid of dimension
`P` and a right grid of dimension `Q = N − P`, and stored as

    ρ(x) = Σ_k a_k · ρ_k(x_1, …, x_P) · μ_k(x_{P+1}, …, x_N)

Norms and compressions use the discrete L² inner product given by the tensor
Gauss-Lobatto weights of each grid. Grids come from
[`lobatto-grid`](https://crates.io/crates/lobatto-grid).

Status: structure only, algorithms not yet implemented.

## License

LGPL-3.0-only.
