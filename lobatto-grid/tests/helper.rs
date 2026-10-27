//! Tests for the ND indexing helpers `grid::nd_col_major_strides`,
//! `grid::nd_fiber_base` and `grid::nd_multi_indices`.

use lobatto_grid::grid::{nd_col_major_strides, nd_fiber_base, nd_multi_indices};
use lobatto_grid::{Extension, GridND};

fn grid_4d() -> GridND<4> {
    GridND::new(
        [2, 3, 1, 2],
        [3, 2, 4, 2],
        [1.0; 4],
        [0.0; 4],
        [Extension::Periodic, Extension::Odd, Extension::Even, Extension::Even],
    )
}

#[test]
fn strides_are_column_major() {
    assert_eq!(nd_col_major_strides(&[7]), [1]);
    assert_eq!(nd_col_major_strides(&[4, 3, 5, 2]), [1, 4, 12, 60]);
    // A size-1 direction leaves the next stride unchanged.
    assert_eq!(nd_col_major_strides(&[3, 1, 2]), [1, 3, 3]);
    let g = grid_4d();
    assert_eq!(nd_col_major_strides(&g.ndofs()), g.strides());
}

#[test]
fn strides_map_multi_indices_bijectively() {
    // Σ_d i_d·st[d] over every multi-index hits each flat index once, in
    // order when direction 0 varies fastest.
    let sizes = [4, 3, 5];
    let st = nd_col_major_strides(&sizes);
    let mut p = 0;
    for i2 in 0..sizes[2] {
        for i1 in 0..sizes[1] {
            for i0 in 0..sizes[0] {
                assert_eq!(i0 * st[0] + i1 * st[1] + i2 * st[2], p);
                p += 1;
            }
        }
    }
    assert_eq!(p, sizes.iter().product::<usize>());
}

#[test]
fn fiber_base_matches_digit_loop_and_covers_tensor_disjointly() {
    let g = grid_4d();
    let (sizes, strides) = (g.ndofs(), g.strides());
    for d in 0..4 {
        let mut seen = vec![false; g.total()];
        for s in 0..g.total() / sizes[d] {
            // Reference: peel the digits of s over the directions other than d.
            let (mut expected, mut rem) = (0, s);
            for d2 in (0..4).filter(|&d2| d2 != d) {
                expected += (rem % sizes[d2]) * strides[d2];
                rem /= sizes[d2];
            }
            let base = nd_fiber_base(s, d, &sizes, &strides);
            assert_eq!(base, expected, "d={d} s={s}");
            for j in 0..sizes[d] {
                let p = base + j * strides[d];
                assert!(!seen[p], "d={d}: index {p} hit twice");
                seen[p] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "d={d}: tensor not covered");
    }
}

#[test]
fn multi_indices_are_the_column_major_digits_of_each_flat_index() {
    let g = grid_4d();
    let (sizes, strides) = (g.ndofs(), g.strides());
    let all: Vec<[usize; 4]> = nd_multi_indices(sizes).collect();
    assert_eq!(all.len(), g.total());
    for (p, i) in all.iter().enumerate() {
        let expected: [usize; 4] = std::array::from_fn(|d| (p / strides[d]) % sizes[d]);
        assert_eq!(*i, expected, "flat index {p}");
        assert_eq!((0..4).map(|d| i[d] * strides[d]).sum::<usize>(), p);
    }
    // One direction, a size-1 direction, an empty shape, no direction.
    assert!(nd_multi_indices([5]).eq((0..5).map(|j| [j])));
    assert!(nd_multi_indices([2, 1, 2]).eq([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]));
    assert_eq!(nd_multi_indices([3, 0, 2]).count(), 0);
    assert!(nd_multi_indices([]).eq([[]]));
}
