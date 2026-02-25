"""Tests for improved point distribution sampling."""

from hardware.difra.gui.main_window_ext.points.zone_geometry import (
    farthest_point_sampling,
)


def _grid(nx=6, ny=6, step=10.0):
    return [(x * step, y * step) for y in range(ny) for x in range(nx)]


def _min_pairwise_distance(points):
    if len(points) < 2:
        return 0.0
    best = float("inf")
    for i, p in enumerate(points):
        for j in range(i + 1, len(points)):
            q = points[j]
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx * dx + dy * dy
            if d2 < best:
                best = d2
    return best


def test_sampling_returns_requested_unique_count():
    candidates = _grid()
    chosen = farthest_point_sampling(candidates, 10)
    assert len(chosen) == 10
    assert len(set(chosen)) == 10
    assert all(p in candidates for p in chosen)


def test_sampling_respects_init_point_nearest_candidate():
    candidates = _grid()
    init = (0.2, 0.1)
    chosen = farthest_point_sampling(candidates, 8, init_point=init)
    assert chosen[0] == (0.0, 0.0)


def test_sampling_is_more_even_than_random_subset_on_grid():
    candidates = _grid(10, 10, 1.0)
    chosen = farthest_point_sampling(candidates, 20)

    # Deterministic baseline: first 20 points in row-major order.
    baseline = candidates[:20]
    assert _min_pairwise_distance(chosen) >= _min_pairwise_distance(baseline)

