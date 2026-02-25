import math
import random
from typing import List, Optional, Sequence, Tuple


def sample_points_in_circle(
    center: Tuple[float, float], radius: float, num_points: int
) -> List[Tuple[float, float]]:
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.random()) * radius
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append((x, y))
    return points


def sample_points_in_ellipse(
    center: Tuple[float, float], radius_x: float, radius_y: float, num_points: int
) -> List[Tuple[float, float]]:
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.random())
        x = center[0] + r * radius_x * math.cos(angle)
        y = center[1] + r * radius_y * math.sin(angle)
        points.append((x, y))
    return points


def sample_points_in_rect(
    x_min: float, y_min: float, x_max: float, y_max: float, num_points: int
) -> List[Tuple[float, float]]:
    return [
        (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
        for _ in range(num_points)
    ]


def _distance_sq(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _nearest_candidate_index(
    candidates: Sequence[Tuple[float, float]], point: Tuple[float, float]
) -> int:
    return min(
        range(len(candidates)),
        key=lambda i: _distance_sq(candidates[i], point),
    )


def _distribution_score(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Score spread quality: maximize min NN distance, then mean NN, then uniformity."""
    if len(points) < 2:
        return (0.0, 0.0, 0.0)

    nearest = []
    for i, p in enumerate(points):
        best = float("inf")
        for j, q in enumerate(points):
            if i == j:
                continue
            d2 = _distance_sq(p, q)
            if d2 < best:
                best = d2
        nearest.append(best)

    min_nn = min(nearest)
    mean_nn = sum(nearest) / len(nearest)
    variance = sum((d - mean_nn) ** 2 for d in nearest) / len(nearest)
    return (min_nn, mean_nn, -variance)


def _farthest_from_seed(
    candidates: Sequence[Tuple[float, float]], n: int, seed_idx: int
) -> List[Tuple[float, float]]:
    if n <= 0 or not candidates:
        return []
    if n == 1:
        return [candidates[seed_idx]]

    chosen_indices = [seed_idx]
    min_d2 = [_distance_sq(p, candidates[seed_idx]) for p in candidates]
    min_d2[seed_idx] = -1.0

    while len(chosen_indices) < n:
        next_idx = max(range(len(candidates)), key=lambda i: min_d2[i])
        if min_d2[next_idx] < 0:
            break
        chosen_indices.append(next_idx)
        min_d2[next_idx] = -1.0

        new_point = candidates[next_idx]
        for i, d2 in enumerate(min_d2):
            if d2 < 0:
                continue
            nd2 = _distance_sq(candidates[i], new_point)
            if nd2 < d2:
                min_d2[i] = nd2

    return [candidates[i] for i in chosen_indices]


def _unique_candidates(
    candidates: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    unique = []
    seen = set()
    for p in candidates:
        key = (float(p[0]), float(p[1]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def farthest_point_sampling(
    candidates: List[Tuple[float, float]],
    N: int,
    init_point: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    candidates = _unique_candidates(candidates)
    if not candidates:
        return []
    if N <= 0:
        return []
    if N >= len(candidates):
        return list(candidates)

    if init_point is not None:
        seed_idx = _nearest_candidate_index(candidates, init_point)
        return _farthest_from_seed(candidates, N, seed_idx)

    centroid = (
        sum(x for x, _ in candidates) / len(candidates),
        sum(y for _, y in candidates) / len(candidates),
    )

    # Multi-seed deterministic starts to reduce spatial bias and improve uniformity.
    extrema_seed_indices = [
        min(range(len(candidates)), key=lambda i: candidates[i][0] + candidates[i][1]),
        max(range(len(candidates)), key=lambda i: candidates[i][0] + candidates[i][1]),
        min(range(len(candidates)), key=lambda i: candidates[i][0] - candidates[i][1]),
        max(range(len(candidates)), key=lambda i: candidates[i][0] - candidates[i][1]),
        _nearest_candidate_index(candidates, centroid),
        max(range(len(candidates)), key=lambda i: _distance_sq(candidates[i], centroid)),
    ]

    seed_indices = []
    seen = set()
    for idx in extrema_seed_indices:
        if idx in seen:
            continue
        seen.add(idx)
        seed_indices.append(idx)

    best = None
    best_score = None
    for seed_idx in seed_indices:
        chosen = _farthest_from_seed(candidates, N, seed_idx)
        score = _distribution_score(chosen)
        if best is None or score > best_score:
            best = chosen
            best_score = score

    return best if best is not None else []


def compute_ideal_radius(allowed_area: float, N: int) -> float:
    if N <= 0:
        return 0
    circle_area = allowed_area / N
    return math.sqrt(circle_area / math.pi)
