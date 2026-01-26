import math
import random
from typing import List, Optional, Tuple


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


def sample_points_in_rect(
    x_min: float, y_min: float, x_max: float, y_max: float, num_points: int
) -> List[Tuple[float, float]]:
    return [
        (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
        for _ in range(num_points)
    ]


def farthest_point_sampling(
    candidates: List[Tuple[float, float]],
    N: int,
    init_point: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    if not candidates:
        return []
    if init_point is None:
        centroid = (
            sum(x for x, _ in candidates) / len(candidates),
            sum(y for _, y in candidates) / len(candidates),
        )
        init_point = min(
            candidates,
            key=lambda p: math.hypot(p[0] - centroid[0], p[1] - centroid[1]),
        )
    chosen = [init_point]
    candidates = [c for c in candidates if c != init_point]
    while len(chosen) < N and candidates:
        best_candidate = max(
            candidates,
            key=lambda c: min(math.hypot(c[0] - p[0], c[1] - p[1]) for p in chosen),
        )
        chosen.append(best_candidate)
        candidates.remove(best_candidate)
    return chosen


def compute_ideal_radius(allowed_area: float, N: int) -> float:
    if N <= 0:
        return 0
    circle_area = allowed_area / N
    return math.sqrt(circle_area / math.pi)
