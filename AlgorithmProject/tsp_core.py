# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any


Point = Tuple[float, float]


@dataclass
class TSPInstance:
    name: str
    n: int
    seed: int
    points: List[Point]
    dist: List[List[float]]


def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_distance_matrix(points: List[Point]) -> List[List[float]]:
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(points[i], points[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def save_instance_json(inst: TSPInstance, path: str, force: bool = False) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if (not force) and os.path.exists(path):
        return False

    payload: Dict[str, Any] = asdict(inst)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return True


def load_instance_json(path: str) -> TSPInstance:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return TSPInstance(
        name=str(payload["name"]),
        n=int(payload["n"]),
        seed=int(payload["seed"]),
        points=[(float(x), float(y)) for x, y in payload["points"]],
        dist=[[float(v) for v in row] for row in payload["dist"]],
    )


def tour_length(dist: List[List[float]], tour: List[int]) -> float:
    n = len(tour)
    if n == 0:
        return 0.0

    total = 0.0
    for i in range(n - 1):
        total += dist[tour[i]][tour[i + 1]]
    total += dist[tour[-1]][tour[0]]
    return total


def is_valid_tour(n: int, tour: List[int]) -> bool:
    if len(tour) != n:
        return False
    s = set(tour)
    return len(s) == n and min(s) == 0 and max(s) == n - 1
