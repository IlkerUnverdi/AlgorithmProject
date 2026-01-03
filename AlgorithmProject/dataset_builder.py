# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
from typing import List, Tuple, Optional

from tsp_core import (
    Point,
    TSPInstance,
    build_distance_matrix,
    save_instance_json,
    load_instance_json,
    tour_length,
    is_valid_tour,
)


class DatasetBuilder:
    def __init__(
        self,
        base_dir: str = os.path.join("data", "instances"),
        x_range: Tuple[float, float] = (0.0, 1000.0),
        y_range: Tuple[float, float] = (0.0, 1000.0),
    ) -> None:
        self.base_dir = base_dir
        self.x_range = x_range
        self.y_range = y_range

    def generate_points(
        self,
        n: int,
        seed: int,
        mode: str = "uniform",
        clusters: int = 3,
    ) -> List[Point]:
        rng = random.Random(seed)

        if mode == "uniform":
            return [(rng.uniform(*self.x_range), rng.uniform(*self.y_range)) for _ in range(n)]

        if mode == "cluster":
            centers: List[Point] = [
                (rng.uniform(*self.x_range), rng.uniform(*self.y_range)) for _ in range(clusters)
            ]
            spread = 0.08 * (self.x_range[1] - self.x_range[0])

            pts: List[Point] = []
            for _ in range(n):
                cx, cy = centers[rng.randrange(clusters)]
                x = rng.gauss(cx, spread)
                y = rng.gauss(cy, spread)

                x = min(max(x, self.x_range[0]), self.x_range[1])
                y = min(max(y, self.y_range[0]), self.y_range[1])
                pts.append((x, y))
            return pts

        raise ValueError(f"Unknown mode: {mode}")

    def generate_instance(
        self,
        n: int,
        seed: int,
        name: Optional[str] = None,
        mode: str = "uniform",
        clusters: int = 3,
    ) -> TSPInstance:
        if n < 3:
            raise ValueError("TSP icin n en az 3 olmali.")

        if name is None:
            name = f"tsp_{mode}_n{n}_seed{seed}"

        points = self.generate_points(n=n, seed=seed, mode=mode, clusters=clusters)
        dist = build_distance_matrix(points)
        return TSPInstance(name=name, n=n, seed=seed, points=points, dist=dist)

    def build_default_dataset(self, force: bool = False) -> None:
        specs = [
            ("small", 10, 42, "uniform"),
            ("medium", 50, 43, "uniform"),
            ("large", 100, 44, "uniform"),
            ("medium_cluster", 50, 45, "cluster"),
        ]

        for tag, n, seed, mode in specs:
            inst = self.generate_instance(
                n=n,
                seed=seed,
                name=f"{tag}_{mode}_n{n}_seed{seed}",
                mode=mode,
            )
            out_path = os.path.join(self.base_dir, f"{inst.name}.json")

            saved = save_instance_json(inst, out_path, force=force)
            if saved:
                print(f"[OK] saved: {out_path} (n={n}, mode={mode})")
            else:
                print(f"[SKIP] exists: {out_path}")

        small_path = os.path.join(self.base_dir, "small_uniform_n10_seed42.json")
        if os.path.exists(small_path):
            inst = load_instance_json(small_path)
            tour = list(range(inst.n))
            random.Random(123).shuffle(tour)
            print("Random tour valid?", is_valid_tour(inst.n, tour))
            print("Random tour length:", tour_length(inst.dist, tour))
        else:
            print("[WARN] small instance not found for quick check.")


if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_default_dataset(force=False)
