# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import time
from itertools import permutations
from typing import List, Dict, Any, Tuple

from tsp_core import load_instance_json, tour_length


class TSPBruteForceSolver:
    """
    Exact TSP solver using brute-force enumeration.
    Intended ONLY for small instances (n <= ~10).
    """

    def __init__(self, opt_dir: str = os.path.join("data", "opt")) -> None:
        self.opt_dir = opt_dir

    def _opt_path_for_instance(self, instance_name: str) -> str:
        os.makedirs(self.opt_dir, exist_ok=True)
        return os.path.join(self.opt_dir, f"{instance_name}_bruteforce_opt.json")

    def solve_and_save(self, instance_path: str, force: bool = False) -> Dict[str, Any]:
        inst = load_instance_json(instance_path)
        out_path = self._opt_path_for_instance(inst.name)

        if (not force) and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            print(f"[SKIP] brute-force optimum already exists for {inst.name}")
            print("best_length :", payload["best_length"])
            print("runtime_sec :", payload["runtime_sec"])
            return payload

        print(f"[RUN] brute-force TSP solving: {inst.name} (n={inst.n})")

        t0 = time.time()
        best_len, best_tour = self._solve_by_bruteforce(inst.dist)
        t1 = time.time()

        payload: Dict[str, Any] = {
            "instance_name": inst.name,
            "method": "Exact TSP via Brute Force Enumeration",
            "n": inst.n,
            "best_length": best_len,
            "best_tour": best_tour,
            "runtime_sec": round(t1 - t0, 6),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("[OK] brute-force optimum saved")
        print("best_length :", best_len)
        print("runtime_sec :", payload["runtime_sec"])

        return payload

    def _solve_by_bruteforce(self, dist: List[List[float]]) -> Tuple[float, List[int]]:
        n = len(dist)
        if n < 3:
            raise ValueError("n must be >= 3")

        best_len = math.inf
        best_tour: List[int] = []

        nodes = list(range(1, n))  # fix start city at 0

        for perm in permutations(nodes):
            tour = [0] + list(perm)
            length = tour_length(dist, tour)
            if length < best_len:
                best_len = length
                best_tour = tour

        return best_len, best_tour
