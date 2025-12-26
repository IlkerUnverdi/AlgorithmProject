# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import List, Dict, Any, Tuple, Optional

from tsp_core import load_instance_json, tour_length


class TSPSimulatedAnnealingSolver:
    """
    TSP solver using Simulated Annealing with 2-opt neighborhood.
    Works for small/medium/large instances.
    """

    def __init__(self, out_dir: str = os.path.join("data", "sa")) -> None:
        self.out_dir = out_dir

    def _out_path(self, instance_name: str, tag: str) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, f"{instance_name}_sa_{tag}.json")

    def solve_and_save(
        self,
        instance_path: str,
        tag: str,
        seed: int = 123,
        max_iters: int = 20000,
        t0: float = 1000.0,
        alpha: float = 0.995,
        t_min: float = 1e-6,
        force: bool = False
    ) -> Dict[str, Any]:
        inst = load_instance_json(instance_path)
        out_path = self._out_path(inst.name, tag)

        if (not force) and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            print(f"[SKIP] SA result exists: {out_path}")
            print("best_length :", payload["best_length"])
            print("runtime_sec :", payload["runtime_sec"])
            return payload

        rng = random.Random(seed)

        # initial solution: random permutation
        curr = list(range(inst.n))
        rng.shuffle(curr)
        curr_len = tour_length(inst.dist, curr)

        best = curr[:]
        best_len = curr_len

        T = float(t0)
        t_start = time.time()

        it = 0
        while it < max_iters and T > t_min:
            # 2-opt neighbor
            i, k = self._random_2opt_indices(inst.n, rng)
            cand = self._two_opt(curr, i, k)
            cand_len = tour_length(inst.dist, cand)

            delta = cand_len - curr_len
            if delta <= 0:
                curr, curr_len = cand, cand_len
            else:
                # accept worse with probability exp(-delta/T)
                p = math.exp(-delta / T)
                if rng.random() < p:
                    curr, curr_len = cand, cand_len

            if curr_len < best_len:
                best, best_len = curr[:], curr_len

            T *= alpha
            it += 1

        t_end = time.time()

        payload: Dict[str, Any] = {
            "instance_name": inst.name,
            "method": "TSP Simulated Annealing (2-opt)",
            "params": {
                "seed": seed,
                "max_iters": max_iters,
                "t0": t0,
                "alpha": alpha,
                "t_min": t_min,
            },
            "best_length": best_len,
            "best_tour": best,
            "runtime_sec": round(t_end - t_start, 6),
            "iters_done": it,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] SA saved: {out_path}")
        print("best_length :", best_len)
        print("runtime_sec :", payload["runtime_sec"])
        print("iters_done  :", it)

        return payload

    @staticmethod
    def _random_2opt_indices(n: int, rng: random.Random) -> Tuple[int, int]:
        # pick i < k, avoid trivial reversals
        i = rng.randrange(0, n - 1)
        k = rng.randrange(i + 1, n)
        # avoid reversing whole tour too often (optional)
        if i == 0 and k == n - 1 and n > 3:
            k = n - 2
        return i, k

    @staticmethod
    def _two_opt(tour: List[int], i: int, k: int) -> List[int]:
        # reverse segment tour[i:k+1]
        new_tour = tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]
        return new_tour
