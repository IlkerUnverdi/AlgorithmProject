# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import List, Dict, Any, Tuple

from tsp_core import load_instance_json, tour_length


class TSPAntColonySolver:
    """
    Ant Colony Optimization for symmetric Euclidean TSP.
    """

    def __init__(self, out_dir: str = os.path.join("data", "aco")) -> None:
        self.out_dir = out_dir

    def _out_path(self, instance_name: str, tag: str) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, f"{instance_name}_aco_{tag}.json")

    def solve_and_save(
        self,
        instance_path: str,
        tag: str,
        seed: int = 123,
        n_ants: int = 30,
        n_iters: int = 200,
        alpha: float = 1.0,
        beta: float = 5.0,
        rho: float = 0.5,
        q: float = 1.0,
        tau0: float = 1.0,
        force: bool = False
    ) -> Dict[str, Any]:
        inst = load_instance_json(instance_path)
        out_path = self._out_path(inst.name, tag)

        if (not force) and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            print(f"[SKIP] ACO result exists: {out_path}")
            print("best_length :", payload["best_length"])
            print("runtime_sec :", payload["runtime_sec"])
            return payload

        rng = random.Random(seed)
        n = inst.n
        dist = inst.dist

        # Precompute eta = 1/d for i!=j
        eta = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = dist[i][j]
                    eta[i][j] = (1.0 / d) if d > 0 else 0.0

        # Pheromone matrix
        tau = [[tau0] * n for _ in range(n)]
        for i in range(n):
            tau[i][i] = 0.0

        best_tour: List[int] = []
        best_len = float("inf")

        t_start = time.time()

        for it in range(n_iters):
            iter_best_tour: List[int] = []
            iter_best_len = float("inf")

            for _ in range(n_ants):
                tour = self._construct_tour(rng, tau, eta, alpha, beta)
                L = tour_length(dist, tour)

                if L < iter_best_len:
                    iter_best_len = L
                    iter_best_tour = tour

            # Evaporation
            for i in range(n):
                row = tau[i]
                for j in range(n):
                    row[j] *= (1.0 - rho)

            # Reinforcement: deposit only from iteration-best (simple and stable)
            if iter_best_tour:
                deposit = q / iter_best_len if iter_best_len > 0 else 0.0
                self._deposit_pheromone(tau, iter_best_tour, deposit)

            # Track global best
            if iter_best_len < best_len:
                best_len = iter_best_len
                best_tour = iter_best_tour[:]

        t_end = time.time()

        payload: Dict[str, Any] = {
            "instance_name": inst.name,
            "method": "TSP Ant Colony Optimization",
            "params": {
                "seed": seed,
                "n_ants": n_ants,
                "n_iters": n_iters,
                "alpha": alpha,
                "beta": beta,
                "rho": rho,
                "q": q,
                "tau0": tau0,
            },
            "best_length": best_len,
            "best_tour": best_tour,
            "runtime_sec": round(t_end - t_start, 6),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"[OK] ACO saved: {out_path}")
        print("best_length :", best_len)
        print("runtime_sec :", payload["runtime_sec"])

        return payload

    @staticmethod
    def _construct_tour(
        rng: random.Random,
        tau: List[List[float]],
        eta: List[List[float]],
        alpha: float,
        beta: float
    ) -> List[int]:
        n = len(tau)
        start = rng.randrange(n)
        visited = [False] * n
        visited[start] = True
        tour = [start]
        current = start

        for _ in range(n - 1):
            next_city = TSPAntColonySolver._select_next_city(
                rng, current, visited, tau, eta, alpha, beta
            )
            visited[next_city] = True
            tour.append(next_city)
            current = next_city

        return tour

    @staticmethod
    def _select_next_city(
        rng: random.Random,
        current: int,
        visited: List[bool],
        tau: List[List[float]],
        eta: List[List[float]],
        alpha: float,
        beta: float
    ) -> int:
        n = len(visited)

        weights = []
        cities = []
        for j in range(n):
            if not visited[j]:
                t = tau[current][j]
                h = eta[current][j]
                w = (t ** alpha) * (h ** beta) if t > 0 and h > 0 else 0.0
                cities.append(j)
                weights.append(w)

        total = sum(weights)
        if total <= 0.0:
            # fallback: random among unvisited
            return cities[rng.randrange(len(cities))]

        r = rng.random() * total
        cum = 0.0
        for city, w in zip(cities, weights):
            cum += w
            if cum >= r:
                return city

        return cities[-1]

    @staticmethod
    def _deposit_pheromone(tau: List[List[float]], tour: List[int], amount: float) -> None:
        n = len(tour)
        for i in range(n - 1):
            a = tour[i]
            b = tour[i + 1]
            tau[a][b] += amount
            tau[b][a] += amount
        # close the tour
        a = tour[-1]
        b = tour[0]
        tau[a][b] += amount
        tau[b][a] += amount
