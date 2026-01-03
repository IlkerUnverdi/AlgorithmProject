# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random

from dataset_builder import DatasetBuilder
from tsp_bruteforce_solver import TSPBruteForceSolver
from tsp_sa_solver import TSPSimulatedAnnealingSolver
from tsp_aco_solver import TSPAntColonySolver
from tsp_core import load_instance_json
from result_utils import load_json
from experiments_runner import run_experiments


INSTANCES = {
    "1": ("small", os.path.join("data", "instances", "small_uniform_n10_seed42.json")),
    "2": ("medium", os.path.join("data", "instances", "medium_uniform_n50_seed43.json")),
    "3": ("large", os.path.join("data", "instances", "large_uniform_n100_seed44.json")),
    "4": ("medium_cluster", os.path.join("data", "instances", "medium_cluster_cluster_n50_seed45.json")),
}


def pick_instance() -> str:
    print("\nSelect instance:")
    print("1) small         (n=10)  [exact OK]")
    print("2) medium        (n=50)  [exact NO]")
    print("3) large         (n=100) [exact NO]")
    print("4) medium_cluster(n=50)  [exact NO]")
    return input("Choice (1/2/3/4): ").strip()


def ask_seed(default: int = 123) -> int:
    s = input(f"Seed (enter for {default}, or type 'r' for random): ").strip().lower()
    if s == "":
        return default
    if s == "r":
        return random.randint(1, 10**9)
    return int(s)


def menu() -> str:
    print("\n=== MENU ===")
    print("1) Build dataset (skip if exists)")
    print("2) Exact solve (small only)")
    print("3) SA solve (single run)")
    print("4) Compare SMALL (SA vs BruteForce OPT)")
    print("5) ACO solve (single run)")
    print("6) Compare SMALL (ACO vs BruteForce OPT)")
    print("7) Run experiments (small, 30 runs)")
    print("8) Run experiments (medium, 10 runs)")
    print("9) Run experiments (large, 10 runs)")
    print("10) Run experiments (medium_cluster, 10 runs)")
    print("0) Exit")
    return input("Select: ").strip()


def find_opt_path(instance_name: str) -> str | None:
    candidates = [
        os.path.join("data", "opt", f"{instance_name}_bruteforce_opt.json"),
        os.path.join("data", "opt", f"{instance_name}_opt.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def find_sa_path(instance_name: str, tag: str) -> str | None:
    p = os.path.join("data", "sa", f"{instance_name}_sa_{tag}.json")
    return p if os.path.exists(p) else None


def find_aco_path(instance_name: str, tag: str) -> str | None:
    p = os.path.join("data", "aco", f"{instance_name}_aco_{tag}.json")
    return p if os.path.exists(p) else None


def compare_small_sa_vs_opt() -> None:
    tag, small_path = INSTANCES["1"]
    if not os.path.exists(small_path):
        print("[ERR] small instance missing. Build dataset first.")
        return

    inst = load_instance_json(small_path)

    opt_path = find_opt_path(inst.name)
    sa_path = find_sa_path(inst.name, tag="small")

    if opt_path is None:
        print("[ERR] Optimum file not found. Run brute-force on SMALL first.")
        return
    if sa_path is None:
        print("[ERR] SA file not found. Run SA on SMALL first.")
        return

    opt = load_json(opt_path)
    sa = load_json(sa_path)

    opt_len = float(opt["best_length"])
    sa_len = float(sa["best_length"])

    ratio = sa_len / opt_len if opt_len > 0 else float("inf")
    gap_pct = (ratio - 1.0) * 100.0

    print("\n===== SMALL COMPARISON (SA vs OPT) =====")
    print("instance    :", inst.name)
    print("opt_length  :", opt_len)
    print("sa_length   :", sa_len)
    print("ratio SA/OPT:", round(ratio, 6))
    print("gap_percent :", round(gap_pct, 4), "%")
    print("========================================\n")


def compare_small_aco_vs_opt() -> None:
    tag, small_path = INSTANCES["1"]
    if not os.path.exists(small_path):
        print("[ERR] small instance missing. Build dataset first.")
        return

    inst = load_instance_json(small_path)

    opt_path = find_opt_path(inst.name)
    aco_path = find_aco_path(inst.name, tag="small")

    if opt_path is None:
        print("[ERR] Optimum file not found. Run brute-force on SMALL first.")
        return
    if aco_path is None:
        print("[ERR] ACO file not found. Run ACO on SMALL first.")
        return

    opt = load_json(opt_path)
    aco = load_json(aco_path)

    opt_len = float(opt["best_length"])
    aco_len = float(aco["best_length"])

    ratio = aco_len / opt_len if opt_len > 0 else float("inf")
    gap_pct = (ratio - 1.0) * 100.0

    print("\n===== SMALL COMPARISON (ACO vs OPT) =====")
    print("instance     :", inst.name)
    print("opt_length   :", opt_len)
    print("aco_length   :", aco_len)
    print("ratio ACO/OPT:", round(ratio, 6))
    print("gap_percent  :", round(gap_pct, 4), "%")
    print("========================================\n")


def main() -> None:
    builder = DatasetBuilder()
    brute = TSPBruteForceSolver()

    while True:
        action = menu()

        if action == "1":
            builder.build_default_dataset(force=False)

        elif action == "2":
            c = pick_instance()
            if c not in INSTANCES:
                print("[ERR] invalid selection.")
                continue

            tag, path = INSTANCES[c]
            if tag != "small":
                print("[NO] Exact solve is only allowed for SMALL (n=10).")
                continue

            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            brute.solve_and_save(path, force=False)

        elif action == "3":
            c = pick_instance()
            if c not in INSTANCES:
                print("[ERR] invalid selection.")
                continue

            tag, path = INSTANCES[c]
            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            seed = ask_seed(default=123)

            sa = TSPSimulatedAnnealingSolver()
            sa.solve_and_save(
                instance_path=path,
                tag=tag,
                seed=seed,
                max_iters=20000,
                t0=1000.0,
                alpha=0.995,
                t_min=1e-6,
                force=False
            )

        elif action == "4":
            compare_small_sa_vs_opt()

        elif action == "5":
            c = pick_instance()
            if c not in INSTANCES:
                print("[ERR] invalid selection.")
                continue

            tag, path = INSTANCES[c]
            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            seed = ask_seed(default=123)

            aco = TSPAntColonySolver()
            aco.solve_and_save(
                instance_path=path,
                tag=tag,
                seed=seed,
                n_ants=30,
                n_iters=200,
                alpha=1.0,
                beta=5.0,
                rho=0.5,
                q=1.0,
                tau0=1.0,
                force=False
            )

        elif action == "6":
            compare_small_aco_vs_opt()

        elif action == "7":
            run_experiments("small", runs=30, base_seed=5000)

        elif action == "8":
            run_experiments("medium", runs=10, base_seed=1000)

        elif action == "9":
            run_experiments("large", runs=10, base_seed=2000)

        elif action == "10":
            run_experiments("medium_cluster", runs=10, base_seed=1500)

        elif action == "0":
            print("Bye.")
            break

        else:
            print("[ERR] invalid menu option.")


if __name__ == "__main__":
    main()
