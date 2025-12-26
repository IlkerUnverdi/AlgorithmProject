# -*- coding: utf-8 -*-

from __future__ import annotations

import os

from dataset_builder import DatasetBuilder
from tsp_bruteforce_solver import TSPBruteForceSolver
from tsp_sa_solver import TSPSimulatedAnnealingSolver
from result_utils import load_json
from tsp_aco_solver import TSPAntColonySolver
from experiments_runner import run_experiments

INSTANCES = {
    "1": ("small", os.path.join("data", "instances", "small_uniform_n11_seed42.json")),
    "2": ("medium", os.path.join("data", "instances", "medium_uniform_n50_seed43.json")),
    "3": ("large", os.path.join("data", "instances", "large_uniform_n100_seed44.json")),
}


def find_aco_path(instance_name: str, tag: str = "small") -> str | None:
    p = os.path.join("data", "aco", f"{instance_name}_aco_{tag}.json")
    return p if os.path.exists(p) else None

def pick_instance() -> str:
    print("\nSelect instance:")
    print("1) small  (n=11)  [exact OK]")
    print("2) medium (n=50)  [exact NO]")
    print("3) large  (n=100) [exact NO]")
    choice = input("Choice (1/2/3): ").strip()
    return choice


def menu() -> str:
    print("\n=== MENU ===")
    print("1) Build dataset (skip if exists)")
    print("2) Exact solve (small only)")
    print("3) SA solve (small/medium/large)")
    print("4) Compare SMALL (SA vs BruteForce OPT)")
    print("5) ACO solve (small/medium/large)")
    print("6) Compare SMALL (ACO vs BruteForce OPT)")
    print("7) Run experiments (medium, 10 runs)")
    print("8) Run experiments (large, 10 runs)")
    print("9) Run experiments (medium+large, 10 runs each)")

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


def find_sa_path(instance_name: str) -> str | None:
    # we saved SA as: data/sa/<instance>_sa_small.json
    p = os.path.join("data", "sa", f"{instance_name}_sa_small.json")
    return p if os.path.exists(p) else None


def compare_small_sa_vs_opt(instance_name: str) -> None:
    opt_path = find_opt_path(instance_name)
    sa_path = find_sa_path(instance_name)

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

    print("\n===== SMALL COMPARISON =====")
    print("instance    :", instance_name)
    print("opt_length  :", opt_len)
    print("sa_length   :", sa_len)
    print("ratio SA/OPT:", round(ratio, 6))
    print("gap_percent :", round(gap_pct, 4), "%")
    print("============================\n")


def main() -> None:
    builder = DatasetBuilder()
    solver = TSPBruteForceSolver()

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
                print("[NO] Exact solve is only allowed for SMALL (n=11).")
                print("     For medium/large we will use SA/ACO.")
                continue

            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            solver.solve_and_save(path, force=False)

        elif action == "3":
            c = pick_instance()
            if c not in INSTANCES:
                print("[ERR] invalid selection.")
                continue

            tag, path = INSTANCES[c]
            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            sa = TSPSimulatedAnnealingSolver()
            # tag is used in filename; can be small/medium/large
            sa.solve_and_save(
                instance_path=path,
                tag=tag,
                seed=123,
                max_iters=20000,
                t0=1000.0,
                alpha=0.995,
                t_min=1e-6,
                force=False
            )

        elif action == "4":
            # compare only small instance
            tag, small_path = INSTANCES["1"]  # small
            if not os.path.exists(small_path):
                print("[ERR] small instance missing. Build dataset first.")
                continue

            # instance name is inside the JSON file, but we can infer from filename too.
            # safest: load the instance and use its name
            from tsp_core import load_instance_json
            inst = load_instance_json(small_path)

            compare_small_sa_vs_opt(inst.name)

        elif action == "5":
            c = pick_instance()
            if c not in INSTANCES:
                print("[ERR] invalid selection.")
                continue

            tag, path = INSTANCES[c]
            if not os.path.exists(path):
                print("[ERR] instance file not found. Run dataset build first.")
                continue

            aco = TSPAntColonySolver()
            aco.solve_and_save(
                instance_path=path,
                tag=tag,
                seed=123,
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
            tag, small_path = INSTANCES["1"]  # small
            if not os.path.exists(small_path):
                print("[ERR] small instance missing. Build dataset first.")
                continue

            from tsp_core import load_instance_json
            inst = load_instance_json(small_path)

            opt_path = find_opt_path(inst.name)
            aco_path = find_aco_path(inst.name, tag="small")

            if opt_path is None:
                print("[ERR] Optimum file not found. Run brute-force on SMALL first.")
                continue
            if aco_path is None:
                print("[ERR] ACO file not found. Run ACO on SMALL first.")
                continue

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
        elif action == "7":
            run_experiments("medium", runs=10, base_seed=1000)

        elif action == "8":
            run_experiments("large", runs=10, base_seed=2000)

        elif action == "9":
            run_experiments("medium", runs=10, base_seed=1000)
            run_experiments("large", runs=10, base_seed=2000)


        elif action == "0":
            print("Bye.")
            break

        else:
            print("[ERR] invalid menu option.")


if __name__ == "__main__":
    main()
