# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
import statistics
import time
from typing import Dict, Any, List, Tuple

from tsp_core import load_instance_json
from tsp_sa_solver import TSPSimulatedAnnealingSolver
from tsp_aco_solver import TSPAntColonySolver


INSTANCES = {
    "small": os.path.join("data", "instances", "small_uniform_n10_seed42.json"),
    "medium": os.path.join("data", "instances", "medium_uniform_n50_seed43.json"),
    "medium_cluster": os.path.join("data", "instances", "medium_cluster_cluster_n50_seed45.json"),
    "large": os.path.join("data", "instances", "large_uniform_n100_seed44.json"),
}

OUT_DIR = os.path.join("data", "experiments")


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def find_opt_path(instance_name: str) -> str | None:
    candidates = [
        os.path.join("data", "opt", f"{instance_name}_bruteforce_opt.json"),
        os.path.join("data", "opt", f"{instance_name}_opt.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_json(path: str) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sa(instance_path: str, tag: str, seed: int) -> Dict[str, Any]:
    sa = TSPSimulatedAnnealingSolver()
    return sa.solve_and_save(
        instance_path=instance_path,
        tag=tag,
        seed=seed,
        max_iters=40000,
        t0=1500.0,
        alpha=0.995,
        t_min=1e-6,
        force=True,
    )


def run_aco(instance_path: str, tag: str, seed: int) -> Dict[str, Any]:
    aco = TSPAntColonySolver()
    return aco.solve_and_save(
        instance_path=instance_path,
        tag=tag,
        seed=seed,
        n_ants=40,
        n_iters=300,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        q=1.0,
        tau0=1.0,
        force=True,
    )


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    ensure_out_dir()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: List[float]) -> Tuple[float, float, float]:
    vmin = min(values)
    vmean = statistics.mean(values)
    vstdev = statistics.stdev(values) if len(values) >= 2 else 0.0
    return vmin, vmean, vstdev


def run_experiments(instance_key: str, runs: int = 10, base_seed: int = 1000) -> None:
    if instance_key not in INSTANCES:
        print("[ERR] unknown instance_key")
        return

    instance_path = INSTANCES[instance_key]
    if not os.path.exists(instance_path):
        print("[ERR] instance file not found. Build dataset first.")
        return

    inst = load_instance_json(instance_path)

    opt_len: float | None = None
    if instance_key == "small":
        opt_path = find_opt_path(inst.name)
        if opt_path is None:
            print("[ERR] small OPT file not found. Run brute force first.")
            print("      Expected under data/opt/ with *_bruteforce_opt.json")
            return
        opt = load_json(opt_path)
        opt_len = float(opt["best_length"])

    rows: List[Dict[str, Any]] = []
    sa_lengths: List[float] = []
    sa_times: List[float] = []
    aco_lengths: List[float] = []
    aco_times: List[float] = []

    sa_ratios: List[float] = []
    aco_ratios: List[float] = []

    print(f"\n[RUN] experiments on {instance_key}: {inst.name} (n={inst.n}), runs={runs}")

    t_global0 = time.time()

    for r in range(runs):
        seed = base_seed + r

        sa_tag = f"{instance_key}_run{r+1}"
        aco_tag = f"{instance_key}_run{r+1}"

        sa_res = run_sa(instance_path, tag=sa_tag, seed=seed)
        aco_res = run_aco(instance_path, tag=aco_tag, seed=seed)

        sa_len = float(sa_res["best_length"])
        sa_rt = float(sa_res["runtime_sec"])
        aco_len = float(aco_res["best_length"])
        aco_rt = float(aco_res["runtime_sec"])

        sa_lengths.append(sa_len)
        sa_times.append(sa_rt)
        aco_lengths.append(aco_len)
        aco_times.append(aco_rt)

        row: Dict[str, Any] = {
            "instance": instance_key,
            "n": inst.n,
            "run": r + 1,
            "seed": seed,
            "sa_length": sa_len,
            "sa_runtime_sec": sa_rt,
            "aco_length": aco_len,
            "aco_runtime_sec": aco_rt,
        }

        if opt_len is not None and opt_len > 0:
            sa_ratio = sa_len / opt_len
            aco_ratio = aco_len / opt_len
            sa_ratios.append(sa_ratio)
            aco_ratios.append(aco_ratio)
            row["opt_length"] = opt_len
            row["sa_ratio"] = sa_ratio
            row["aco_ratio"] = aco_ratio

        rows.append(row)

        if opt_len is not None:
            print(
                f"run {r+1:02d} seed={seed} | SA={sa_len:.3f} ({sa_rt:.3f}s) ratio={sa_len/opt_len:.4f} | "
                f"ACO={aco_len:.3f} ({aco_rt:.3f}s) ratio={aco_len/opt_len:.4f}"
            )
        else:
            print(f"run {r+1:02d} seed={seed} | SA={sa_len:.3f} ({sa_rt:.3f}s) | ACO={aco_len:.3f} ({aco_rt:.3f}s)")

    t_global1 = time.time()

    csv_path = os.path.join(OUT_DIR, f"exp_{instance_key}_runs{runs}.csv")
    write_csv(rows, csv_path)

    sa_min, sa_mean, sa_std = summarize(sa_lengths)
    aco_min, aco_mean, aco_std = summarize(aco_lengths)

    sa_t_mean = statistics.mean(sa_times)
    aco_t_mean = statistics.mean(aco_times)

    print("\n===== SUMMARY =====")
    print(f"Instance: {instance_key} (n={inst.n})")
    if opt_len is not None:
        print(f"OPT length           : {opt_len:.3f}")
    print(f"SA  length  min/mean/std : {sa_min:.3f} / {sa_mean:.3f} / {sa_std:.3f}")
    print(f"ACO length  min/mean/std : {aco_min:.3f} / {aco_mean:.3f} / {aco_std:.3f}")
    print(f"SA  runtime mean (sec)   : {sa_t_mean:.3f}")
    print(f"ACO runtime mean (sec)   : {aco_t_mean:.3f}")

    if opt_len is not None and sa_ratios and aco_ratios:
        sa_rmin, sa_rmean, sa_rstd = summarize(sa_ratios)
        aco_rmin, aco_rmean, aco_rstd = summarize(aco_ratios)
        print(f"SA  ratio   min/mean/std : {sa_rmin:.4f} / {sa_rmean:.4f} / {sa_rstd:.4f}")
        print(f"ACO ratio   min/mean/std : {aco_rmin:.4f} / {aco_rmean:.4f} / {aco_rstd:.4f}")

    print(f"CSV saved to             : {csv_path}")
    print(f"Total experiment time(s) : {round(t_global1 - t_global0, 3)}")
    print("===================\n")


if __name__ == "__main__":
    ensure_out_dir()
    run_experiments("medium", runs=10, base_seed=1000)
