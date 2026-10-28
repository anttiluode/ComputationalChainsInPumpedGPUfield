"""Held-out temporal-memory benchmark for the pumped wave-field chains.

Compare bonded, cut, mechanically disabled (sham), and bare-field responses
to the *same* unpredictable input bits. Fit a ridge LINEAR readout to
DISTAL wave-field probes only. Use chronological held-out test data.

This is an experimental benchmark. No improvement is assumed beforehand.

    python experiments/temporal_memory.py --symbols 320 --seeds 1
    python experiments/temporal_memory.py --symbols 320 --seeds 1 --freeze
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world


CONDITIONS = ("bonded", "cut", "sham", "bare")


def make_probes(N, d0, length):
    """15 spatial probes beyond the last chain bead, identical for all worlds."""
    right = N / 2 + (length - 1) * d0 / 2
    if right + 12 >= N - 3:
        raise ValueError("world too narrow for distal probes")
    return [(float(right + x), float(N / 2 + y))
            for x in (2, 4, 6, 9, 12) for y in (-5, 0, 5)]


def validate_sequences(sequences, N, d0):
    if not sequences or not sequences[0]:
        raise ValueError("provide one or two nonempty sequences")
    letters = set(Config().alphabet)
    for seq in sequences:
        if any(ch not in letters for ch in seq):
            raise ValueError("sequences must use only A, C, G, T")
    if len(sequences) == 2 and Counter(sequences[0]) != Counter(sequences[1]):
        raise ValueError("two sequences must have the same length and composition")
    if (len(sequences[0]) - 1) * d0 >= N / 2 - 8:
        raise ValueError("sequences too long for probe geometry")


def make_worlds(args, seed, sequence):
    cfg = Config(N=args.N, seed=seed, device=args.device,
                 p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0)
    base = World(cfg)
    start = (args.N / 2 - (len(sequence) - 1) * cfg.d0 / 2, args.N / 2)
    base.add_chain(sequence, start)
    for _ in range(args.warmup):
        base.step()

    worlds = {kind: fork_world(base) for kind in CONDITIONS if kind != "bare"}
    worlds["cut"].nb.fill_(-1)
    worlds["sham"].cfg.k_bond = 0.0
    worlds["sham"].cfg.k_bend = 0.0
    if args.freeze:
        for w in worlds.values():
            w.pinned[:] = True

    # Exactly the same wave equation and forcing, without any oscillators.
    bare = World(Config(N=args.N, seed=seed, device=args.device,
                        p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0))
    bare.t = base.t
    worlds["bare"] = bare
    return worlds


def acquire(args, seed, sequence, inputs, positions):
    """One probe sample at the end of every input-hold interval."""
    worlds = make_worlds(args, seed, sequence)
    device_world = worlds["bonded"]
    sensor = FieldProbes(device_world, positions)
    start_x = args.N / 2 - (len(sequence) - 1) * device_world.cfg.d0 / 2
    kernel = source_kernel(device_world, (start_x - 5, args.N / 2), args.width)
    traces = {name: [] for name in CONDITIONS}
    for bit in inputs:
        ext = kernel * (args.amplitude * float(bit))
        for _ in range(args.steps_per_symbol):
            for w in worlds.values():
                w.step(external_source=ext)
        for name, w in worlds.items():
            traces[name].append(sensor.field(w).detach().cpu().numpy().copy())
    return {name: np.stack(v) for name, v in traces.items()}


def standardized_ridge(train, test, y_train, alpha):
    """Fit standardized features using train data only, with an unpenalized bias."""
    center = train.mean(0)
    scale = train.std(0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    a = (train - center) / scale
    b = (test - center) / scale
    a = np.column_stack((np.ones(len(a)), a))
    b = np.column_stack((np.ones(len(b)), b))
    gram = a.T @ a + alpha * np.diag([0.0] + [1.0] * (a.shape[1] - 1))
    coeff = np.linalg.solve(gram, a.T @ y_train)
    return b @ coeff


def score(pred, truth, y_train):
    """Test R^2 relative to the training mean, not the test mean."""
    mse = float(np.mean(np.square(pred - truth)))
    baseline = float(np.mean(np.square(truth - y_train.mean())))
    r2 = 1 - mse / max(baseline, 1e-15)
    return {"test_r2": r2,
            "test_mse": mse,
            "baseline_mse": baseline,
            "sign_accuracy": float(np.mean(np.where(pred >= 0, 1, -1) == truth))}


def fit_tasks(args, traces, bits, seed):
    """Fixed chronological train/test, with a gap preventing shared labels."""
    n = len(bits)
    max_lag = max(8, args.max_lag)
    start = max(args.discard, max_lag)
    split = int(n * args.train_fraction)
    train = np.arange(start, split, dtype=int)
    test = np.arange(split + max_lag + 1, n, dtype=int)
    if len(train) < 30 or len(test) < 30:
        raise ValueError("too few train/test symbols; raise --symbols or reduce --discard")
    delays = sorted(set([0, 1, 2, 4, 8, args.max_lag]))
    delays = [d for d in delays if d <= args.max_lag]
    targets = {f"delay_{d}": bits[np.arange(n) - d] for d in delays}
    targets["parity_lag1_2"] = bits[np.arange(n) - 1] * bits[np.arange(n) - 2]
    targets["parity_lag1_3"] = bits[np.arange(n) - 1] * bits[np.arange(n) - 3]
    result = []
    for condition, matrix in traces.items():
        for task, target in targets.items():
            pred = standardized_ridge(matrix[train], matrix[test],
                                      target[train], args.ridge)
            metrics = score(pred, target[test], target[train])
            result.append({"seed": seed, "condition": condition,
                           "task": task, "n_train": int(len(train)),
                           "n_test": int(len(test)), **metrics})
    return result


def run(args):
    sequences = [s.strip().upper() for s in args.sequences.split(",") if s.strip()]
    validate_sequences(sequences, args.N, Config().d0)
    if args.seeds < 1 or args.steps_per_symbol < 1 or args.warmup < 0:
        raise ValueError("invalid seeds, warmup or steps_per_symbol")
    if args.max_lag < 3 or args.symbols < 100 or args.discard < 0:
        raise ValueError("--max-lag must be >=3, --symbols >=100, --discard >=0")
    if not 0.3 < args.train_fraction < 0.9 or args.ridge <= 0:
        raise ValueError("train_fraction must be (0.3, 0.9), ridge > 0")
    if args.width <= 0 or args.amplitude <= 0:
        raise ValueError("width and amplitude must be positive")
    probes = make_probes(args.N, Config().d0, len(sequences[0]))
    os.makedirs(args.out, exist_ok=True)
    all_results = []
    summaries = []
    sample_arrays = {}
    for seed in range(args.seed, args.seed + args.seeds):
        bits = np.random.default_rng(args.input_seed + seed).choice([-1.0, 1.0],
                                                                     size=args.symbols)
        for label, sequence in enumerate(sequences):
            series = acquire(args, seed, sequence, bits, probes)
            scores = fit_tasks(args, series, bits, seed)
            for item in scores:
                item["sequence"] = sequence
                all_results.append(item)
            sham = float(np.max(np.abs(series["sham"] - series["cut"])))
            bonded_cut = float(np.sqrt(np.mean(np.square(series["bonded"] - series["cut"]))))
            summaries.append({"seed": seed, "sequence": sequence,
                              "sham_cut_max_abs": sham,
                              "bond_cut_field_rms": bonded_cut})
            if sham > 1e-5:
                raise AssertionError("sham vs cut mismatch: investigate the control before scoring")
            if args.freeze and bonded_cut > 1e-5:
                raise AssertionError("frozen bonded and cut worlds differ unexpectedly")
            for name, matrix in series.items():
                sample_arrays[f"seed{seed}_seq{label}_{name}"] = matrix
            sample_arrays[f"seed{seed}_inputs"] = bits
            for task in ("delay_1", "delay_4", "parity_lag1_2"):
                print(f"seed {seed} seq {label} {task}: " + ", ".join(
                    f"{row['condition']} R2={row['test_r2']:+.3f}" for row in scores
                    if row["task"] == task), flush=True)
            print(f"  controls: sham-cut max={sham:.2e}; bond-cut RMS={bonded_cut:.3g}",
                  flush=True)
    np.savez_compressed(os.path.join(args.out, "samples.npz"), **sample_arrays)
    report = {"experiment": "held_out_temporal_memory",
              "task": "linear readout of delayed random input bits and non-linear bit products",
              "sequences": sequences, "probe_positions": probes,
              "parameters": vars(args),
              "controls": summaries, "scores": all_results,
              "split": "chronological train/test with max-lag gap; standardize on train only",
              "important_limits": [
                  "No growth/learning of bond structure takes place: preassembled chains.",
                  "Bare field can store input history; compare bonded with cut and bare.",
                  "No parameter tuning on held-out test; task results can be negative.",
                  "Strong near-input or instantaneous responses do not show bond computation.",
                  "Three seeds are preliminary; improve sample size and generalize input sequences."]}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"Saved {args.out}/summary.json and samples.npz", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequences", default="ACGTACGT,AGCTAGCT",
                   help="one or two comma-separated sequences, matched composition if two")
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--input-seed", type=int, default=48271)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--symbols", type=int, default=320)
    p.add_argument("--steps-per-symbol", type=int, default=12)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--discard", type=int, default=20)
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--out", default="runs/temporal_memory")
    run(p.parse_args())


if __name__ == "__main__":
    main()
