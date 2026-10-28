"""Corrected temporal-memory benchmark: memory FUNCTIONS over lag, not fixed delays.

Why this replaces the fixed delays 0..8 in temporal_memory.py
-------------------------------------------------------------
The distal probes sit 33-43 cells from the input source. The fastest wave in the
medium moves at sqrt(c2) ~ 0.55 cells/step, so the input cannot reach them in under
~60 steps (5 symbols of 12 steps). Delays 0..4 are outside the light cone and
must score ~0 for EVERY condition. That is what the first run showed. Here we
scan lags 0..L and report the whole curve.

Measured per condition (bonded / cut / sham / bare, all from one warmed state):
  linear memory   R2_lin(d)  for target u[t-d]
  nonlinear       R2_nl(d)   for target u[t-d] * u[t-d-1]   (XOR of two bits)
  capacities      MC = sum_d max(R2, 0)

The bare field is a LINEAR system driven by u, so its nonlinear capacity is
exactly zero up to readout noise -- a built-in honesty check of the readout.

Two readouts:
  raw       probes as a physical readout would see them (includes the chains'
            own input-independent oscillation)
  response  driven minus an undriven twin of the same world (oracle subtraction:
            how much input information is in the field at all)

    python experiments/memory_capacity.py --symbols 1200 --seeds 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.temporal_memory import make_probes, standardized_ridge, score

CONDITIONS = ("bonded", "cut", "sham", "bare")


def make_worlds(args, seed, sequence):
    cfg = Config(N=args.N, seed=seed, device=args.device, p_lig=0.0, kT=0.0,
                 phase_noise=0.0, chain_boost=0.0)
    base = World(cfg)
    if sequence:
        start = (args.N / 2 - (len(sequence) - 1) * cfg.d0 / 2, args.N / 2)
        base.add_chain(sequence, start)
    for _ in range(args.warmup):
        base.step()
    worlds = {}
    for kind in CONDITIONS:
        if kind == "bare":
            w = World(Config(N=args.N, seed=seed, device=args.device, p_lig=0.0,
                             kT=0.0, phase_noise=0.0, chain_boost=0.0))
            w.t = base.t
        else:
            w = fork_world(base)
            if kind == "cut":
                w.nb.fill_(-1)
            elif kind == "sham":
                w.cfg.k_bond = 0.0
                w.cfg.k_bend = 0.0
            if args.freeze:
                w.pinned[:] = True
        worlds[kind] = w
    return worlds


def drive_value(bit, k, args):
    """Input waveform inside one symbol. 'hold': constant bit. 'carrier': the bit
    sets the sign of a sinusoid at args.carrier rad/step (phase continuous)."""
    if args.encoding == "hold":
        return args.amplitude * bit
    return args.amplitude * bit * np.cos(args.carrier * k)


def acquire(args, seed, sequence, bits, probes):
    driven = make_worlds(args, seed, sequence)
    undriven = make_worlds(args, seed, sequence)
    ref = driven["bonded"]
    sensor = FieldProbes(ref, probes)
    x0 = args.N / 2 - (len(sequence) - 1) * ref.cfg.d0 / 2
    kernel = source_kernel(ref, (x0 - args.source_offset, args.N / 2), args.width)
    S = args.steps_per_symbol
    sub = [S - 1 - i * (S // args.samples_per_symbol) for i in range(args.samples_per_symbol)]
    out = {f"{c}_{t}": [] for c in CONDITIONS for t in ("drv", "und")}
    pos = {c: [] for c in ("bonded", "cut")}
    k = 0
    for bit in bits:
        feats = {key: [] for key in out}
        for s in range(S):
            ext = kernel * float(drive_value(bit, k, args))
            k += 1
            for c in CONDITIONS:
                driven[c].step(external_source=ext)
                undriven[c].step()
            if s in sub:
                for c in CONDITIONS:
                    feats[f"{c}_drv"].append(sensor.field(driven[c]).cpu().numpy().copy())
                    feats[f"{c}_und"].append(sensor.field(undriven[c]).cpu().numpy().copy())
        for key in out:
            out[key].append(np.concatenate(feats[key]))
        for c in pos:
            pos[c].append(driven[c].pos.cpu().numpy().copy())
    arr = {key: np.stack(v) for key, v in out.items()}
    arr.update({f"{c}_pos": np.stack(v) for c, v in pos.items()})
    return arr


def memory_curves(X, bits, args):
    n = len(bits)
    L = args.max_lag
    split = int(n * args.train_fraction)
    train = np.arange(max(args.discard, L + 1), split)
    test = np.arange(split + L + 1, n)
    lin, nl = [], []
    for d in range(L + 1):
        idx = np.arange(n)
        y = bits[idx - d]
        p = standardized_ridge(X[train], X[test], y[train], args.ridge)
        lin.append(score(p, y[test], y[train])["test_r2"])
        y2 = bits[idx - d] * bits[idx - d - 1]
        p = standardized_ridge(X[train], X[test], y2[train], args.ridge)
        nl.append(score(p, y2[test], y2[train])["test_r2"])
    lin, nl = np.array(lin), np.array(nl)
    thr = null_threshold(X, train, test, args)
    return {"lin_r2": lin.tolist(), "nl_r2": nl.tolist(),
            "MC_lin": float(np.clip(lin, 0, None).sum()),
            "MC_nl": float(np.clip(nl, 0, None).sum()),
            # significance-gated capacities: a lag counts only if its R2 beats the
            # 99th percentile of R2 for targets independent of the input
            "null_r2_99": thr,
            "MC_lin_sig": float(lin[lin > thr].sum()),
            "MC_nl_sig": float(nl[nl > thr].sum()),
            "n_train": int(len(train)), "n_test": int(len(test))}


def null_threshold(X, train, test, args, draws=200, q=99):
    """R2 quantile for random +/-1 targets independent of the input. (A product of
    two bits is itself a fair +/-1 sequence, so one null serves both tasks.)
    Summing max(R2, 0) over many lags turns this noise into a spurious positive
    capacity; gating on the quantile removes it."""
    rng = np.random.default_rng(12345)
    n = X.shape[0]
    vals = []
    for _ in range(draws):
        y = rng.choice([-1.0, 1.0], size=n)
        p =standardized_ridge(X[train], X[test], y[train], args.ridge)
        vals.append(score(p, y[test], y[train])["test_r2"])
    return max(float(np.percentile(vals, q)), 0.0)


def shift_null_pvalue(X, bits, args, task="nl", draws=60, seed=0):
    """The honest test for a capacity: keep the target sequence exactly, break only its
    alignment with the input by circularly shifting the bits by a large random offset,
    and ask how often the shifted capacity reaches the real one. The 21 lagged targets
    are shifted copies of one sequence, so their R2 values are correlated; an i.i.d.
    null underestimates the tail. Returns (observed, null mean, null 95th pct, p)."""
    obs = memory_curves(X, bits, args)["MC_nl" if task == "nl" else "MC_lin"]
    rng = np.random.default_rng(seed)
    null = []
    n = len(bits)
    for _ in range(draws):
        sb = np.roll(bits, int(rng.integers(3 * args.max_lag, n - 3 * args.max_lag)))
        null.append(memory_curves(X, sb, args)["MC_nl" if task == "nl" else "MC_lin"])
    null = np.array(null)
    return obs, float(null.mean()), float(np.percentile(null, 95)), float((null >= obs).mean())


def run(args):
    seqs = [s.strip().upper() for s in args.sequences.split(",") if s.strip()]
    probes = make_probes(args.N, Config().d0, len(seqs[0]))
    os.makedirs(args.out, exist_ok=True)
    rows, raw = [], {}
    for seed in range(args.seed, args.seed + args.seeds):
        bits = np.random.default_rng(args.input_seed + seed).choice([-1.0, 1.0], size=args.symbols)
        for sequence in seqs:
            data = acquire(args, seed, sequence, bits, probes)
            for c in CONDITIONS:
                for mode in ("raw", "response"):
                    X = data[f"{c}_drv"] if mode == "raw" else data[f"{c}_drv"] - data[f"{c}_und"]
                    m = memory_curves(X, bits, args)
                    rows.append({"seed": seed, "sequence": sequence, "condition": c,
                                 "readout": mode, **m})
                    print(f"seed {seed} {sequence} {c:6s} {mode:8s} "
                          f"MC_lin={m['MC_lin']:.3f} MC_nl={m['MC_nl']:.3f}", flush=True)
            for key, v in data.items():
                raw[f"s{seed}_{sequence}_{key}"] = v
            raw[f"s{seed}_bits"] = bits
            drift = {c: float(np.linalg.norm(data[f"{c}_pos"][-1] - data[f"{c}_pos"][0], axis=1).mean())
                     for c in ("bonded", "cut")}
            print(f"  mean bead drift over run: {drift}", flush=True)
            rows.append({"seed": seed, "sequence": sequence, "drift": drift})
    np.savez_compressed(os.path.join(args.out, "features.npz"), **raw)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"parameters": vars(args), "probes": probes, "rows": rows}, f, indent=1)
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", default="ACGTACGT")
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--input-seed", type=int, default=48271)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--symbols", type=int, default=1200)
    p.add_argument("--steps-per-symbol", type=int, default=12)
    p.add_argument("--samples-per-symbol", type=int, default=3)
    p.add_argument("--encoding", choices=("hold", "carrier"), default="hold")
    p.add_argument("--carrier", type=float, default=0.11)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--source-offset", type=float, default=5.0)
    p.add_argument("--discard", type=int, default=40)
    p.add_argument("--max-lag", type=int, default=30)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--out", default="runs/memory_capacity")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
