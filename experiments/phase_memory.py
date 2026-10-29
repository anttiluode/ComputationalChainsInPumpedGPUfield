"""Write - wait - read: frequency-addressed phase memory in locked complement pairs.

Physics (world.py): a bead's phase is pushed by  c(t) * psi * sin(theta),  c = cos(2*Omega*t).
That has a DC part only if the field holds the bead's COMPLEMENT frequency 2*Omega - w.
So a carrier at 0.03 (=complement of T) and 0.11 (=complement of A) address the A/T pair,
0.05 / 0.09 address the C/G pair. For a complement pair (w_i + w_j = 2 Omega) the pump
couples cos(2 Omega t) cos(theta_j) sin(theta_i), whose DC part depends on
theta_i + theta_j - 2 Omega t: the SUM phase is locked to the pump (restoring, forgets) and
the DIFFERENCE theta_i - theta_j is neutral (remembers). Measured: a write first kicks the
sum; the pump lock pulls the sum back and leaves a lasting shift in the difference.
This is the memory the overnight fading-memory benchmark could not see: its hold-encoded
bits had no complement component at all.

Protocol, one trial: warm (shared) -> write bits as BPSK on carriers for --write steps ->
silence -> passive read windows of --read steps starting --delays after the write ends.
Same protocol on the bare field (no beads). Each trial has its own phase-noise seed.
Readout: per probe, quadrature demodulation at the 4 letter frequencies and their pump
sidebands (w + 2 Omega, which propagate above the gap) -> ridge across trials, train/test
8-fold cross-validated over trials, probe noise sigma added before demodulation.

Modes
  multiplex : b1 on 0.03 (A/T register), b2 on 0.09 (C/G register), written together.
              Targets b1, b2.
  addition  : b1 then b2, both on 0.03 into the SAME register. If each write turns the
              difference phase by ~pi/2, it is ~(b1+b2)pi/2 and its cosine is -b1*b2:
              XOR produced by phase addition + sinusoidal emission. Targets b1, b2, b1*b2.

    python experiments/phase_memory.py --mode multiplex --trials 64 --phase-noise 0.003
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.temporal_memory import standardized_ridge

FREQS = (0.03, 0.05, 0.09, 0.11)


def circ(a, b):
    return torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi


def build(args, beads):
    cfg = Config(N=args.N, seed=args.seed, device="cpu", p_lig=0.0, kT=0.0,
                 phase_noise=0.0, chain_boost=args.chain_boost)
    w = World(cfg)
    L = len(args.sequence)
    xs = [args.N / 2 - (L - 1) * args.spacing / 2 + args.spacing * i for i in range(L)]
    if beads:
        w.add_monomers([w.letters.index(c) for c in args.sequence],
                       pos=[(x, args.N / 2) for x in xs])
        w.pinned[:] = True
    for _ in range(args.warmup):
        w.step()
    return w, xs


def probes_for(args, xs):
    y = args.N / 2
    near = [(x, y - 2.0) for x in xs] + [(x, y - 3.5) for x in xs]
    far = [(xs[0] - 18, y), (xs[-1] + 18, y), (args.N / 2, y - 20), (args.N / 2, y + 20)]
    return near, far


def write_wave(args, bits, k):
    """External source amplitude at write step k (clocked: same start every trial)."""
    if args.mode == "multiplex":
        return args.amplitude * (bits[0] * math.cos(0.03 * k) + bits[1] * math.cos(0.09 * k))
    half = args.write
    b = bits[0] if k < half else bits[1]
    return args.amplitude * b * math.cos(0.03 * k)


def run_trial(args, base, xs, bits, trial):
    w = fork_world(base)
    w.cfg.phase_noise = args.phase_noise
    w.gen.manual_seed(10_000 + trial)
    theta0 = base.theta.clone()
    near, far = probes_for(args, xs)
    pn, pf = FieldProbes(w, near), FieldProbes(w, far)
    kernel = source_kernel(w, (args.N / 2, args.N / 2 + args.source_dy), 1.0)
    n_write = args.write * (2 if args.mode == "addition" else 1)
    for k in range(n_write):
        w.step(external_source=kernel * float(write_wave(args, bits, k)))
    windows, hidden = {}, {}
    starts = sorted(args.delays)
    t = 0
    for D in starts:
        while t < D:
            w.step(); t += 1
        hidden[D] = circ(w.theta, theta0).numpy().copy() if w.P else np.zeros(0)
        rec_n, rec_f = [], []
        for _ in range(args.read):
            w.step(); t += 1
            rec_n.append(pn.field(w).numpy().copy())
            rec_f.append(pf.field(w).numpy().copy())
        windows[D] = (np.array(rec_n), np.array(rec_f))
    return windows, hidden


def demod(rec, sigma, rng, t0):
    x = rec + (rng.normal(0, sigma, rec.shape) if sigma > 0 else 0)
    k = t0 + np.arange(rec.shape[0])
    out = []
    for om in FREQS + tuple(f + 0.14 for f in FREQS):
        out.append((x * np.cos(om * k)[:, None]).mean(0))
        out.append((x * np.sin(om * k)[:, None]).mean(0))
    return np.concatenate(out)


def decode(X, y, train, test, ridge):
    p = standardized_ridge(X[train], X[test], y[train], ridge)
    return float(np.mean(np.sign(p) == y[test]))


def cv_decode(X, y, ridge, folds=8):
    """k-fold cross-validated sign accuracy over ALL trials (lower variance than one split)."""
    n = len(y)
    idx = np.arange(n)
    hits = 0
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        p = standardized_ridge(X[train], X[test], y[train], ridge)
        hits += int(np.sum(np.sign(p) == y[test]))
    return hits / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("multiplex", "addition"), default="multiplex")
    p.add_argument("--sequence", default="ACGT")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--chain-boost", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--write", type=int, default=600)
    p.add_argument("--amplitude", type=float, default=0.2)
    p.add_argument("--source-dy", type=float, default=2.5)
    p.add_argument("--delays", type=int, nargs="+", default=[100, 600, 2000, 6000])
    p.add_argument("--read", type=int, default=400)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--phase-noise", type=float, default=0.003)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1e-3])
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--out", default="runs/phase_memory")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed + 99)
    # balanced bit patterns
    pats = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]] * (a.trials // 4 + 1))[: a.trials]
    pats = pats[rng.permutation(len(pats))].astype(float)
    worlds = {}
    for name, beads in (("chain", True), ("bare", False)):
        worlds[name] = build(a, beads)
    feats = {(n, D, s, side): [] for n in worlds for D in a.delays for s in a.sigmas for side in ("near", "far")}
    hidden_all = []
    for i, bits in enumerate(pats):
        for name, (base, xs) in worlds.items():
            win, hid = run_trial(a, base, xs, bits, i)
            if name == "chain":
                hidden_all.append({D: hid[D].tolist() for D in a.delays})
            for D, (rn, rf) in win.items():
                for s in a.sigmas:
                    r = np.random.default_rng(1000 * i + int(1e6 * s))
                    feats[(name, D, s, "near")].append(demod(rn, s, r, D))
                    feats[(name, D, s, "far")].append(demod(rf, s, r, D))
        if (i + 1) % 16 == 0:
            print(f"  {i + 1}/{a.trials} trials", flush=True)
    n = len(pats)
    order = np.arange(n)
    train, test = order[: int(0.7 * n)], order[int(0.7 * n):]
    targets = {"b1": pats[:, 0], "b2": pats[:, 1], "b1*b2": pats[:, 0] * pats[:, 1]}
    rows = []
    np.savez_compressed(os.path.join(a.out, "features.npz"), pats=pats,
                        **{f"{n}_{D}_{s:g}_{side}": np.array(F) for (n, D, s, side), F in feats.items()})
    for (name, D, s, side), F in feats.items():
        X = np.array(F)
        acc = {k: cv_decode(X, y, a.ridge) for k, y in targets.items()}
        rows.append({"world": name, "delay": D, "sigma": s, "probes": side, **acc})
    # hidden-state ground truth: shift of each pair's DIFFERENCE phase (the neutral mode)
    H = {D: np.array([h[D] for h in hidden_all]) for D in a.delays}
    truth = {}
    seq = a.sequence
    for D, h in H.items():
        reg = {}
        for name_r, (li, lj) in (("AT", ("A", "T")), ("CG", ("C", "G"))):
            if li in seq and lj in seq:
                i, j = seq.index(li), seq.index(lj)
                diff = np.angle(np.exp(1j * (h[:, i] - h[:, j])))
                reg[name_r] = {"mean_abs_shift": float(np.abs(diff).mean()),
                               "corr_with_b1": float(np.corrcoef(np.sin(diff), pats[:, 0])[0, 1]),
                               "corr_with_b2": float(np.corrcoef(np.sin(diff), pats[:, 1])[0, 1]),
                               "corr_cos_with_b1b2": float(np.corrcoef(np.cos(diff), pats[:, 0] * pats[:, 1])[0, 1])}
        truth[D] = reg
    json.dump({"parameters": vars(a), "rows": rows, "hidden_truth": truth},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["probes"], r["sigma"], r["world"], r["delay"])):
        print(f"{r['probes']:4s} sigma={r['sigma']:<6g} {r['world']:5s} delay={r['delay']:5d}  "
              f"b1={r['b1']:.2f} b2={r['b2']:.2f} b1*b2={r['b1*b2']:.2f}")
    for D, reg in truth.items():
        print(f"hidden state at delay {D}: " + "; ".join(
            f"{k}: |shift|={v['mean_abs_shift']:.2f} corr(sin,b1)={v['corr_with_b1']:+.2f} "
            f"corr(sin,b2)={v['corr_with_b2']:+.2f} corr(cos,b1b2)={v['corr_cos_with_b1b2']:+.2f}"
            for k, v in reg.items()))


if __name__ == "__main__":
    main()
