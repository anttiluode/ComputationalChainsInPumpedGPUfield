"""Matched-present test (protocol of Muruga, Ginot, Loos & Bechinger, Nat. Phys. 2026).

The paper's control is a history-blind ensemble whose PRESENT observable state is matched;
any later difference is information flowing back from hidden degrees of freedom.
Here the match is exact: after writing bits (phase_memory.py protocol) and waiting --delay
steps, the ENTIRE field (psi and psi_prev, i.e. its full state) of every trial is replaced
by the field of one reference trial. Every trial now has the identical observable present;
only the beads' phases (and held_by) still differ. Then the worlds evolve freely and the
probes are read in windows after the graft.

  bare field : after the graft all trials are identical by construction -> zero information
  chain      : any decodable bit = history stored in hidden bead phases, flowing back
  chain, hidden-reset control: graft the bead phases (theta, held_by) of the reference
               trial instead and keep each trial's own field -> what the field alone holds

    python experiments/matched_present.py --trials 64 --delay 1000
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.phase_memory import build, probes_for, write_wave, demod, cv_decode

ARM_ID = {"field_grafted": 1, "no_graft": 2, "hidden_grafted": 3}


def written(args, base, bits, trial):
    w = fork_world(base)
    w.cfg.phase_noise = args.phase_noise
    w.gen.manual_seed(10_000 + trial)
    kernel = source_kernel(w, (args.N / 2, args.N / 2 + args.source_dy), 1.0)
    for k in range(args.write):
        w.step(external_source=kernel * float(write_wave(args, bits, k)))
    for _ in range(args.delay):
        w.step()
    return w


def read(args, w, xs, windows):
    near, far = probes_for(args, xs)
    pn = FieldProbes(w, near)
    rec = []
    for _ in range(max(e for _, e in windows)):
        w.step()
        rec.append(pn.field(w).numpy().copy())
    rec = np.array(rec)
    return {f"{s}-{e}": rec[s:e] for s, e in windows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="multiplex")
    p.add_argument("--sequence", default="ACGT")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--chain-boost", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--write", type=int, default=600)
    p.add_argument("--amplitude", type=float, default=0.2)
    p.add_argument("--source-dy", type=float, default=2.5)
    p.add_argument("--delay", type=int, default=1000)
    p.add_argument("--trials", type=int, default=64)
    p.add_argument("--phase-noise", type=float, default=0.003)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--out", default="runs/matched_present")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    windows = [(0, 50), (50, 150), (150, 400), (400, 1000)]
    rng = np.random.default_rng(a.seed + 99)
    pats = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]] * (a.trials // 4 + 1))[: a.trials]
    pats = pats[rng.permutation(len(pats))].astype(float)
    results = {}
    for world_name, beads in (("chain", True), ("bare", False)):
        base, xs = build(a, beads)
        ws = [written(a, base, bits, i) for i, bits in enumerate(pats)]
        ref = ws[0]
        ref_psi, ref_prev = ref.psi.clone(), ref.psi_prev.clone()
        ref_theta, ref_held = ref.theta.clone(), ref.held_by.clone()
        arms = {"field_grafted": [], "no_graft": []}
        if beads:
            arms["hidden_grafted"] = []
        for i, w in enumerate(ws):
            for arm in arms:
                v = fork_world(w)
                v.gen.manual_seed(77_000 + i)        # same noise after the graft in every arm
                if arm == "field_grafted":
                    v.psi, v.psi_prev = ref_psi.clone(), ref_prev.clone()
                elif arm == "hidden_grafted":
                    v.theta, v.held_by = ref_theta.clone(), ref_held.clone()
                arms[arm].append(read(a, v, xs, windows))
        n = len(pats); train, test = np.arange(int(0.7 * n)), np.arange(int(0.7 * n), n)
        targets = {"b1": pats[:, 0], "b2": pats[:, 1], "b1*b2": pats[:, 0] * pats[:, 1]}
        for arm, recs in arms.items():
            for wname in recs[0]:
                X = np.array([demod(r[wname], a.sigma, np.random.default_rng(5000 + i + 1000 * ARM_ID[arm]), 0)
                              for i, r in enumerate(recs)])
                raw_spread = float(np.array([r[wname] for r in recs]).std(0).max())
                acc = {k: cv_decode(X, y, a.ridge) for k, y in targets.items()}
                results[f"{world_name}/{arm}/{wname}"] = {**acc, "max_across_trial_std_raw": raw_spread}
                print(f"{world_name:5s} {arm:15s} window {wname:9s} b1={acc['b1']:.2f} b2={acc['b2']:.2f} "
                      f"b1*b2={acc['b1*b2']:.2f}  raw across-trial spread={raw_spread:.2e}", flush=True)
    json.dump({"parameters": vars(a), "results": results}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
