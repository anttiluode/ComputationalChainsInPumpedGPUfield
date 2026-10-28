"""Causal mechanical-bond ablation from identical warmed-up simulator states.

python experiments/bond_ablation.py --steps 600 --warmup 300 --seeds 3

Four key counterfactual conditions share an *identical* initial state at the
intervention: bonded vs cut, each with and without the external stimulus.
A fifth/sixth pair is a "sham": retain topology but set spring constants to
zero. Since chain_boost=0 and ligation is disabled, cut and sham should coincide.

No computation/learning claim follows automatically from a nonzero difference.
"""
import argparse
import copy
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from chains import Config, World
from chains.probing import FieldProbes, Stimulus, source_kernel
from experiments.sequence_response import layouts


CONDITIONS = ("bonded", "cut", "sham")
TRIALS = ("driven", "undriven")


def fork_world(world):
    """Full mutable-state fork without redoing the 3,000-step calibration.

    Stable arrays (frequencies, kernel offsets, calibrated self response) may
    be shared read-only. All dynamics, RNGs, geometry and topology are distinct.
    """
    w = copy.copy(world)
    w.cfg = copy.deepcopy(world.cfg)
    for name in ("psi", "psi_prev", "pos", "theta", "typ", "nb",
                 "held_by", "pinned", "field_force"):
        setattr(w, name, getattr(world, name).clone())
    w.gen = torch.Generator(device=world.device)
    w.gen.set_state(world.gen.get_state().clone())
    w.rng = np.random.default_rng()
    w.rng.bit_generator.state = copy.deepcopy(world.rng.bit_generator.state)
    w.held = copy.deepcopy(world.held)
    w.events = list(world.events)
    return w


def build_counterfactuals(base, freeze=False):
    """Identical pre-intervention microstate; only mechanical bond law changes."""
    worlds = {}
    for condition in CONDITIONS:
        for trial in TRIALS:
            w = fork_world(base)
            if condition == "cut":
                w.nb.fill_(-1)
            elif condition == "sham":
                w.cfg.k_bond = 0.0
                w.cfg.k_bend = 0.0
            if freeze:
                w.pinned[:] = True
            worlds[(condition, trial)] = w
    return worlds


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def wrapped_distance(a, b, N):
    diff = np.remainder(a - b + N / 2, N) - N / 2
    return np.linalg.norm(diff, axis=-1)


def measure(args, sequence, positions, source, sensors, seed):
    config = Config(N=args.N, seed=seed, device=args.device, kT=0.0,
                    phase_noise=0.0, p_lig=0.0, chain_boost=0.0)
    world = World(config)
    world.add_chain(sequence, positions[0])
    for _ in range(args.warmup):
        world.step()
    initial = {"step": world.t, "position": world.pos.detach().cpu().tolist(),
               "phase": world.theta.detach().cpu().tolist(),
               "field_norm": float(torch.linalg.vector_norm(world.psi).item())}

    states = build_counterfactuals(world, freeze=args.freeze)
    sensors = FieldProbes(world, sensors)
    signal = Stimulus(position=source, kind=args.kind, amplitude=args.amplitude,
                      omega=args.omega, start=args.start, duration=args.duration,
                      width=args.width)
    kernel = source_kernel(world, source, width=args.width)
    readings = {key: {"field": [], "position": []} for key in states}
    clock, drive = [], []
    for t in range(args.steps):
        val = signal.value(t)
        for (condition, trial), state in states.items():
            external = kernel * val if (trial == "driven" and val != 0) else None
            state.step(external_source=external)
        if (t + 1) % args.sample_every != 0 and t + 1 != args.steps:
            continue
        clock.append(t + 1)
        drive.append(val)
        for key, state in states.items():
            readings[key]["field"].append(sensors.field(state).detach().cpu().numpy().copy())
            readings[key]["position"].append(state.pos.detach().cpu().numpy().copy())
    for row in readings.values():
        row["field"] = np.stack(row["field"])
        row["position"] = np.stack(row["position"])
    response = {cond: readings[(cond, "driven")]["field"] -
                readings[(cond, "undriven")]["field"] for cond in CONDITIONS}
    # A difference-in-differences: removes the separate spontaneous waveform
    # of each post-intervention topology, but not its trajectory response.
    interaction = response["bonded"] - response["cut"]
    unforced_difference = (readings[("bonded", "undriven")]["field"] -
                           readings[("cut", "undriven")]["field"])
    sham_difference = response["sham"] - response["cut"]
    geometry = wrapped_distance(readings[("bonded", "driven")]["position"],
                                readings[("cut", "driven")]["position"], args.N)
    input_induced_geometry = wrapped_distance(
        readings[("bonded", "driven")]["position"],
        readings[("bonded", "undriven")]["position"], args.N)
    stats = {"seed": seed, "sequence": sequence,
             "interaction_rms_by_probe": [rms(interaction[:, i]) for i in range(interaction.shape[1])],
             "interaction_output_rms": rms(interaction[:, 2:]),
             "interaction_center_rms": rms(interaction[:, 1]),
             "interaction_near_input_rms": rms(interaction[:, 0]),
             "bonded_output_response_rms": rms(response["bonded"][:, 2:]),
             "cut_output_response_rms": rms(response["cut"][:, 2:]),
             "sham_minus_cut_rms": rms(sham_difference),
             "unforced_bond_cut_rms": rms(unforced_difference),
             "bond_cut_geometry_mean_distance": float(np.mean(geometry)),
             "bonded_input_induced_geometry_mean_distance": float(np.mean(input_induced_geometry)),
             "bonded_output_to_cut_output_rms_ratio": (
                 rms(response["bonded"][:, 2:]) /
                 max(rms(response["cut"][:, 2:]), 1e-15)),
             "initial": initial}
    return stats, np.asarray(clock), np.asarray(drive), readings, interaction


def run(args):
    seqs, positions, source, sensors = layouts(args.seq_a, args.seq_b, args.N, Config().d0)
    if args.seeds < 1 or args.steps < 1 or args.warmup < 0 or args.sample_every < 1:
        raise ValueError("seeds/steps/sample_every must be positive and warmup >= 0")
    if args.start < 0:
        raise ValueError("start must be >= 0")
    Stimulus(source, kind=args.kind, duration=args.duration, start=args.start,
             amplitude=args.amplitude, width=args.width, omega=args.omega)
    os.makedirs(args.out, exist_ok=True)
    rows, stats, trajectories = [], [], {}
    for seed in range(args.seed, args.seed + args.seeds):
        for label, sequence in (("A", seqs[0]), ("B", seqs[1])):
            item, clock, drive, readings, interaction = measure(
                args, sequence, positions, source, sensors, seed)
            item["label"] = label
            stats.append(item)
            for j, step in enumerate(clock):
                for condition in CONDITIONS:
                    for trial in TRIALS:
                        row = readings[(condition, trial)]
                        rows.append([seed, label, condition, trial, int(step),
                                     float(drive[j]) if trial == "driven" else 0.0,
                                     *[float(v) for v in row["field"][j]]])
                rows.append([seed, label, "bond_effect", "difference_in_differences",
                             int(step), float(drive[j]), *[float(v) for v in interaction[j]]])
            for condition in CONDITIONS:
                for trial in TRIALS:
                    trajectories[f"seed{seed}_{label}_{condition}_{trial}"] = (
                        readings[(condition, trial)]["position"])
            print(f"seed {seed} seq {label}: causal bond effect at outputs "
                  f"{item['interaction_output_rms']:.6g}, "
                  f"sham-null {item['sham_minus_cut_rms']:.3g}, "
                  f"position separation {item['bond_cut_geometry_mean_distance']:.4g}",
                  flush=True)

    with open(os.path.join(args.out, "trace.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "label", "structure", "trial", "step", "drive",
                         *[f"probe_{i}" for i in range(len(sensors))]])
        writer.writerows(rows)
    np.savez_compressed(os.path.join(args.out, "positions.npz"), **trajectories)
    summary = {"experiment": "counterfactual_bond_ablation",
               "description": "Same warmed-up world state; bonds kept, cut, or mechanically disabled",
               "sequences": list(seqs), "source": source, "probes": sensors,
               "output_probe_indices": [2, 3, 4],
               "parameters": vars(args),
               "physics_config": json.loads(Config(
                   N=args.N, seed=args.seed, device=args.device, kT=0.0,
                   phase_noise=0.0, p_lig=0.0, chain_boost=0.0).to_json()),
               "statistics": stats,
               "meaning": {
                   "interaction": "(bonded_driven - bonded_undriven) - "
                                  "(cut_driven - cut_undriven)",
                   "sham_null": "Retain bonds but set k_bond=k_bend=0; should match cut",
                   "freeze": "All monomers pinned after warmup; bond effect must be zero",
                   "limitations": "This tests a causal influence of mechanical bonds on "
                                  "field response, not information processing ability. "
                                  "The response may arise from changed trajectories; "
                                  "compare distal outputs and input-conditioned motion."
               }}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {args.out}/trace.csv, positions.npz, summary.json", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-a", default="ACGTACGT")
    p.add_argument("--seq-b", default="AGCTAGCT")
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--kind", choices=("sine", "impulse", "square"), default="sine")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--duration", type=int, default=250)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--omega", type=float, default=0.11, help="radians per step")
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--freeze", action="store_true",
                   help="pin particles after warmup; bond effect should vanish")
    p.add_argument("--out", default="runs/bond_ablation")
    run(p.parse_args())


if __name__ == "__main__":
    main()
