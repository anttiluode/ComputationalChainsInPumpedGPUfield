"""Matched-composition stimulus/response experiment.

Run directly: python experiments/sequence_response.py --steps 400 --seeds 2
The readout is deliberately untrained. Separate undriven trials allow us to
subtract each layout's own ongoing oscillator emissions from the probed response.
"""
import argparse
from collections import Counter
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from chains import Config, World
from chains.probing import Stimulus, record_response


def layouts(seq_a, seq_b, N, d0):
    seq_a, seq_b = seq_a.upper(), seq_b.upper()
    letters = set(Config().alphabet)
    if not seq_a or seq_a == seq_b or Counter(seq_a) != Counter(seq_b):
        raise ValueError("sequences must be different and have identical letter counts")
    if any(ch not in letters for ch in seq_a + seq_b):
        raise ValueError("sequences may only contain A, C, G and T")
    if (len(seq_a) - 1) * d0 >= N / 2 - 8:
        raise ValueError("chain too long for a centrally placed, isolated probe geometry")
    left = N / 2 - (len(seq_a) - 1) * d0 / 2
    y = N / 2
    positions = [(left + i * d0, y) for i in range(len(seq_a))]
    source = (left - 5, y)
    sensors = [(left - 2, y), (N / 2, y),
               (positions[-1][0] + 3, y), (positions[-1][0] + 7, y),
               (positions[-1][0] + 3, y + 5)]
    return (seq_a, seq_b), positions, source, sensors


def setup_world(args, seed, sequence, structure, positions):
    cfg = Config(N=args.N, seed=seed, device=args.device,
                 phase_noise=0.0, kT=0.0, p_lig=0.0,
                 chain_boost=args.chain_boost)
    world = World(cfg)
    if structure == "bonded":
        ids = world.add_chain(sequence, positions[0])
    elif structure == "unbonded":
        ids = world.add_monomers([world.letters.index(ch) for ch in sequence],
                                 pos=positions).tolist()
    elif structure == "bare":
        ids = []
    else:
        raise ValueError("unrecognized structure")
    if not args.mobile:
        world.pinned[:] = True
    return world, ids


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def run_pair(args, seed, sequence, structure, positions, stimulus, sensors):
    results = {}
    for trial in ("driven", "undriven"):
        world, ids = setup_world(args, seed, sequence, structure, positions)
        results[trial] = record_response(
            world, stimulus if trial == "driven" else None, sensors,
            steps=args.steps, warmup=args.warmup,
            sample_every=args.sample_every)
    a, b = results["driven"], results["undriven"]
    if not np.array_equal(a["step"], b["step"]):
        raise AssertionError("paired trials have mismatched sampling times")
    results["response"] = a["field"] - b["field"]
    return results


def run(args):
    seqs, positions, source, sensors = layouts(args.seq_a, args.seq_b, args.N, Config().d0)
    if args.steps < 1 or args.warmup < 0 or args.sample_every < 1 or args.seeds < 1:
        raise ValueError("steps, sample_every and seeds must be positive; warmup >= 0")
    if args.start < 0:
        raise ValueError("start must be >= 0")
    samples = ()
    kind, duration = args.kind, args.duration
    if args.signal_file:
        with open(args.signal_file, encoding="utf-8") as fp:
            samples = tuple(float(v) for v in json.load(fp))
        kind, duration = "samples", len(samples)
    stimulus = Stimulus(position=source, kind=kind, amplitude=args.amplitude,
                        omega=args.omega, start=args.start, duration=duration,
                        width=args.width, samples=samples)
    os.makedirs(args.out, exist_ok=True)
    records, summaries = [], []
    for seed in range(args.seed, args.seed + args.seeds):
        experiments = {}
        for structure in ("bonded", "unbonded", "bare"):
            for label, seq in (("A", seqs[0]), ("B", seqs[1])) if structure != "bare" else (("none", ""),):
                outcome = run_pair(args, seed, seq, structure, positions, stimulus, sensors)
                experiments[(structure, label)] = outcome["response"]
                for j, step in enumerate(outcome["driven"]["step"]):
                    for trial in ("driven", "undriven", "response"):
                        field = (outcome[trial]["field"][j] if trial != "response"
                                 else outcome["response"][j])
                        records.append([seed, structure, label, trial, int(step),
                                        float(outcome["driven"]["drive"][j]) if trial == "driven" else 0.0,
                                        *[float(v) for v in field]])
                summaries.append({"seed": seed, "structure": structure,
                                  "sequence": seq or None, "label": label,
                                  "response_rms": rms(outcome["response"])})
        for structure in ("bonded", "unbonded"):
            contrast = rms(experiments[(structure, "A")] - experiments[(structure, "B")])
            summaries.append({"seed": seed, "structure": structure,
                              "metric": "sequence_response_contrast_rms", "value": contrast})
        for label in ("A", "B"):
            difference = rms(experiments[("bonded", label)] - experiments[("unbonded", label)])
            summaries.append({"seed": seed, "metric": "bond_vs_unbonded_response_rms",
                              "label": label, "value": difference})
        summaries.append({"seed": seed, "metric": "bare_field_response_rms",
                          "value": rms(experiments[("bare", "none")])})
        print(f"seed {seed}: bonded sequence contrast="
              f"{rms(experiments[('bonded', 'A')] - experiments[('bonded', 'B')]):.6g}; "
              f"unbonded sequence contrast="
              f"{rms(experiments[('unbonded', 'A')] - experiments[('unbonded', 'B')]):.6g}",
              flush=True)
    with open(os.path.join(args.out, "trace.csv"), "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["seed", "structure", "sequence_label", "trial", "relative_step",
                         "drive_amplitude", *[f"probe_{i}" for i in range(len(sensors))]])
        writer.writerows(records)
    report = {"experiment": "matched_composition_sequence_response",
              "sequences": list(seqs), "positions": positions,
              "source": source, "probes": sensors,
              "drive": {"kind": stimulus.kind, "amplitude": stimulus.amplitude,
                        "omega_rad_per_step": stimulus.omega, "start": stimulus.start,
                        "duration": stimulus.duration, "width": stimulus.width,
                        "samples": list(stimulus.samples)},
              "parameters": {k: v for k, v in vars(args).items() if k != "signal_file"},
              "physics_config": json.loads(Config(N=args.N, seed=args.seed, device=args.device,
                                                  phase_noise=0.0, kT=0.0, p_lig=0.0,
                                                  chain_boost=args.chain_boost).to_json()),
              "measure": "RMS of driven field minus matched undriven field at fixed probes",
              "summary": summaries,
              "caveats": ["Fixed bonded and unbonded layouts with chain_boost=0 should be dynamically indistinguishable: bonds have no mechanical effect if motion is pinned.",
                          "Sequence contrasts can reflect frequency placement, not computation or topology.",
                          "This experiment trains no readout and measures no task performance."]}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"Saved {args.out}/trace.csv and summary.json", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-a", default="ACGTACGT")
    p.add_argument("--seq-b", default="AGCTAGCT")
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--sample-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--kind", choices=("impulse", "square", "sine"), default="sine")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--duration", type=int, default=250)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--omega", type=float, default=0.11, help="drive frequency, rad/step")
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--signal-file", help="JSON array of per-step samples (overrides --kind/--duration)")
    p.add_argument("--chain-boost", type=float, default=0.0,
                   help="0 matches per-monomer emission in bonded/unbonded controls")
    p.add_argument("--mobile", action="store_true",
                   help="allow motion and bond forces; ligation remains disabled")
    p.add_argument("--out", default="runs/sequence_response")
    run(p.parse_args())


if __name__ == "__main__":
    main()
