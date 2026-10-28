# Computational Chains in a Pumped GPU Field

![Pumped wave-field chain simulation](pic.png)

**Can a wave-driven material grow its own computational structure?**

This repository investigates **self-assembling physical computation**. Oscillating monomers write to a shared field, respond to one another's waves, and form persistent, type-blind backbone bonds. The resulting chains alter the field that subsequently drives them.

The central hypothesis is not that a chain must first become DNA. It is that **a system can accumulate interaction history as physical connectivity, and that changed connectivity can change how it processes future inputs**.

> Input changes activity → activity changes connections → connections change future responses.

## How this differs from the parent project

This is a computational research branch of [ChainsInPumpedGPUfield](https://github.com/anttiluode/ChainsInPumpedGPUfield), which asks whether complementary pairing and template copying can arise from frequency-selective wave interactions without a hard-coded letter-binding table.

| | Original: ChainsInPumpedGPUfield | This branch: ComputationalChainsInPumpedGPUfield |
|---|---|---|
| Primary question | Can monomers assemble complementary chains and copy them? | Can self-assembled chains **store, transform and use information**? |
| Main object of study | Template, complement, copying fidelity | **Structure, dynamics, memory and input–output response** |
| Success criterion | Faithful reusable copying across generations | Reproducible, nontrivial **computation attributable to the grown structure** |
| Main failure mode to investigate | Runaway growth and poor round-2 copying | Apparent computation explained by the bare field, instantaneous input, geometry alone or trivial readout |
| Role of DNA-like chemistry | Goal | Optional way to make persistent structures; replication is **not required** |

**Current implementation status (28 October 2026):** The physics, viewer, templating experiments and saved results are inherited from the original project. **No trained reservoir readout, controlled computational benchmark, learnable input interface, or demonstrated sequence-to-function advantage is implemented yet.** A renamed repository is not itself evidence of computation. The experiments below are the intended research program.

## What the existing simulator gives us

- **A shared, damped, discrete Klein–Gordon-like field** with a previous-time state, spatial coupling and an externally driven pump.
- **Oscillator monomers** with position, phase and one of four intrinsic frequencies (A, C, G, T).
- **Frequency-selective interactions** in the default pumped-receiver mode. The chosen frequencies satisfy `ω_A + ω_T = ω_C + ω_G = 2Ω` at the default pump frequency.
- **History-dependent, type-blind ligation.** Free backbone ends can join when the particles are simultaneously held, remain near the backbone distance, and accumulate sufficient co-residence score. Letters are not consulted by the ligation rule.
- **Persistent chains**, spring-like bonds and bending forces. Bonded monomers also emit more strongly, so growth changes the wave sources themselves.
- **PyTorch CPU/CUDA simulation**, a Pygame live viewer and headless JSON logging.

This already closes a feedback loop:

```text
wave field → oscillator response → motion and prolonged contact → new bond
     ↑                                                   |
     └────────── changed geometry and emission ─────────┘
```

Whether that loop realizes useful *computation* remains an experimental question.

## Research program

### 1. Sequence → physical function

**Question:** If two chains contain the same numbers of A, C, G and T, but in different orders, do they produce measurably different responses to identical probes?

Proposed protocol:

1. Construct equal-length chains with the **same letter counts**, changing only their order.
2. Match initial geometry, orientation, phase treatment, pump and input amplitudes; repeat across seeds.
3. Apply controlled pulses or oscillatory inputs at a defined location.
4. Record field response and forces at multiple receiver locations, including time-resolved phase and amplitude.
5. Compare to order-shuffled chains, reverse sequences, unbonded monomers and source-matched controls.

Candidate outputs: impulse responses, spectral selectivity, directional output, nonlinear mixing and response to stimulus pairs. A classifier distinguishing chain orders is *not* enough by itself: a useful effect must survive nuisance controls and demonstrate a functional difference, ideally on held-out input conditions.

**Milestone:** Establish an experimentally reproducible sequence-dependent transfer function not explained solely by composition or trivial spatial displacement.

### 2. Grow → set → compute

**Question:** Does a network grown under a particular history process future inputs better than a random or independently grown network?

Proposed protocol:

1. **Grow:** expose a soup to a structured, reproducible stimulation history while normal ligation is enabled.
2. **Set:** prevent further structural change; immobilize or otherwise control geometry while optionally allowing oscillator phases and the field to keep evolving.
3. **Probe:** present new input sequences, record a fixed collection of field samples and/or oscillator observables.
4. **Read:** fit only a lightweight regularized linear output layer on training trials, keeping physics and topology fixed during readout training.
5. **Test:** evaluate on new seeds and input sequences, not simply on the stimulus that built the network.

Compare against at least:

- Bare pumped field, with no monomers.
- Unbonded monomers, with source power matched where practical.
- Random and shuffled connectivity with comparable size, density, geometry and total emission.
- The same final structure with randomized phase/history.
- A frozen network grown without structured input.

**Milestone:** An advantage in *held-out task performance* attributable specifically to the grown structure and its interaction history, rather than merely having more sources or a more capable output reader.

### 3. Temporal memory and nonlinear computation

The field already has two time slices (`psi` and `psi_prev`), monomer phases, a running `held_by` average, and decaying pair co-residence scores. That is physical and algorithmic history; it is **not yet evidence of usable task memory**.

Proposed tasks:

- **Linear memory:** recover a past input `u[t-d]` using current measured state as delay `d` increases.
- **Nonlinear memory:** recover `u[t] * u[t-d]` or delayed XOR from current state using a linear readout.
- **History-dependent prediction:** distinguish identical observed present states reached through different past drives.
- **Memory backflow:** compare future response ensembles matched on present observables but conditioned on differing histories.

Sweep input cadence, probe spacing, pump frequency, damping, network growth duration and bond density. Use temporally separated train/test segments and shuffled-history controls. Report memory versus delay, task error, seed variability and computational cost—not visual complexity alone.

The [Muruga et al. *Nature Physics* paper on work extraction from hidden degrees of freedom](https://doi.org/10.1038/s41567-026-03442-w) suggests a useful **measurement-control strategy**: compare histories leading to matched present observations before assigning a benefit to hidden memory. This simulator is externally driven, so improved prediction or control **must not be presented as thermodynamic work extraction** without a separately defined energy accounting and feedback protocol.

**Milestone:** Show a history-based improvement above instantaneous-input, bare-field and shuffled-history baselines.

### 4. Pump-programmable interactions

The interaction preferences depend on the pump and the oscillator frequencies. With the default alphabet, simple resonance arithmetic predicts:

| Pump frequency `2Ω` | Candidate sum resonance |
|---|---|
| 0.14 | A–T and C–G |
| 0.12 | A–G |
| 0.16 | C–T |
| 0.10 | C–C |
| 0.18 | G–G |

**These are candidate resonances, not demonstrated forces or guaranteed stable pairing regimes.** Pair tests are needed because finite linewidth, phase locking, field response and pump stability matter.

Explore whether changing the pump can redirect assembly or alter an *already grown* network's transfer function. Compare matched initial states and normalize total input energy.

**Milestone:** Reconfigurable response or assembly attributable to pump choice, not a hand-coded pair matrix.

## Existing evidence, and what it does *not* establish

The copied `results/` directory contains **parent-project baseline experiments**, not computational results from this branch:

| Inherited result | Reported outcome | Interpretation |
|---|---|---|
| Pinned pair table, receiver mode | ~45× attraction selectivity for designed complements versus other pairs | Frequency-selective physical interaction in the chosen setup |
| Template-site recruitment, 3 receiver seeds | 60/102 = 59% complementary (25% nominal chance baseline) | Selective recruitment at templates |
| Pumped-medium control | 22/100 = 22% complementary | Substantial linking without corresponding selective recruitment |
| Round-two template test | 18/70 = 26% complementary | **No demonstrated reusable replication** |
| Dense soup | Long sticky chains and clumps | The original dynamics favor growth more readily than faithful copying |

These data motivate studying self-built networks, **but they do not prove learning, memory-capacity improvements, signal processing advantages or sequence-to-function behavior**. The receiver observations are not necessarily independent samples; independent world seeds and better-matched controls remain important.

## Physical model

The discrete wave-field update is:

```text
psi_next = (2 - gamma)*psi
         - (1 - gamma)*psi_prev
         + c2*laplacian(psi)
         - m2(t)*psi
         + sources

defaults: c2 = 0.3, m2 = 0.02, gamma = 0.01
```

Each monomer emits `g_emit * cos(theta)`, hears the sampled field, updates its phase and moves in response to the pumped gradient plus collision and bond forces. In default `receiver` mode the hearing/force coupling is `cos(2*Omega*t)`; in `medium` mode the mass term is modulated instead.

| Letter | Natural frequency (rad/step) |
|---|---:|
| A | 0.03 |
| C | 0.05 |
| G | 0.09 |
| T | 0.11 |

Default `Omega = 0.07`, hence `2*Omega = 0.14`. The gapped field gives sub-gap sources a predominantly localized, screened response. The monomers' emission/hearing setup is actively driven and non-reciprocal by construction; it is a model choice, **not a claim of passive equilibrium physics**.

Backbone bonds are type-blind: two available chain ends must be within `d0 = 3.6 ± 0.3`, currently held above `held_min`, and have accumulated sufficient decaying co-residence score before stochastic ligation. Each monomer has at most two backbone neighbors. The existing `chain_boost` raises emissions from bonded monomers: this positive feedback must be controlled in computational comparisons.

**Scalability note:** the current all-pairs core and candidate-ligation distance computations grow approximately quadratically with particle count. Spatial neighbor search or cell lists will be needed to make large GPU reservoir experiments practical.

## Run the inherited baseline

```bash
pip install -r requirements.txt
python run.py                           # live wave field with a GATTACAG template
python run.py --no-template             # assembly without the pinned template
python run.py --N 256 --n_each 600      # larger world; CUDA if available
python run.py --headless --steps 50000 --out runs/baseline

python experiments/pair_table.py        # inherited attraction test
python experiments/gate.py --seeds 3 --round2  # inherited templating test
python -m pytest -q tests
```

Live-viewer keys: `SPACE` pause, `P` toggle pump, `F` toggle field display, `+/-` change steps per frame, `S` save screenshot, left-drag stir, `ESC` exit. A red, C yellow, G green, T blue; white lines are backbone bonds.

**Not yet available:** a CLI for controlled input/output probing, growing and freezing a reservoir, or benchmarking a trained readout. Those require new code; the commands above reproduce the inherited assembly simulator.

## Development order

1. **Add controlled stimulus and probe APIs.** Keep reproducible input schedules, field/phase readings, checkpoints and per-seed outputs.
2. **Establish sequence-order transfer functions.** Same composition, matched geometry and drive, shuffled-order controls.
3. **Add a grow/set/probe lifecycle.** Separate assembly time from measurement and permit a fixed structure with ongoing wave/phase dynamics.
4. **Benchmark temporal and nonlinear memory.** Start with a linear readout and strict held-out controls.
5. **Optimize GPU neighbor search**, then repeat across large ensembles and examine pump-programmable computation.

This repository is deliberately separate from the original: **copying is one possible outcome of the physics, but computing with structures formed by that physics is the goal here.** The test is not whether the scene resembles neurons or molecules. The test is whether its grown physical state performs a reproducible computation that the appropriate simpler controls cannot explain.
