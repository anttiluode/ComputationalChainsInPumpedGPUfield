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

**Implementation status (28 October 2026):** The physics, viewer, templating experiments and saved results are inherited from the original project. **Controlled external field stimulation, spatial/phase probes, and a matched-composition sequence-response experiment are now implemented.** A held-out linear-readout benchmark is now available, but it has not yet been validated on experimental runs; no growth-dependent learning result or demonstrated computational advantage exists yet. A measurable response is not automatically a computation.

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

## Run a controlled input/output experiment

The first computational-instrumentation milestone is implemented in [chains/probing.py](chains/probing.py) and [experiments/sequence_response.py](experiments/sequence_response.py). It adds an **optional external wave source** to the simulator without changing existing no-input runs.

### Quick start

```bash
pip install -r requirements.txt
pip install pytest                    # for tests only
python experiments/sequence_response.py --steps 400 --warmup 300 --seeds 1
python experiments/sequence_response.py --seq-a ACGTACGT --seq-b AGCTAGCT \
    --steps 800 --warmup 500 --seeds 3 --omega 0.11 --out runs/order_test
python experiments/sequence_response.py --kind impulse --steps 500 --out runs/impulse
python experiments/sequence_response.py --mobile --steps 400 --out runs/mobile
python -m pytest -q tests/test_probing.py
```

For a custom waveform, save a JSON array of sample values, such as `[0, 1, 0, -1, 0]`, and provide `--signal-file drive.json`. The input is multiplied by `--amplitude`; `--start` specifies its first step relative to the end of warmup. Other waveforms are `--kind sine`, `square` and `impulse`. Stimulus width and position define a unit-sum periodic spatial source; sine frequency `--omega` is measured in **radians per simulation step**.

The experiment constructs two different sequences with **identical letter counts and bead positions** (default ACGTACGT versus AGCTAGCT), applies the same stimulus, and records five field sensors. For each seed and structure, it runs an identical **undriven counterfactual**, then calculates:

```text
response(t, sensor) = field_driven(t, sensor) - field_undriven(t, sensor)
sequence contrast  = RMS(response_sequence_A - response_sequence_B)
bond contrast      = RMS(response_bonded - response_unbonded)
```

It compares **bonded chains**, **unbonded monomers at the same locations**, and an **empty-field** control. `--chain-boost 0` (the experiment's default) matches emission strength per monomer across the bonded and unbonded cases. Ordinary simulator defaults are unchanged.

Output directory:

- `trace.csv`: time series for driven, undriven and difference traces, by seed, layout, and sensor.
- `summary.json`: full input geometry, parameters, drive schedule, per-layout RMS responses and sequence/bond contrast statistics.

By default, particles are **pinned** so changes in spatial configuration cannot masquerade as sequence effects. That creates an important negative control: with matched emissions and fixed positions, backbone bonds cannot influence the dynamics. **A nonzero sequence contrast in pinned layouts is therefore evidence of frequency arrangement affecting response, not evidence of the bonds doing computation.** Use `--mobile` to permit bond mechanics to influence motion (new ligation stays disabled), and interpret differences alongside geometry controls.

This tool establishes a reproducible measurement protocol. It does **not** train a decoder, assess held-out task accuracy, prove learning, or establish a computational advantage.

### Python API

```python
from chains import Config, World
from chains.probing import Stimulus, FieldProbes, record_response

w = World(Config(N=64, seed=3))
stim = Stimulus(position=(20, 32), kind="sine", amplitude=0.05,
                omega=0.11, duration=250, width=1.0)
trace = record_response(w, stim, [(30, 32), (40, 32)],
                        steps=400, warmup=200, sample_every=2)
# trace["step"], trace["drive"], trace["field"]
# For oscillator phase sensors, also pass monomer_ids=(0, 1, ...)
```

The core world step also accepts `world.step(external_source=field_tensor)`, where `field_tensor` must have the same 2D shape as the field. Omitting it retains the parent simulator's behavior.

## Next experiment: causal bond ablation

The uploaded three-seed mobile run (28 October 2026, `--steps 800 --seeds 3 --mobile`) gave sequence-contrast RMS values of **0.00580/0.00919/0.00433** with bonded chains, versus **0.00858/0.00718/0.00374** with free particles. There was no consistent bonded advantage across seeds. Importantly, the averaged sequence contrast was dominated by the first field probe near the input (**~0.015**), whereas the three far-side probes each showed **~0.0003–0.0004**. These are measurements of a particular layout and drive, not a validated computational advantage.

The next control is now implemented in [experiments/bond_ablation.py](experiments/bond_ablation.py):

```bash
python experiments/bond_ablation.py --steps 600 --warmup 300 --seeds 3
python experiments/bond_ablation.py --steps 600 --seeds 3 --freeze --out runs/ablation_frozen
python experiments/bond_ablation.py --steps 800 --seeds 5 --omega 0.09 --out runs/ablation_009
python -m pytest -q tests/test_ablation.py
```

**Intervention:** Warm up each bonded chain once; then fork all field, oscillator, particle-position and random-generator states *identically*. In one pair of worlds keep the bonds; in a second pair cut them; in a third pair keep the topology but set both mechanical spring constants to zero (**sham**). Apply the same probe to the driven member of each pair, and leave its twin undriven. Ligation, phase noise, thermal noise and bond-dependent emission gain are all disabled for this experiment.

For every output sensor, the causal response comparison is:

```text
(bonded_driven − bonded_undriven)
  − (cut_driven − cut_undriven)
```

This is a *difference-in-differences* in wave response, and is more carefully matched than independently warming up bonded and free worlds. It does not eliminate effects arising from changed particle trajectories; those are part of what the mechanical bond causes. The experiment records both output fields and per-particle positions to assess that mediation.

The test must pass two null controls: **cut vs sham** should agree when mechanical spring forces and bond-specific emission are disabled, and **bonded vs cut after pinning** (`--freeze`) should also agree. If these fail, interpret no other findings until the bug is resolved.

Outputs under `runs/bond_ablation/`:

- `trace.csv`: six raw experimental conditions and the bond-intervention difference trace for every time/sample/probe.
- `positions.npz`: the measured bead trajectories for each run and condition.
- `summary.json`: per-probe and far-output bond-effect RMS, sham-null magnitude, unforced topology contrast and mean bead trajectory separation, including the initial warmed-up microstate.

**Interpretation:** A reproducible nonzero effect at the three distal probes, with passing null controls, would establish that the mechanical links causally alter field transfer under this drive. It would *not* by itself establish a learning mechanism, a reservoir computing advantage, or neural behavior. Those require distinct held-out tasks and appropriate readout controls.

## Next milestone implemented: held-out temporal computation

The 28 October 2026 bond ablation experiment showed a **nonzero distal causal bond effect** across six cases (three seeds, two sequences): output difference-in-differences RMS of **0.000096–0.000159**. The sham-cut and frozen-position controls were **exactly zero** in the user's reported runs. Bonded and cut systems' distal responses remained highly similar (~0.984 pooled correlation); the effect is real within the simulation, but **is not evidence of better computation**. Bonds primarily constrained the particles' positions.

To test whether *that* difference is useful rather than incidental, run the new [temporal_memory.py](experiments/temporal_memory.py) benchmark:

```bat
git pull
python3.13 -m pytest -q tests/test_temporal_memory.py
python3.13 experiments/temporal_memory.py --symbols 320 --seeds 1
python3.13 experiments/temporal_memory.py --symbols 480 --seeds 3 --out runs/memory_three_seeds
python3.13 experiments/temporal_memory.py --symbols 320 --seeds 1 --freeze --out runs/memory_frozen
```

**Protocol:** Generate random independent input symbols (+1 or −1), hold each for 12 simulation steps, sample 15 **distal** field sensors, and train a ridge-regularized **linear readout** on the first 70% of the input stream. Discard initial washout samples and add a delay-sized gap before evaluating on the final chronological held-out segment. All reservoir physics and readout settings are fixed in advance; the readout training never sees test targets. Each seed and sequence receives the same random input stream.

Four systems are measured: **bonded chain**, **mechanically cut chain**, **sham chain with zero bond/bend springs**, and **empty field**. The bonded/cut/sham worlds are exact copies of the same warmed-up microstate before the intervention. Thermal/phase noise, growth and bond-dependent emission gain are disabled.

The tasks are recovery of past input bits at delays 0, 1, 2, 4 and 8, and nonlinear two-bit products (a parity/XOR-equivalent target). The script reports held-out `test_r2` (1 is perfect, 0 is the train-mean baseline, negative is worse) and binary `sign_accuracy` (about 50% is chance for balanced bits). A bare field is an essential baseline because even a linear damped wave field stores recent input history.

Output `runs/temporal_memory/summary.json` includes task metrics, probe positions, randomized-input seeds and null-check statistics. `samples.npz` contains sampled wave-field features and full input streams for independent analysis.

**What would matter:** A consistent held-out task improvement of bonded over both cut and bare, with sham equal to cut and frozen bonded equal to cut. The new experiment may instead yield no advantage; that would be a meaningful negative finding. This is *not* a growth-dependent learning experiment: the chains are preassembled and warmed up rather than trained to change their connections. A truly self-grown computational medium is the next separate test.

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

**Now available:** controlled input/output probing via `experiments/sequence_response.py`, causal post-warmup bond ablation via `experiments/bond_ablation.py`, and held-out ridge-readout benchmark via `experiments/temporal_memory.py`. **Not yet available:** self-grown task-dependent networks or a validated computational advantage. The inherited commands above remain unchanged.

## Development order

1. **Controlled stimuli and probes — first version implemented.** Per-seed measurements and reproducible input schedules now work; checkpoint/restart remains to be added.
2. **Measure and validate sequence-order transfer functions.** The paired protocol exists; run seed ensembles, multiple input frequencies and orientation/phase-shuffle controls before drawing conclusions.
3. **Ablate bond topology causally (implemented), then add grow/set/probe lifecycle.** A new post-warmup fork-and-cut experiment measures the effect of existing bonds; the *learning-by-growth* experiment is still future work.
4. **Benchmark temporal and nonlinear memory (initial version implemented).** A linear readout and strict held-out bonded/cut/sham/bare controls are available; full task outcomes remain to be measured.
5. **Optimize GPU neighbor search**, then repeat across large ensembles and examine pump-programmable computation.

This repository is deliberately separate from the original: **copying is one possible outcome of the physics, but computing with structures formed by that physics is the goal here.** The test is not whether the scene resembles neurons or molecules. The test is whether its grown physical state performs a reproducible computation that the appropriate simpler controls cannot explain.
