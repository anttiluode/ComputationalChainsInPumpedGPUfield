# The way out: frequency-addressed phase memory (29 Oct 2026, morning)

Overnight (`RESULTS_OVERNIGHT.md`) tested one interface and over-generalised from it. Astra's
review was right on the main points, and the Nature Physics paper it pointed to (Muruga, Ginot,
Loos & Bechinger, *The extraction of work from hidden degrees of freedom*, 2026) supplied the
reframe: **memory lives in hidden degrees of freedom, a single snapshot cannot see it, and it
is harvested by time-delayed measurement against a history-blind control with the present
matched.** This file reports what happened when the chains were tested that way.

Every number comes from a `summary.json` under `runs/`. Decoding is 8-fold cross-validated
over trials, with Gaussian probe noise σ = 1e-3 added before demodulation unless stated.

## 1. Why overnight failed, and what the physics actually offers

`world.py` pushes a bead's phase by `c(t)·ψ·sin θ` with pump `c(t) = cos 2Ωt`. That has a DC part
only when the field holds the bead's **pump complement** `2Ω − ω`. Overnight's hold-encoded bits
had no such component, so they never wrote the beads. They only added interference.

For a complement pair (A+T or C+G, frequencies summing to 2Ω), the pump couples
`cos 2Ωt · cos θ_j · sin θ_i`, whose DC part depends on `θ_i + θ_j − 2Ωt`. So:

* the **sum** phase is locked to the pump. It has a restoring force and forgets;
* the **difference** phase `θ_i − θ_j` is neutral. It has no restoring force and **remembers**.

Measured (`tests/test_phase_memory_physics.py`; 600-step write at ±carrier, then silence):

| pair | t after write start | Δ sum (rad) | Δ difference (rad) |
|---|---|---|---|
| A/T (carrier 0.03) | 600 | −2.12 | 0.07 |
| A/T | 6600 | −0.12 | **1.73** |
| C/G (carrier 0.09) | 600 | −1.68 | 2.41 |
| C/G | 6600 | 0.05 | **−2.47** |

The write first kicks the sum. The pump lock pulls the sum back and leaves the shift in the
difference. Overnight's fading-memory benchmark (lags 0–30 symbols, random bits) could not
see a non-fading integrator, and its input never drove one anyway.

Addressing comes from the arithmetic, not a lookup table: carrier 0.03 or 0.11 writes the A/T pair,
0.05 or 0.09 writes the C/G pair. **The chain's sequence is the address map, frequency is the
address, the difference phase is the storage.**

## 2. Write → wait → read beats the bare field (`phase_memory.py`, 128 trials)

Block `ACGT`, pinned. b1 is written on 0.03 and b2 on 0.09, together (amplitude 0.2, 600 steps).
Silence, then passive 400-step reads. "Near" probes sit 2–3.5 cells from the beads; "far"
probes sit 18–20 cells away and see the pump sidebands (ω + 2Ω, above the gap). The bare field
gets the identical write.

Phase noise 0.003 / step (`runs/pm_mux_a02_n003`):

| delay (steps) | chain b1 | chain b2 | chain b1·b2 | bare b1 | bare b2 | bare b1·b2 |
|---|---|---|---|---|---|---|
| 600 | 1.00 | 1.00 | 0.99 | 0.90 | 0.96 | 0.43 |
| 2000 | 1.00 | 1.00 | 1.00 | 0.52 | 0.52 | 0.39 |
| 6000 | 0.98 | 1.00 | 0.95 | 0.52 | 0.52 | 0.38 |

Far probes, 6000 steps: chain 0.98 / 1.00 / 0.94, bare 0.50 / 0.47 / 0.57.

Default phase noise 0.01 / step (`runs/pm_mux_a02_n010`), near probes: chain 0.83 / 0.91 / 0.78
at 2000 steps and 0.71 / 0.83 / 0.63 at 6000; bare 0.52 / 0.52 / 0.39 at both. The memory
lifetime is set by phase diffusion, as it should be for a neutral mode.

**Is the XOR real nonlinearity?** With four patterns, "b1·b2 decodable" means "the response is
not additive in the two bits". `interaction_contrast.py` measures that directly: the size of
f(++) − f(+−) − f(−+) + f(−−) in units of trial-to-trial noise (maximum over 64 features; null
level ≈ 2.5–3):

| | 600 | 2000 | 6000 |
|---|---|---|---|
| chain, noise 0.003 | 11.3σ | 11.3σ | 7.8σ |
| chain, noise 0.01 | 3.1σ | 2.9σ | 1.4σ |
| bare | 1.3σ | 1.3σ | 1.6σ |

The bare field is exactly additive, as a linear medium must be. The chain is not.

## 3. XOR by phase addition (`phase_memory.py --mode addition`)

Both bits are written into the **same** register, one after the other (carrier 0.03,
amplitude 0.3). The difference phase stores the sum. When each write turns it by about π/2,
the cosine emission makes b1·b2 a linear function of the field: interference parity, the
Kaiku result, done by matter instead of by the readout.

`runs/pm_add_a03_n003`, 128 trials, phase noise 0.003, near probes:

| delay | chain b1 | chain b2 | chain **b1·b2** | bare b1 | bare b2 | bare b1·b2 | interaction, chain / bare |
|---|---|---|---|---|---|---|---|
| 100 | 1.00 | 1.00 | **1.00** | 0.98 | 1.00 | 0.41 | 30.4σ / 1.6σ |
| 600 | 0.93 | 0.98 | **1.00** | 0.51 | 0.86 | 0.41 | 25.0σ / 1.3σ |
| 2000 | 0.73 | 0.73 | **0.99** | 0.52 | 0.51 | 0.39 | 12.7σ / 1.3σ |
| 6000 | 0.78 | 0.73 | **0.97** | 0.52 | 0.52 | 0.38 | 9.5σ / 1.6σ |

Far probes at 6000 steps: chain b1·b2 0.94, bare 0.57.

The register stores b1 + b2. That sum fixes the XOR exactly, but a sum of 0 cannot say which
bit was +1, so single bits cap near 75%. That ceiling is the signature of an adder.

## 4. The paper's protocol: matched present, hidden history (`matched_present.py`, 128 trials)

After the write and a 1000-step wait, the **entire field state** (ψ and ψ_prev) of every trial
is replaced by one reference trial's. Every trial now has an identical observable present.
Then the worlds run on and are read in windows after the graft.

| arm | window 0–50 | 50–150 | 150–400 | 400–1000 |
|---|---|---|---|---|
| bare, field grafted | 0.49 / 0.50 / 0.51 | 0.53 / 0.45 / 0.53 | 0.52 / 0.52 / 0.55 | 0.50 / 0.50 / 0.48 |
| chain, field grafted | 1.00 / 1.00 / 0.98 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| chain, bead phases grafted instead | 1.00 / 1.00 / 0.94 | 1.00 / 1.00 / 0.92 | 1.00 / 0.99 / 0.65 | 0.84 / 0.66 / 0.44 |

Entries are b1 / b2 / b1·b2. After the graft, the bare field's trials differ by 1e-10 (float
rounding), so it holds no information. The chain's history flows back out of the hidden phases
into the field, and keeps doing so. Resetting the hidden phases instead leaves only the
field's own fading echo. This is the paper's "information backflow from hidden degrees of
freedom", measured with an exact rather than statistical match of the present.

## 5. No reset needed: saturation writes (`reset_free.py`)

Sections 2–4 start every trial from the same warmed state, which amounts to a reset. From
**random** initial phases, a weak write's direction depends on where the difference phase
started (prototype: the ± write separation flipped sign in 2 of 8 random starts). The fix is
the one magnetic recording uses: write hard enough to saturate. A strong carrier
injection-locks the addressed bead, erasing the old state. Because the pump locks the sum, a
π flip of the carrier leaves the difference unchanged. So the bit is a **quarter turn** of
carrier phase (0 vs π/2).

Prototype, 10 random starts, read 3000 steps after the write (R = 1 means every start ends in
the same state):

* A/T, carrier 0.03 alone, amplitude 1.0 × 1200 steps: R = 1.00 / 1.00 for the two bits, separation 2.70 rad.
* C/G, carrier 0.09 alone, same: R = 0.86 / 0.76, separation 2.46 rad.
* The C/G carrier scrambles A/T (R ≈ 0.1), while A/T only partly disturbs C/G. So the write order is C/G, then A/T.

Decoding from random initial phases (`runs/reset_free`, 96 trials, phase noise 0.003). Entries
are b1 (A/T) / b2 (C/G):

| write | delay 600 chain | delay 4000 chain | delay 600 bare | delay 4000 bare |
|---|---|---|---|---|
| weak (amp 0.2, BPSK, both at once) | 0.84 / 0.94 | 0.54 / 0.76 | 1.00 / 1.00 | 0.49 / 0.49 |
| strong (amp 1.0, quarter-turn, C/G then A/T) | **0.99** / 0.62 | **0.99** / 0.61 | 1.00 / 0.48 | 0.48 / 0.51 |

Without a reset, weak writes lose most of their memory, as the prototype predicted. A
saturation write restores it fully for the A/T register (0.99 at 4000 steps from any starting
state, bare 0.48). C/G overwrites less cleanly and is partly disturbed by the A/T write after it
(0.61). Reset-free operation is shown for one register of the two.

## 6. Capacity: position × frequency addressing (`phase_ram.py`)

Sub-gap carriers are evanescent (decay length 4–6 cells), so a source addresses only the beads
next to it. K blocks of `ACGT`, 24 cells apart, give 2K registers. Each trial writes 2K random
bits at once. Readout is local: bit (k, register) is decoded only from block k's probes, like a
RAM row select. Crosstalk is decoded from the neighbouring block's probes.

`runs/phase_ram`, 96 trials, phase noise 0.003. "Bits stored" is the information equivalent
Σ(1 − H₂(accuracy)) over all written bits:

| blocks | bits written | bits stored at 2000 steps (chain / bare) | at 6000 (chain / bare) | worst register at 6000 | neighbour-probe crosstalk at 6000 |
|---|---|---|---|---|---|
| 1 | 2 | 2.00 / 0.06 | 1.59 / 0.06 | 0.92 | — |
| 2 | 4 | 3.33 / 0.01 | 2.72 / 0.01 | 0.54 | 0.84 |
| 4 | 8 | 6.32 / 0.10 | 5.20 / 0.09 | 0.54 | 0.76 |

Capacity grows with chain length. Most registers read at 0.97–1.00, but the A/T register of
interior blocks is weak (0.54–0.88), and a block's bits can also be decoded from its neighbour's
probes (0.76–0.84). Position addressing is therefore partial: some leakage comes through
propagating pump sidebands or write spill-over. Wider block spacing, or reading at the letter
frequency only (which is evanescent), are the obvious fixes; neither is tested yet.

## 7. Experience decides what assembles (`experience_growth.py`)

A soup of 4 × 80 monomers is grown under a spatially uniform carrier (amplitude 0.002) or quiet.
`held_by` and ligation reward a monomer that hears its complement in phase, so the carrier
should select letters.

Short growth, 6000 steps, before bonding saturates (`runs/exp_growth_short`, 3 seeds). Entries
are bonded fraction per letter, min–max over seeds:

| arm | A | C | G | T | chains | longest |
|---|---|---|---|---|---|---|
| quiet | 0.05–0.06 | 0.03–0.19 | 0.00–0.12 | 0.00–0.01 | 3–11 | 2–5 |
| carrier 0.03 | 0.56–0.75 | 0.33–0.35 | **0.03–0.06** | 0.34–0.59 | 27–32 | 9–16 |
| carrier 0.09 | 0.57–0.72 | 0.78–0.81 | 0.60–0.66 | **0.04–0.16** | 34–40 | 15–23 |

Experience both triggers assembly and selects its composition. A 0.03 soup builds A/T(+C)
chains and leaves G out; a 0.09 soup builds C/G(+A) chains and leaves T out. That is the
material writing, from what it hears, the address map that section 2 later writes and reads
through. At 20 000 steps bonding saturates, but the C channel stays clean: C's `held_by` is
0.95–0.99 under 0.09 versus 0.10–0.81 quiet (3 seeds).

## 8. Corrections to overnight claims

* **Chaos**, re-measured with the wrapped phase distance and under input (`driven_stability.py`).
  The default grown soup diverges at +0.033 per step without input (overnight said +0.040 with the
  unwrapped distance). Under the same input stream it still diverges at +0.021 to +0.067 per step,
  for every tested input (hold-encoded bits and summed complement carriers, amplitudes 0.05 to 1;
  g_phase 0.005 gives +0.003 to +0.006). Scope: a point source, which sub-gap carriers cannot
  carry far, so most of the soup never hears it. Entrainment by a soup-wide drive is untested.
* "Tuning won't fix that; it would take a change of physics": withdrawn. The nonlinearity was
  there all along at large δθ, and the memory in the neutral mode.

## What this is and is not

* **Is:** a frequency- and position-addressed, phase-stored, non-fading memory that the bare
  medium does not have. It survives an exact match of the observable present, computes the XOR of
  two writes by phase addition, can be overwritten without a reset (A/T fully, C/G partly), and its
  address map is set by what the soup experienced while it grew.
* **Is not (yet):** a general reservoir computer, or a learner of input regularities beyond
  frequency. The storage is per bead pair: bonds are not needed to store, only to assemble and hold
  the pairs (all memory runs use pinned beads). The simulation is deterministic apart from phase
  noise, and probe noise is idealised.

## Lineage

ResonaattoriAivo (rhythm-addressed resonant memory: "the resonating weight is addressed by
rhythmic time") → here, frequency-addressed phase registers in the physics. Kaiku and Ristikko
(parity by interference with a homodyne reference) → XOR by phase addition with cosine emission.
Evening and WallMemoryLaw ("writers write the material, a ping reads it") → carriers write bead
phases, a passive or ping read recovers them. The Muruga et al. paper → the matched-present test.

## Next gates

1. Full loop: grow under carrier X, freeze, and show the grown structure stores more at address X
   than a soup grown under Y.
2. Reset-free C/G: per-register write tuning so both registers overwrite cleanly (R ≥ 0.95).
3. Feedback, the paper's other half: measure first, then choose the write phase from the
   measurement (the paper's double-measurement protocol), to recover weak writes without saturation.
4. A browser demo: the chain as a visible RAM, writing bits by playing tones and reading them back
   minutes later while the empty field forgets.

## Reproduce

```bash
python -m pytest -q tests
python experiments/phase_memory.py --trials 128 --amplitude 0.2 --phase-noise 0.003 --out runs/pm_mux_a02_n003
python experiments/phase_memory.py --trials 128 --amplitude 0.2 --phase-noise 0.01  --out runs/pm_mux_a02_n010
python experiments/phase_memory.py --mode addition --trials 128 --amplitude 0.3 --out runs/pm_add_a03_n003
python experiments/interaction_contrast.py runs/pm_mux_a02_n003
python experiments/matched_present.py --trials 128
python experiments/reset_free.py --trials 96
python experiments/phase_ram.py --blocks 1 2 4 --trials 96
python experiments/experience_growth.py --seeds 3 --grow-steps 6000 --amplitude 0.002 --arms quiet c0.03 c0.09 --out runs/exp_growth_short
python experiments/driven_stability.py
```
