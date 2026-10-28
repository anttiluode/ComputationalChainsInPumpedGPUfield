# Overnight results, 28–29 Oct 2026

Run by Claude (Opus 5.5) in a cloud CPU workspace, taking over the line Astra set up
(`bond_ablation.py`, `temporal_memory.py`). All 21 existing tests pass. Every number
below comes from a file under `runs/`; commands to reproduce are at the end.

## Verdict in one paragraph

In this physics, chains and soups of pumped oscillators do **not** compute anything the
empty wave field does not already do better. The bare field has more linear memory than
any chain world; no configuration produces nonlinear (XOR) capacity above a significance
gate; a grown soup with the default coupling is **chaotic** (a 10⁻⁶ rad phase kick grows
~0.04 per step) and therefore has no memory at all. Bonds do have a reproducible effect
(+6% linear memory over cut chains, 6/6 paired runs), but only by holding the scatterers
still: a pinned chain does better than either. One genuine physical phenomenon turned up —
sequence-specific coherent relay of a sub-gap carrier through A/T chains (33× the bare
field's far amplitude) — but it carries *less* decodable information than the bare field,
because the relay is the chains' own oscillation with its phase nudged.

## Ledger

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | `temporal_memory.py` as written measures memory | **Wrong, fixed** | Delays 0–4 lie outside the light cone (probes 33–43 cells away, max wave speed √0.3 ≈ 0.55 cells/step → ≥60 steps = 5 symbols). Every condition scores ≤0 there, bare included. `runs/temporal_memory/` |
| 2 | Chains add memory over the bare field | **No** | Lags 0–20, 3 seeds × 2 sequences: bare 12.81 (12.49–13.25), bonded 10.29 (9.62–10.91), cut 9.73 (8.99–10.32). Same order under probe noise σ=1e-3: 8.54 / 7.20 / 6.79. `runs/mc_hold_3seeds/` |
| 3 | Bonds change computation | **Yes, small, mechanical** | bonded − cut = +0.37…+0.65 in 6/6 paired runs. Cut beads drift 1.4–1.8 cells per run, bonded 0.4–0.8; the pinned chain (no drift) scored 14.70 vs bonded 13.78, cut 13.24 (seed 0, lags 0–30). Bonds help by reducing drift. |
| 4 | Chains compute nonlinear functions (XOR of bit pairs) | **No** | Gated XOR capacity ≤0.011 mean for every condition; the bare field (exactly linear, true value 0) is the null. Positive control: an XOR feature planted at 1% of feature scale is recovered at R²=1.000. |
| 5 | Stronger input pushes chains into a nonlinear regime | **No** | Amplitude 0.2 / 0.6 / 2.0: XOR 0.000 everywhere; frozen chains converge toward bare (11.3 → 11.8 → 12.1 vs 12.49) as the input outshouts their emission. `runs/mc_amp*` |
| 6 | Chains relay sub-gap signals the medium blocks | **Physically yes, informationally no** | Steady carrier ω=0.11 (pump-complement of A): far lock-in ATATATAT 2.05e-3, AAAATTTT 1.23e-3, AAAAAAAA 4.4e-4, ACGTACGT 1.1e-4, CG chains ≈ bare 6.3e-5. Above the gap (ω≥0.16) all chains = bare. But BPSK bits at the same carrier, same demodulating readout: bare 100% / 97% sign accuracy (σ=0 / 1e-3) vs chains 85–93% / 76–90%. `runs/tx_*`, `runs/wire011_*` |
| 7 | A soup grown under structured input becomes a better computer than one grown quiet | **No — both are opaque** | 320 monomers, 20k growth steps, 2 seeds × 2 arms (21–38 chains, longest 15–37): grown, cut and shuffled controls all ≈0 (max 1.16) vs bare 12.49–12.69. `runs/grow_set_compute/` |
| 8 | The soup is opaque because input is buried under its emission | **No — it is scrambled (chaos)** | 10⁻⁶ rad kick on one bead grows at +0.040/step to O(1); the oracle response (driven − undriven twin) also has zero memory. `runs/soup_diagnosis/` |
| 9 | Turning coupling down finds an edge-of-chaos window with XOR | **No** | See sweep below. Chaos at default coupling; stable below g_phase ≈ 0.003; linear memory returns to the bare level, never above; XOR is zero under the shifted-bits null (raw p = 0.05–0.07, same as the linear g = 0 control). `runs/soup_sweep*` |

## The coupling sweep (one frozen grown soup, g_phase varied)

Bare-field reference for this benchmark (600 symbols, lags 0–20): 12.04 — at g_phase = 0 the
frozen soup does not scatter the field, so its oracle response *is* the bare response.

| g_phase | growth of a 10⁻⁶ rad kick | raw linear memory | oracle-response linear memory | XOR |
|---|---|---|---|---|
| 0.02 (default) | +0.043 /step | 0.00 | 0.00 | 0 |
| 0.005 | +0.0051 /step | 0.00 | 0.00 | 0 |
| 0.003 | +0.0015 (seed 0) / +0.0012 (seed 1) | 0.06 / 1.62 | 0.07 / 7.10 | 0 |
| 0.002 | +0.0013 | 0.46 | 8.88 | 0 |
| 0.001 | +0.0015 | 2.10 | 11.66 | 0 |
| 0.0005 | +0.0009 (seed 0) / +0.0006 (seed 1) | 4.52 / 7.81 | 11.92 / 12.84 | 0 (raw p = 0.07) |
| 0.0001 | +0.0001 | 7.49 | 12.04 | 0 (raw p = 0.05) |
| 0 (control) | 0 | 9.74 | 12.04 | 0 (raw p = 0.07, linear by construction) |

Reading: the default soup sits deep in chaos. Below g_phase ≈ 0.003 it stops scrambling and
the input information comes back (oracle column) to the bare level — never above it. The
physical (raw) readout stays worse than bare at every coupling, because the soup's own
oscillation is interference. XOR is zero everywhere under the shifted-bits null; the raw
p-values of the stable soups match the linear g = 0 control. (Growth rates are a crude
log-linear fit; the kicked separation's final size, in `summary.json`, tells the same story.)

`g_phase = 0` is a control: every oscillator becomes a fixed clock, so the medium is
exactly linear in the input. Its *ungated* raw XOR read 0.206 in the first sweep
(`runs/soup_sweep_v1_ungated/`), which is how the estimator artifact below was caught.

## Estimator lesson (applies to every capacity number in this repo)

`MC = Σ_d max(R²_d, 0)` over 21 lags turns noise centred near zero into a positive
capacity. In the stable soup it produced "XOR capacity" 0.45–0.54 built entirely from
lags with R² 0.01–0.08 — including lags 0–3, which are physically impossible. The
gated versions `MC_lin_sig`, `MC_nl_sig` in `memory_capacity.py` count a lag only if its
R² beats the 99th percentile of R² for targets independent of the input (200 draws).
Linear capacities are unchanged by the gate.

That gate was still not enough in the stable soups: the 21 XOR targets are shifted copies of one
sequence z(t) = u(t)·u(t−1), so a chance alignment between the soup's deterministic clocks and z
is counted many times, and an i.i.d. null underestimates the tail. Gated raw XOR read 0.44–0.46
there and 0.086 in the linear g = 0 control. The honest test is `shift_null_pvalue`: keep the
target exactly, circularly shift the bits by a large random offset, and compare. Against it the
stable soups give p = 0.05–0.07 — the same as the linear control (p = 0.07) — while the linear
task gives p = 0 (observed 9.74 vs null mean 0.02), so the test does detect real capacity.
**Any future capacity claim in this repo should quote the shifted-bits p-value.**

## Why the oscillators cannot compute here (mechanism)

* The field is linear and the readout is linear, so any nonlinearity must come from the
  beads. The input reaches a bead only through its **phase**, and a bead's emission
  `cos θ` does not depend on how hard it is driven.
* Weak phase coupling: perturbations stay small, `cos(θ₀+δθ) ≈ cos θ₀ − δθ sin θ₀` is
  linear in δθ, δθ is linear in the input → no XOR, and the beads' own large oscillation
  only adds interference for the readout.
* Strong phase coupling: many coupled phase oscillators go chaotic → no fading memory.
* Nothing in between saturates (no tanh-like element in the input path), so there is no
  regime with both fading memory and nonlinearity. Tuning won't fix that; it would take
  a change of physics, e.g. emission amplitude that saturates with drive, or detuned
  (Adler) locking where the phase sits in a stable, sinusoidally curved well. Untested
  hypotheses, listed so the next attempt targets the actual missing ingredient.

## What is reusable

The instruments are now honest and general, and can be pointed at any field simulation
(Emergence included): light-cone-aware memory curves with probe noise
(`memory_capacity.py`, `analyze_memory.py`), significance-gated capacity, the
planted-feature positive control, counterfactual twins, the chaos test
(`soup_diagnosis.py`), and the transmission / BPSK pair for "does this structure carry
a signal the medium can't".

## Reproduce

```bash
python -m pytest -q tests                                                      # 21 pass
python experiments/memory_capacity.py --sequences ACGTACGT,AGCTAGCT --seeds 3 --symbols 900 --max-lag 20 --out runs/mc_hold_3seeds
python experiments/analyze_memory.py runs/mc_hold_3seeds --sigmas 0 1e-3
python experiments/memory_capacity.py --freeze --out runs/mc_hold_frozen
python experiments/memory_capacity.py --amplitude 2.0 --symbols 900 --max-lag 20 --out runs/mc_amp2.0
python experiments/transmission.py --out runs/tx
python experiments/wire_bpsk.py --carrier 0.11 --out runs/wire011
python experiments/grow_set_compute.py --seeds 2 --n-each 80 --grow-steps 20000
python experiments/soup_diagnosis.py
python experiments/soup_sweep.py --gphases 0.003 0.002 0.001 0.0005 0.0001 0
```

Large raw traces (`*.npy`, `features.npz`) are git-ignored; every `summary.json` is kept.
