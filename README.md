# Chains in a pumped GPU field

![pic](pic.png)

Oscillating monomers in a wave field, a pump at frequency 2Ω, and a single question:
**can complementary pairing, and with it copying, come out of wave physics instead of a rule table?**

This grew out of `train.html` (scouts writing into a Klein–Gordon field and walking on their
own wake, like Couder's walking droplets) and the Emergence repos (tokens as genes). DNA needs
three things the trains did not have: positions on a backbone, complementary pairing, and a
sequence-to-shape map. This repo goes after the first two.

*RESULTS: filled in below from the runs in `results/`.*

## The physics in one paragraph

The field is `train.html`'s update rule on a periodic N×N grid, with a small damping:

```
psi' = (2-γ)·psi − (1-γ)·psi_prev + c²·∇²psi − m²·psi + sources        (c² = 0.3, m² = 0.02, γ = 0.01)
```

The mass term opens a gap at ω₀ = arccos(0.99) ≈ 0.1415 rad/step. Every monomer oscillates
below the gap, so its field is a screened cloud a few cells wide rather than a travelling wave.

A monomer has a position, a phase θ and a natural frequency ω (its letter):

| letter | ω (rad/step) |
|---|---|
| A | 0.03 |
| C | 0.05 |
| G | 0.09 |
| T | 0.11 |

Each step every monomer

1. writes `g_emit · cos θ` into the field around itself;
2. hears the field there through a coupling that pulses with the pump, `c(t) = cos 2Ωt`;
3. lets what it hears pull its phase (the standard phase reduction of a driven oscillator);
4. is pushed by it: `F = k_force · c(t) · cos θ · ∇psi`.

Multiplying by `cos 2Ωt` turns a field oscillating at ω into one at 2Ω − ω with its phase
flipped. So the time-averaged force between two monomers is proportional to

```
⟨ cos 2Ωt · cos(ω_i t + φ_i) · cos(ω_j t + φ_j) ⟩
```

which is zero **unless ω_i + ω_j = 2Ω**. With 2Ω = 0.14 that picks out A+T and C+G and nothing
else. Identical letters average to zero; unrelated letters average to zero. Complements lock
their phases to φ_j = −φ_i (phase conjugation) and attract.

No table anywhere says A binds T. Move the pump and the pairing moves with it
(`Omega = 0.076` makes no pair sum to the pump at all, and that is one of the controls).

### Two ways to pump

- **receiver** (default): the monomers' "ears" pulse at 2Ω; the medium is static.
- **medium**: the field's mass term pulses (a Faraday medium, like the vibrating bath under
  walking droplets) and the ears are static. Here identical letters also couple directly at
  their shared frequency, which is the "attract your own kind" rule `train.html` had to
  hard-code. This variant is kept as a comparison.

Honest note on the model: emission is static and hearing is pumped, so the interaction is
non-reciprocal. These are active particles (each one keeps oscillating on its own), so that is
allowed, but it is a choice. Phase sensitivity (`g_phase`) and mechanical push (`k_force`) are
separate knobs for the same reason.

### Backbone bonds: type-blind

Two monomers with a free end that sit at the backbone spacing d₀ = 3 (±0.4) link with a small
probability per check. That rule never looks at letters. Each monomer has two ends, so chains
are linear. A free monomer drifting past rarely stays at d₀ long enough; two neighbours held
side by side on a template do. Pair partners sit closer (about 2 cells, the soft core), so a
pair is not mistaken for a backbone.

## Run it

```sh
pip install -r requirements.txt
python run.py                       # live viewer: GATTACAG template in a soup of monomers
python run.py --no-template         # soup only
python run.py --N 256 --n_each 600  # bigger world (GPU)
python experiments/pair_table.py    # which letters attract which
python experiments/gate.py --round2 # the templating gate with controls
```

Keys in the viewer: `SPACE` pause, `P` pump on/off, `F` field on/off, `+/-` speed, `S` screenshot,
left-drag stirs. Colours: A red, C yellow, G green, T blue; white lines are backbone bonds;
the template is ringed.
