"""Every law of the world in one place.

A monomer's only identity is its natural oscillation frequency. Nothing in the code
says which letter binds which: pairing comes from how the field is heard (see world.py).
"""
from dataclasses import dataclass, asdict, field
import json

# Letters and their natural frequencies (rad/step). With pump 2*Omega = 0.14 the
# pairs whose frequencies sum to 0.14 are A+T and C+G. That is the only place the
# "rule" lives: in arithmetic between frequencies and the pump, not in a table.
DEFAULT_ALPHABET = {"A": 0.03, "C": 0.05, "G": 0.09, "T": 0.11}


@dataclass
class Config:
    # ---- world ---------------------------------------------------------------
    N: int = 128                  # field grid is N x N, periodic
    seed: int = 0
    device: str = "auto"

    # ---- field: train.html's Klein-Gordon update, plus a little damping --------
    c2: float = 0.3               # wave speed squared (cells^2 / step^2)
    m2: float = 0.02              # mass term -> gap at ~0.1415 rad/step
    gamma: float = 0.01           # field damping per step

    # ---- the pump --------------------------------------------------------------
    Omega: float = 0.07           # pump at 2*Omega = 0.14 (just below the gap)
    mode: str = "receiver"        # "receiver": monomers hear the field through a
                                  #   coupling that pulses at 2*Omega
                                  # "medium": the field's mass term pulses instead
                                  #   (train.html-style Faraday medium), coupling static
    h_medium: float = 0.3         # pump depth in "medium" mode
    pump_on: bool = True

    # ---- monomers ----------------------------------------------------------------
    alphabet: dict = field(default_factory=lambda: dict(DEFAULT_ALPHABET))
    g_emit: float = 0.10          # how strongly a monomer writes its oscillation
    chain_boost: float = 2.0      # bonded monomers write louder: g_emit*(1+chain_boost*bonds/2).
                                  # Type-blind; it only asks whether a monomer is in a chain.
    g_phase: float = 0.02         # how strongly what it hears pulls its phase
    k_force: float = 1.0          # how strongly what it hears pushes it
    phase_noise: float = 0.01     # phase diffusion per step
    hear_self: bool = False       # False: a monomer does not hear its own voice (its own
                                  # steady field, measured once per letter, is subtracted)

    # ---- motion (overdamped) ---------------------------------------------------------
    mobility: float = 1.0
    kT: float = 0.006             # thermal kicks: sqrt(2 * mobility * kT) per step
    r_core: float = 2.0           # soft hard-core radius
    k_core: float = 0.05

    # ---- backbone bonds (type-blind, two per monomer: a chain has two ends) ----------
    d0: float = 3.6               # backbone spacing (well clear of the ~2.2 pairing distance)
    k_bond: float = 0.05
    k_bend: float = 0.02          # spring to second neighbours keeps chains straight
    held_tau: float = 100.0       # steps over which "being held" is averaged
    held_min: float = 0.05         # both monomers must be held at least this much to link
    lig_window: float = 0.3       # two free ends at |r - d0| < window score +1 per check;
    lig_decay: float = 0.97       # the score decays by this factor every check (memory ~330 steps)
    lig_score: float = 6.0        # once the score passes this they may link ...
    p_lig: float = 0.1            # ... with this probability per check.
                                  # A passer-by scores a few points and leaves; neighbours held
                                  # on a template keep coming back to the backbone spacing.
    lig_every: int = 10           # steps between ligation checks

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d):
        c = Config()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
