"""Controlled external field drive and reproducible spatial/oscillator probes.

External drive enters the same field update as monomer sources, but is deliberately
independent of the oscillator alphabet, pair rules and bond dynamics. No readout
is trained here; these are measurement tools, not a computation benchmark.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class Stimulus:
    """One localized input; time is relative to the beginning of a recording.

    An impulse acts once at start. Square and sine last duration steps.
    Samples are arbitrary dimensionless amplitudes (one per step); amplitude
    scales all four waveforms. omega for sine is in radians per simulation step.
    """

    position: tuple[float, float]
    kind: str = "sine"
    amplitude: float = 0.05
    start: int = 0
    duration: int = 400
    omega: float = 0.11
    width: float = 1.0
    samples: tuple[float, ...] = ()

    def __post_init__(self):
        if self.kind not in ("impulse", "square", "sine", "samples"):
            raise ValueError("kind must be impulse, square, sine or samples")
        if self.start < 0 or self.duration < 1:
            raise ValueError("start must be >= 0 and duration must be >= 1")
        if len(self.position) != 2 or not all(math.isfinite(v) for v in self.position):
            raise ValueError("position must contain two finite coordinates")
        if not all(math.isfinite(v) for v in (self.amplitude, self.omega, self.width)):
            raise ValueError("amplitude, omega and width must be finite")
        if self.width < 0 or self.omega < 0:
            raise ValueError("width and omega must be nonnegative")
        if self.kind == "samples" and len(self.samples) != self.duration:
            raise ValueError("samples must have exactly duration entries")
        if not all(math.isfinite(v) for v in self.samples):
            raise ValueError("samples must be finite")

    def value(self, step: int) -> float:
        offset = step - self.start
        if offset < 0 or offset >= self.duration:
            return 0.0
        if self.kind == "impulse":
            return self.amplitude if offset == 0 else 0.0
        if self.kind == "square":
            return self.amplitude
        if self.kind == "samples":
            return self.amplitude * self.samples[offset]
        return self.amplitude * math.cos(self.omega * offset)


def source_kernel(world, position, width=1.0):
    """Unit-sum, periodic point/Gaussian source at fractional (x, y)."""
    N = world.cfg.N
    if width < 0:
        raise ValueError("width must be nonnegative")
    if width == 0:
        src = torch.zeros_like(world.psi)
        x, y = position
        src[round(y) % N, round(x) % N] = 1.0
        return src
    a = torch.arange(N, device=world.device, dtype=world.psi.dtype)
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    x, y = position
    dx = torch.remainder(xx - x + N / 2, N) - N / 2
    dy = torch.remainder(yy - y + N / 2, N) - N / 2
    src = torch.exp(-(dx * dx + dy * dy) / (2 * width * width))
    total = src.sum()
    if not bool(total > 0):
        raise ValueError("source width is too narrow for the chosen fractional position")
    return src / total


class FieldProbes:
    """Bilinear periodic field sensors, optionally with oscillator-phase sensors."""

    def __init__(self, world, positions, monomer_ids=()):
        if not positions:
            raise ValueError("at least one field probe is required")
        self.positions = [tuple(map(float, p)) for p in positions]
        self.monomer_ids = list(monomer_ids)
        if any(i < 0 or i >= world.P for i in self.monomer_ids):
            raise ValueError("monomer sensor index out of range")
        N = world.cfg.N
        xy = torch.as_tensor(self.positions, dtype=world.psi.dtype, device=world.device)
        x, y = xy[:, 0], xy[:, 1]
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        self.x0 = x0 % N
        self.x1 = (x0 + 1) % N
        self.y0 = y0 % N
        self.y1 = (y0 + 1) % N
        self.wx = x - x0
        self.wy = y - y0

    def field(self, world):
        f = world.psi
        lower = f[self.y0, self.x0] * (1 - self.wx) + f[self.y0, self.x1] * self.wx
        upper = f[self.y1, self.x0] * (1 - self.wx) + f[self.y1, self.x1] * self.wx
        return lower * (1 - self.wy) + upper * self.wy

    def phases(self, world):
        """[cos(theta), sin(theta)] for each selected monomer (can be empty)."""
        ids = self.monomer_ids
        theta = world.theta[ids]
        return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)


def record_response(world, stimulus, probe_positions, steps, warmup=0, sample_every=1,
                    monomer_ids=()):
    """Apply a stimulus and record observables; warmup contains no external drive.

    stimulus=None is an undriven run. Use a separately constructed identical
    world with the same seed to obtain a counterfactual driven-minus-undriven
    response. Returned time indices are relative to the end of warmup.
    """
    if steps < 1 or warmup < 0 or sample_every < 1:
        raise ValueError("steps/sample_every must be positive, warmup nonnegative")
    probes = FieldProbes(world, probe_positions, monomer_ids)
    kernel = (source_kernel(world, stimulus.position, stimulus.width)
              if stimulus is not None else None)
    for _ in range(warmup):
        world.step()
    times, values, fields, phases = [], [], [], []
    for t in range(steps):
        val = stimulus.value(t) if stimulus is not None else 0.0
        world.step(external_source=kernel * val if val else None)
        if (t + 1) % sample_every == 0 or t + 1 == steps:
            times.append(t + 1)
            values.append(val)
            fields.append(probes.field(world).detach().cpu().numpy())
            if monomer_ids:
                phases.append(probes.phases(world).detach().cpu().numpy())
    result = {"step": np.asarray(times, dtype=int),
              "drive": np.asarray(values, dtype=float),
              "field": np.stack(fields)}
    if monomer_ids:
        result["phase_xy"] = np.stack(phases)
    return result
