"""A pumped wave field with oscillating monomers that can link into chains.

The field (psi) is train.html's update rule on an N x N periodic grid:
    psi' = (2-gamma) psi - (1-gamma) psi_prev + c2 * lap(psi) - m2(t) * psi + sources

Every monomer i has a position x_i, a phase theta_i and a natural frequency w_i (its
letter). Each step it
  1. writes  g_emit * cos(theta_i)  into the field around itself (a smooth kernel);
  2. hears the field there through a coupling c(t):
        receiver mode:  c(t) = cos(2*Omega*t)   (pumped ear, static medium)
        medium mode:    c(t) = 1                (static ear, pumped mass term)
  3. lets what it hears pull its phase (standard phase reduction of a driven oscillator):
        theta_i += w_i - g_phase * c(t) * psi(x_i) * sin(theta_i) / w_i + noise
  4. is pushed by it:  F_i = k_force * c(t) * cos(theta_i) * grad psi(x_i).

Why this gives pairing with no table: in receiver mode the time average of
    cos(2 Omega t) * cos(w_i t + phi_i) * cos(w_j t + phi_j)
is zero unless w_i + w_j = 2 Omega. So identical letters and unrelated letters feel no
average force, and only complements (frequencies summing to the pump) lock their phases
(phi_j = -phi_i) and attract. In medium mode, same-frequency monomers also couple
directly, which is the identity attraction train.html had to hard-code.

Backbone bonds are type-blind: two monomers with a free end, held at the backbone
spacing d0 for long enough, link with a small probability per check. A free monomer
drifting past rarely stays that long; two neighbours held side by side on a template do.
Each monomer has two ends, so chains are linear. Bonds are springs plus a weaker spring
to second neighbours that keeps chains fairly straight.
"""
import math
import numpy as np
import torch

from .config import Config


def _pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        N = cfg.N
        self.psi = torch.zeros(N, N, device=self.device)
        self.psi_prev = torch.zeros(N, N, device=self.device)
        self.t = 0
        self.letters = list(cfg.alphabet.keys())
        self.freqs = torch.tensor([cfg.alphabet[k] for k in self.letters], device=self.device)
        # kernel offsets: integer points within radius 2 of the nearest grid point
        o = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
        self.koff = torch.tensor(o, device=self.device, dtype=torch.long)    # (K, 2)
        self.P = 0
        self._alloc(0)
        self.events = []          # ligation log: (t, i, j)
        self.held = {}            # (i, j) -> consecutive checks spent at backbone spacing
        self.ligations = 0
        self.self_resp = self._calibrate_self()   # (letters, 2): own field = a*cos + b*sin

    # ------------------------------------------------------------------ self field
    @torch.no_grad()
    def _calibrate_self(self, steps=3000):
        """What a lone monomer of each letter hears from itself, as a*cos(theta)+b*sin(theta).
        Measured once on a small static field (no pump on the medium, no phase pull), so it is
        a property of the medium and the kernel, not a fitted parameter. Subtracting it removes
        the self-parametric sidebands the pumped ear would otherwise put on every monomer."""
        cfg = self.cfg
        n = 40
        dev = self.device
        L = len(self.letters)
        f = torch.zeros(L, n, n, device=dev)
        fp = torch.zeros_like(f)
        o = self.koff
        base = torch.tensor([20, 20], device=dev)
        px = (base[1] + o[:, 1]) % n
        py = (base[0] + o[:, 0]) % n
        d2 = (o[:, 0].float() ** 2 + o[:, 1].float() ** 2)
        w = torch.exp(-d2 / (2 * 0.8 ** 2)) * (d2 <= 2.2 ** 2)
        w = w / w.sum()
        th = torch.zeros(L, device=dev)
        X, Y = [], []
        for t in range(steps):
            hear = (f[:, py, px] * w).sum(1)
            q = torch.cos(th)
            if t > steps // 2:
                X.append(torch.stack([torch.cos(th), torch.sin(th)], 1))
                Y.append(hear)
            th = th + self.freqs
            lap = (torch.roll(f, 1, 1) + torch.roll(f, -1, 1) + torch.roll(f, 1, 2) + torch.roll(f, -1, 2) - 4 * f)
            new = (2 - cfg.gamma) * f - (1 - cfg.gamma) * fp + cfg.c2 * lap - cfg.m2 * f
            new[:, py, px] += cfg.g_emit * q[:, None] * w[None]
            fp, f = f, new
        X = torch.stack(X, 1)            # (L, T, 2)
        Y = torch.stack(Y, 1)            # (L, T)
        sol = torch.linalg.lstsq(X, Y[..., None]).solution[..., 0]   # (L, 2)
        return sol

    # ------------------------------------------------------------------ monomers
    def _alloc(self, n):
        d = self.device
        self.pos = torch.zeros(n, 2, device=d)
        self.theta = torch.zeros(n, device=d)
        self.typ = torch.zeros(n, dtype=torch.long, device=d)
        self.nb = torch.full((n, 2), -1, dtype=torch.long, device=d)
        self.held_by = torch.zeros(n, device=d)
        self.pinned = torch.zeros(n, dtype=torch.bool, device=d)
        self.field_force = torch.zeros(n, 2, device=d)
        self.P = n

    def add_monomers(self, types, pos=None, theta=None):
        """types: list/array of letter indices. Returns the new indices."""
        types = torch.as_tensor(np.asarray(types), dtype=torch.long, device=self.device)
        n = types.numel()
        N = self.cfg.N
        if pos is None:
            pos = torch.rand(n, 2, generator=self.gen, device=self.device) * N
        else:
            pos = torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=self.device)
        if theta is None:
            theta = torch.rand(n, generator=self.gen, device=self.device) * 2 * math.pi
        start = self.P
        self.pos = torch.cat([self.pos, pos])
        self.theta = torch.cat([self.theta, theta])
        self.typ = torch.cat([self.typ, types])
        self.nb = torch.cat([self.nb, torch.full((n, 2), -1, dtype=torch.long, device=self.device)])
        self.held_by = torch.cat([self.held_by, torch.zeros(n, device=self.device)])
        self.pinned = torch.cat([self.pinned, torch.zeros(n, dtype=torch.bool, device=self.device)])
        self.field_force = torch.cat([self.field_force, torch.zeros(n, 2, device=self.device)])
        self.P += n
        return torch.arange(start, self.P, device=self.device)

    def add_chain(self, seq, start, direction=(1.0, 0.0)):
        """Place a pre-bonded straight chain, e.g. a template. seq is a string of letters."""
        idx = [self.letters.index(ch) for ch in seq]
        dx, dy = direction
        nrm = math.hypot(dx, dy)
        dx, dy = dx / nrm, dy / nrm
        pos = [(start[0] + k * self.cfg.d0 * dx, start[1] + k * self.cfg.d0 * dy) for k in range(len(seq))]
        ids = self.add_monomers(idx, pos=pos).tolist()
        for a, b in zip(ids[:-1], ids[1:]):
            self._bond(a, b)
        return ids

    def _bond(self, i, j):
        for a, b in ((i, j), (j, i)):
            slot = 0 if int(self.nb[a, 0]) < 0 else 1
            self.nb[a, slot] = b

    # ------------------------------------------------------------------ kernel
    def _kernel(self):
        """Indices (P,K) into the flattened field and normalised weights (P,K)."""
        N = self.cfg.N
        base = torch.round(self.pos).long()                                   # (P,2) x,y
        px = base[:, None, 0] + self.koff[None, :, 1]
        py = base[:, None, 1] + self.koff[None, :, 0]
        ddx = px.float() - self.pos[:, None, 0]
        ddy = py.float() - self.pos[:, None, 1]
        d2 = ddx * ddx + ddy * ddy
        w = torch.exp(-d2 / (2 * 0.8 ** 2)) * (d2 <= 2.2 ** 2)
        w = w / w.sum(1, keepdim=True)
        idx = (py % N) * N + (px % N)
        return idx, w

    # ------------------------------------------------------------------ dynamics
    def coupling(self, t=None):
        cfg = self.cfg
        t = self.t if t is None else t
        if not cfg.pump_on:
            return 0.0 if cfg.mode == "receiver" else 1.0
        return math.cos(2 * cfg.Omega * t) if cfg.mode == "receiver" else 1.0

    def m2_now(self):
        cfg = self.cfg
        if cfg.mode == "medium" and cfg.pump_on:
            return cfg.m2 * (1 + cfg.h_medium * math.cos(2 * cfg.Omega * self.t))
        return cfg.m2

    @torch.no_grad()
    def step(self, external_source=None):
        cfg = self.cfg
        N = cfg.N
        psi = self.psi
        c = self.coupling()
        P = self.P
        if P:
            idx, w = self._kernel()
            gx = (torch.roll(psi, -1, 1) - torch.roll(psi, 1, 1)) * 0.5
            gy = (torch.roll(psi, -1, 0) - torch.roll(psi, 1, 0)) * 0.5
            flat = psi.flatten()
            hear = (flat[idx] * w).sum(1)
            if not cfg.hear_self:
                sr = self.self_resp[self.typ]
                hear = hear - (sr[:, 0] * torch.cos(self.theta) + sr[:, 1] * torch.sin(self.theta))
            hx = (gx.flatten()[idx] * w).sum(1)
            hy = (gy.flatten()[idx] * w).sum(1)
            om = self.freqs[self.typ]
            q = torch.cos(self.theta)
            # how strongly this monomer is held: running average of the pumped drive it hears
            # in phase with its own oscillation (the same product that makes the pairing force)
            a = 1.0 / cfg.held_tau
            self.held_by = (1 - a) * self.held_by + a * (c * hear * q)

            # ---- phase: driven-oscillator phase reduction
            noise = torch.randn(P, generator=self.gen, device=self.device) * cfg.phase_noise
            self.theta = self.theta + om - cfg.g_phase * c * hear * torch.sin(self.theta) / om + noise
            self.theta = torch.remainder(self.theta, 2 * math.pi)

            # ---- forces
            Ff = cfg.k_force * c * q[:, None] * torch.stack([hx, hy], 1)
            self.field_force = Ff
            F = Ff + self._pair_forces() + self._bond_forces()
            kick = torch.randn(P, 2, generator=self.gen, device=self.device) * math.sqrt(2 * cfg.mobility * cfg.kT)
            move = (cfg.mobility * F + kick) * (~self.pinned)[:, None]
            self.pos = torch.remainder(self.pos + move, N)

            # ---- emission (with the phase used this step)
            src = torch.zeros(N * N, device=self.device)
            gain = cfg.g_emit * (1 + cfg.chain_boost * (self.nb >= 0).sum(1).float() / 2)
            src.index_add_(0, idx.flatten(), (gain[:, None] * q[:, None] * w).flatten())
            src = src.view(N, N)
        else:
            src = 0.0

        lap = (torch.roll(psi, 1, 0) + torch.roll(psi, -1, 0) +
               torch.roll(psi, 1, 1) + torch.roll(psi, -1, 1) - 4 * psi)
        new = (2 - cfg.gamma) * psi - (1 - cfg.gamma) * self.psi_prev + cfg.c2 * lap - self.m2_now() * psi + src
        # Optional experimental forcing: supplied in field-source units per step.
        # The ordinary simulator takes the identical code path when it is absent.
        if external_source is not None:
            ext = torch.as_tensor(external_source, device=self.device, dtype=psi.dtype)
            if ext.shape != psi.shape:
                raise ValueError(f"external_source must have shape {tuple(psi.shape)}")
            new = new + ext
        self.psi_prev, self.psi = psi, new
        self.t += 1
        if P and self.t % cfg.lig_every == 0:
            self._ligate()

    def _delta(self, a, b):
        """Minimum-image displacement b - a on the torus."""
        N = self.cfg.N
        d = b - a
        return d - N * torch.round(d / N)

    def _pair_forces(self, chunk=4096):
        cfg = self.cfg
        F = torch.zeros_like(self.pos)
        P = self.P
        for s in range(0, P, chunk):
            d = self._delta(self.pos[s:s + chunk, None, :], self.pos[None, :, :])   # (c,P,2) i->j
            r = d.norm(dim=-1)
            over = (r < cfg.r_core) & (r > 1e-6)
            mag = torch.where(over, cfg.k_core * (cfg.r_core - r), torch.zeros_like(r))
            F[s:s + chunk] -= (mag[..., None] * d / r.clamp(min=1e-6)[..., None]).sum(1)
        return F

    def _bond_forces(self):
        cfg = self.cfg
        F = torch.zeros_like(self.pos)
        has = self.nb >= 0
        if not bool(has.any()):
            return F
        i = torch.arange(self.P, device=self.device)[:, None].expand(-1, 2)[has]
        j = self.nb[has]
        d = self._delta(self.pos[i], self.pos[j])
        r = d.norm(dim=-1).clamp(min=1e-6)
        f = (cfg.k_bond * (r - cfg.d0) / r)[:, None] * d        # pulls i toward j if stretched
        F.index_add_(0, i, f)
        # bending: monomers with two neighbours keep those two at 2*d0
        both = (self.nb >= 0).all(1)
        if bool(both.any()):
            a = self.nb[both, 0]
            b = self.nb[both, 1]
            d2 = self._delta(self.pos[a], self.pos[b])
            r2 = d2.norm(dim=-1).clamp(min=1e-6)
            f2 = (cfg.k_bend * (r2 - 2 * cfg.d0) / r2)[:, None] * d2
            F.index_add_(0, a, f2)
            F.index_add_(0, b, -f2)
        return F

    # ------------------------------------------------------------------ ligation
    def chain_ids(self):
        """Connected components of the backbone graph (numpy, cheap for P in the thousands)."""
        nb = self.nb.cpu().numpy()
        P = self.P
        parent = np.arange(P)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(P):
            for j in nb[i]:
                if j >= 0:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        return np.array([find(i) for i in range(P)])

    def _ligate(self):
        """Type-blind ligation. Each pair of free ends keeps a leaky score: +1 for every check
        spent at the backbone spacing d0 (within lig_window), times lig_decay per check. Past
        lig_score they link with probability p_lig per check. Only monomers that are being HELD
        take part: their running average of the pumped drive they hear in phase with their own
        oscillation (held_by) must exceed held_min. A free monomer averages near zero; one locked
        to a complementary partner does not. A random encounter scores a few
        points and drifts off; neighbours held on a template keep returning to the spacing.
        Letters are never looked at."""
        cfg = self.cfg
        free = (self.nb < 0).any(1) & (self.held_by > cfg.held_min) & (~self.pinned)
        fi = torch.nonzero(free).flatten()
        if fi.numel() < 2:
            self.held = {}
            return
        d = self._delta(self.pos[fi][:, None, :], self.pos[fi][None, :, :])
        r = d.norm(dim=-1)
        ok = torch.triu((r - cfg.d0).abs() < cfg.lig_window, diagonal=1)
        cand = torch.nonzero(ok).cpu().numpy()
        fi_np = fi.cpu().numpy()
        held = {k: v * cfg.lig_decay for k, v in self.held.items() if v * cfg.lig_decay > 0.5}
        now = set()
        for a, b in cand:
            key = (int(fi_np[a]), int(fi_np[b]))
            held[key] = held.get(key, 0.0) + 1.0
            now.add(key)
        self.held = held
        # only pairs that are at the backbone spacing AND both held right now may link
        # (the score remembers the past, but a bond is made in the present)
        ready = [k for k in now if held[k] >= cfg.lig_score]
        if not ready:
            return
        cid = self.chain_ids()
        used = set()
        for i, j in ready:
            if self.rng.random() >= cfg.p_lig:
                continue
            if i in used or j in used or cid[i] == cid[j]:
                continue
            if int((self.nb[i] >= 0).sum()) >= 2 or int((self.nb[j] >= 0).sum()) >= 2:
                continue
            self._bond(i, j)
            used.update((i, j))
            cid[cid == cid[i]] = cid[j]
            self.events.append((self.t, i, j))
            self.ligations += 1
            self.held.pop((i, j), None)

    # ------------------------------------------------------------------ reading
    def chains(self, min_len=2):
        """List of chains, each a list of monomer indices in backbone order."""
        nb = self.nb.cpu().numpy()
        seen = np.zeros(self.P, bool)
        out = []
        for s in range(self.P):
            if seen[s]:
                continue
            deg = (nb[s] >= 0).sum()
            if deg == 2:
                continue          # start from ends only
            if deg == 0:
                seen[s] = True
                continue
            path = [s]
            seen[s] = True
            prev, cur = -1, s
            while True:
                nxt = [j for j in nb[cur] if j >= 0 and j != prev]
                if not nxt or seen[nxt[0]]:
                    break
                prev, cur = cur, int(nxt[0])
                path.append(cur)
                seen[cur] = True
            if len(path) >= min_len:
                out.append(path)
        return out

    def sequence(self, path):
        typ = self.typ.cpu().numpy()
        return "".join(self.letters[typ[i]] for i in path)
