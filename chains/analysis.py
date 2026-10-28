"""Reading results. The complement map here is used only for SCORING, and it is derived
from the physics (frequencies that sum to the pump), never fed back into the dynamics."""
import numpy as np
import torch


def complement_map(cfg, tol=1e-6):
    """letter -> letter whose frequency sums with it to 2*Omega (None if no such letter)."""
    out = {}
    for a, wa in cfg.alphabet.items():
        out[a] = None
        for b, wb in cfg.alphabet.items():
            if abs(wa + wb - 2 * cfg.Omega) < tol:
                out[a] = b
    return out


def nearest(world, idx, among, max_r):
    """For each monomer in idx, the nearest monomer in `among` within max_r (or -1) and distance."""
    if len(idx) == 0 or len(among) == 0:
        return np.full(len(idx), -1), np.full(len(idx), np.inf)
    a = world.pos[torch.as_tensor(idx, device=world.device)]
    b = world.pos[torch.as_tensor(among, device=world.device)]
    d = world._delta(a[:, None, :], b[None, :, :]).norm(dim=-1)
    r, k = d.min(1)
    r = r.cpu().numpy()
    k = k.cpu().numpy()
    among = np.asarray(among)
    j = np.where(r <= max_r, among[k], -1)
    return j, r


def chain_report(world, exclude=()):
    """Lengths and sequences of every chain of length >= 2, excluding chains that contain
    any monomer in `exclude` (e.g. the template)."""
    ex = set(exclude)
    rows = []
    for path in world.chains(min_len=2):
        if ex.intersection(path):
            continue
        rows.append({"len": len(path), "seq": world.sequence(path), "ids": path})
    return rows


def copies_on_template(world, template_ids, cmap, max_r=2.8):
    """Chains lying along the template: every member within max_r of a template monomer.
    For each, return its sequence, the template letters it sits against, and how many
    positions are complementary."""
    tset = set(template_ids)
    letters = world.letters
    typ = world.typ.cpu().numpy()
    out = []
    for path in world.chains(min_len=2):
        if tset.intersection(path):
            continue
        partner, r = nearest(world, path, template_ids, max_r)
        if (partner < 0).any():
            continue
        seq = "".join(letters[typ[i]] for i in path)
        against = "".join(letters[typ[p]] for p in partner)
        ok = sum(1 for s, t in zip(seq, against) if cmap.get(t) == s)
        tpos = [template_ids.index(p) for p in partner]
        out.append({"seq": seq, "against": against, "match": ok, "len": len(path),
                    "template_pos": tpos})
    return out


def duplex_report(world, cmap, max_r=2.6, exclude=()):
    """Do chains lie against complementary chains? For every monomer that is in a chain, find
    the nearest monomer that is in a DIFFERENT chain (within max_r). Count how often the two
    letters are complements. The chance level for these same letters is computed by pairing
    the observed letters at random (sum over letters of p(a) * p(complement of a))."""
    cid = world.chain_ids()
    nb = world.nb.cpu().numpy()
    inchain = (nb >= 0).any(1)
    ex = set(exclude)
    idx = [i for i in range(world.P) if inchain[i] and i not in ex]
    if len(idx) < 2:
        return {"pairs": 0}
    pos = world.pos[torch.as_tensor(idx, device=world.device)]
    d = world._delta(pos[:, None, :], pos[None, :, :]).norm(dim=-1)
    cc = torch.as_tensor(cid[idx], device=world.device)
    d[cc[:, None] == cc[None, :]] = float("inf")
    r, k = d.min(1)
    typ = world.typ.cpu().numpy()
    L = world.letters
    a_l, b_l = [], []
    for n, (rr, kk) in enumerate(zip(r.tolist(), k.tolist())):
        if rr <= max_r:
            a_l.append(L[typ[idx[n]]]); b_l.append(L[typ[idx[kk]]])
    if not a_l:
        return {"pairs": 0}
    ok = sum(cmap.get(a) == b for a, b in zip(a_l, b_l))
    allv = a_l + b_l
    p = {x: allv.count(x) / len(allv) for x in set(allv)}
    chance = sum(p[x] * p.get(cmap.get(x), 0) for x in p)
    return {"pairs": len(a_l), "complementary": ok, "fraction": ok / len(a_l), "chance": chance}
