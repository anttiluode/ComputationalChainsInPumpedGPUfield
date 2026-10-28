"""Live viewer (pygame) or headless run.

    python run.py                         # soup with a GATTACAG template, live
    python run.py --no-template           # soup only (stage 1: do chains form?)
    python run.py --N 256 --n_each 600    # bigger world on the GPU
    python run.py --headless --steps 50000 --out runs/a

Keys:  SPACE pause   P pump on/off   F field view on/off   S save PNG   +/- steps per frame
       left-drag  stir (pushes monomers)   ESC quit
Colours: A red, C yellow, G green, T blue. White lines are backbone bonds.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
from chains import Config, World
from chains.analysis import complement_map, copies_on_template, chain_report

COL = {"A": (235, 80, 70), "C": (240, 205, 60), "G": (70, 205, 110), "T": (80, 150, 245)}


def build(a):
    cfg = Config(N=a.N, seed=a.seed, mode=a.mode)
    w = World(cfg)
    tids = []
    if not a.no_template:
        start = (a.N / 2 - (len(a.template) - 1) * cfg.d0 / 2, a.N / 2)
        tids = w.add_chain(a.template, start)
        w.pinned[tids] = True
    free = np.repeat(np.arange(len(w.letters)), a.n_each)
    w.rng.shuffle(free)
    w.add_monomers(free)
    return w, tids


def summary(w, tids):
    soup = chain_report(w, exclude=tids)
    lens = [c["len"] for c in soup]
    s = {"t": w.t, "ligations": w.ligations, "chains": len(soup),
         "longest": max(lens, default=0), "mean_len": float(np.mean(lens)) if lens else 0.0}
    if tids:
        cp = copies_on_template(w, tids, complement_map(w.cfg))
        n = sum(c["len"] for c in cp)
        s["copies"] = [c["seq"] for c in cp]
        s["copy_fidelity"] = sum(c["match"] for c in cp) / n if n else None
    return s


def headless(a):
    w, tids = build(a)
    os.makedirs(a.out, exist_ok=True)
    json.dump(json.loads(w.cfg.to_json()), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w")
    t0 = time.time()
    for t in range(a.steps):
        w.step()
        if (t + 1) % a.log_every == 0:
            s = summary(w, tids)
            s["sec"] = round(time.time() - t0, 1)
            log.write(json.dumps(s) + "\n"); log.flush()
            print(s, flush=True)
    seqs = [w.sequence(p) for p in w.chains(min_len=2)]
    json.dump(seqs, open(os.path.join(a.out, "chains.json"), "w"), indent=1)


def live(a):
    import pygame
    w, tids = build(a)
    N = a.N
    pygame.init()
    info = pygame.display.Info()
    scale = max(2, min((info.current_h - 120) // N, (info.current_w - 360) // N))
    W = N * scale
    screen = pygame.display.set_mode((W + 300, W))
    pygame.display.set_caption("Chains in a pumped field")
    font = pygame.font.SysFont("consolas,menlo,monospace", 15)
    clock = pygame.time.Clock()
    paused, show_field, spf = False, True, 10
    tset = set(tids)
    stats = summary(w, tids)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_p: w.cfg.pump_on = not w.cfg.pump_on
                elif e.key == pygame.K_f: show_field = not show_field
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS): spf = min(400, spf * 2)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS): spf = max(1, spf // 2)
                elif e.key == pygame.K_s:
                    pygame.image.save(screen, f"frame_{w.t:07d}.png")
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if mx < W:
                c = torch.tensor([mx / scale, my / scale], device=w.device)
                d = w._delta(c[None], w.pos)
                r = d.norm(dim=-1)
                near = (r < 6) & (~w.pinned)
                w.pos[near] = torch.remainder(w.pos[near] + 0.8 * d[near] / r[near, None].clamp(min=1e-3), N)
        if not paused:
            for _ in range(spf):
                w.step()
            if w.t % 200 < spf:
                stats = summary(w, tids)
        if show_field:
            f = w.psi.detach().cpu().numpy()
            s = np.abs(f).max() + 1e-9
            g = np.clip(f / s, -1, 1)
            img = np.stack([np.clip(g, 0, 1) * 90, np.abs(g) * 40, np.clip(-g, 0, 1) * 110], -1).astype(np.uint8)
            surf = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            screen.blit(pygame.transform.scale(surf, (W, W)), (0, 0))
        else:
            screen.fill((8, 8, 14), (0, 0, W, W))
        pos = w.pos.cpu().numpy()
        typ = w.typ.cpu().numpy()
        nb = w.nb.cpu().numpy()
        for i in range(w.P):
            for j in nb[i]:
                if j > i:
                    p, q = pos[i], pos[j]
                    if np.abs(p - q).max() < N / 2:
                        pygame.draw.line(
                            screen,
                            (230, 230, 230),
                            (float(p[0]) * scale, float(p[1]) * scale),
                            (float(q[0]) * scale, float(q[1]) * scale),
                            2,
                        )
        r = max(2, int(scale * 0.8))
        for i in range(w.P):
            col = COL[w.letters[typ[i]]]
            pygame.draw.circle(screen, col, (int(pos[i, 0] * scale), int(pos[i, 1] * scale)), r)
            if i in tset:
                pygame.draw.circle(screen, (255, 255, 255), (int(pos[i, 0] * scale), int(pos[i, 1] * scale)), r + 2, 1)
        screen.fill((16, 16, 22), (W, 0, 300, W))
        lines = [f"t {w.t}", f"pump {'ON' if w.cfg.pump_on else 'off'} ({w.cfg.mode})",
                 f"steps/frame {spf}", f"ligations {stats['ligations']}",
                 f"chains {stats['chains']}  longest {stats['longest']}"]
        if tids:
            lines += [f"template {a.template}", "chains lying on template:"] + [f"  {s}" for s in stats.get("copies", [])[:12]]
            fid = stats.get("copy_fidelity")
            lines.append(f"complementary to nearest bead {fid:.2f}" if fid is not None else "complementary to nearest bead -")
        lines += ["", "A red C yellow G green T blue", "SPACE P F S +/- drag"]
        for k, ln in enumerate(lines):
            screen.blit(font.render(ln, True, (220, 220, 220)), (W + 12, 12 + 19 * k))
        pygame.display.flip()
        clock.tick(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--n_each", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="receiver")
    ap.add_argument("--template", default="GATTACAG")
    ap.add_argument("--no-template", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--out", default="runs/default")
    a = ap.parse_args()
    headless(a) if a.headless else live(a)


if __name__ == "__main__":
    main()
