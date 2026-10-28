"""The templating gate.

Round 1: a pinned template chain sits in a soup of free monomers (equal numbers of each
letter). Pump on. New backbone bonds form only by the type-blind ligation rule. We record
every ligation that happens next to the template and score whether each new monomer is the
complement of the template monomer it sits against. Chance is about 25%.

Round 2: the longest copy from round 1 becomes the template in a fresh soup. If copying is
real, its copies should spell the ORIGINAL template again (complement of the complement).

Controls (same seeds):
  pump_off   receiver mode with the pump switched off: no pairing force at all
  detuned    pump moved to 2*Omega = 0.152 so no pair of letters sums to it
  medium     the pump drives the field's mass term instead of the monomers' ears

    python experiments/gate.py --seeds 3 --round2
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from chains import Config, World
from chains.analysis import complement_map, nearest, copies_on_template, chain_report, duplex_report

TEMPLATE = "GATTACAG"


def arm_config(arm, seed, N):
    cfg = Config(N=N, seed=seed)
    if arm == "pump_off":
        cfg.pump_on = False
    elif arm == "detuned":
        cfg.Omega = 0.076
    elif arm == "medium":
        cfg.mode = "medium"
    return cfg


def run_round(cfg, template, n_each, steps, log_every=2000, label=""):
    w = World(cfg)
    N = cfg.N
    start = (N / 2 - (len(template) - 1) * cfg.d0 / 2, N / 2)
    tids = w.add_chain(template, start)
    w.pinned[tids] = True
    free = np.repeat(np.arange(len(w.letters)), n_each)
    w.rng.shuffle(free)
    w.add_monomers(free)
    scoring = complement_map(Config(Omega=Config().Omega))   # score against the A-T / C-G pairing
    seen = 0
    events = []          # ligations next to the template: (letter, template letter, correct?)
    t0 = time.time()
    for t in range(steps):
        w.step()
        if len(w.events) > seen:
            new = w.events[seen:]
            seen = len(w.events)
            for (_, i, j) in new:
                part, _ = nearest(w, [i, j], tids, 2.8)
                for m, p in zip((i, j), part):
                    if p < 0:
                        continue
                    a = w.letters[int(w.typ[m])]
                    b = w.letters[int(w.typ[p])]
                    events.append((a, b, scoring[b] == a))
        if log_every and (t + 1) % log_every == 0:
            ch = chain_report(w, exclude=tids)
            print(f"  {label} t={t+1} ligations={w.ligations} chains={len(ch)} "
                  f"template-site events={len(events)} ({time.time()-t0:.0f}s)", flush=True)
    copies = copies_on_template(w, tids, scoring)
    soup = chain_report(w, exclude=tids)
    w.duplex = duplex_report(w, scoring, exclude=tids)
    return w, events, copies, soup


def summarize(events, copies):
    n = len(events)
    ok = sum(e[2] for e in events)
    cpos = sum(c["len"] for c in copies)
    cok = sum(c["match"] for c in copies)
    return {"site_events": n, "site_correct": ok, "site_fidelity": ok / n if n else None,
            "copies": len(copies), "copy_monomers": cpos, "copy_matches": cok,
            "copy_fidelity": cok / cpos if cpos else None,
            "copy_seqs": [(c["seq"], c["against"]) for c in copies]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--n_each", type=int, default=30)
    ap.add_argument("--arms", nargs="+", default=["receiver", "pump_off", "detuned", "medium"])
    ap.add_argument("--round2", action="store_true")
    ap.add_argument("--held_min", type=float, default=None)
    ap.add_argument("--out", default="results/gate.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res = {}
    for arm in a.arms:
        res[arm] = []
        for s in range(a.seeds):
            cfg = arm_config(arm, s, a.N)
            if a.held_min is not None:
                cfg.held_min = a.held_min
            w, ev, cp, soup = run_round(cfg, TEMPLATE, a.n_each, a.steps, label=f"{arm}/s{s}")
            r = summarize(ev, cp)
            r["seed"] = s
            r["soup_chains"] = len(soup)
            r["soup_len_max"] = max([c["len"] for c in soup], default=0)
            r["soup_seqs"] = sorted([c["seq"] for c in soup], key=len, reverse=True)[:20]
            r["duplex"] = w.duplex
            if a.round2 and arm == "receiver" and cp:
                best = max(cp, key=lambda c: (c["len"], c["match"]))
                tmpl2 = best["seq"]
                if len(tmpl2) >= 3:
                    cfg2 = arm_config(arm, 100 + s, a.N)
                    if a.held_min is not None:
                        cfg2.held_min = a.held_min
                    _, ev2, cp2, _ = run_round(cfg2, tmpl2, a.n_each, a.steps, label=f"round2/s{s}")
                    r2 = summarize(ev2, cp2)
                    r2["template"] = tmpl2
                    r["round2"] = r2
            res[arm].append(r)
            print(f"{arm} seed {s}: duplex {w.duplex}", flush=True)
            print(f"{arm} seed {s}: site fidelity {r['site_fidelity']} over {r['site_events']} events; "
                  f"copies {r['copies']} ({r['copy_matches']}/{r['copy_monomers']} complementary)", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
    print("\nSUMMARY (chance ~0.25)")
    for arm, rs in res.items():
        n = sum(r["site_events"] for r in rs)
        ok = sum(r["site_correct"] for r in rs)
        cm = sum(r["copy_monomers"] for r in rs)
        cmok = sum(r["copy_matches"] for r in rs)
        print(f"  {arm:9s} site fidelity {ok}/{n} = {ok/max(n,1):.2f}   copy fidelity {cmok}/{cm} = {cmok/max(cm,1):.2f}")


if __name__ == "__main__":
    main()
