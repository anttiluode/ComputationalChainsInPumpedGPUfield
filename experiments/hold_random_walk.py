"""Do the overnight random hold-encoded bits write the pairs' free (difference) phases?

Six different random bit streams, overnight source geometry (5 cells before bead 0), pinned ACGT
block. Prints the across-stream spread of the A/T and C/G difference phases every 600 steps.
Growing spread = the input does write the neutral mode, as a random walk (RESULTS_MORNING.md section 1).

    python experiments/hold_random_walk.py
"""
import os
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math, numpy as np, torch
torch.set_num_threads(1)
from chains import Config, World
from chains.probing import source_kernel
from experiments.bond_ablation import fork_world
def wrap(x): return (x+math.pi)%(2*math.pi)-math.pi
N=64
w=World(Config(N=N,seed=0,device='cpu',p_lig=0.0,kT=0.0,phase_noise=0.0,chain_boost=0.0))
xs=[N/2-5.4+3.6*i for i in range(4)]
w.add_monomers([w.letters.index(c) for c in "ACGT"], pos=[(x,N/2) for x in xs]); w.pinned[:]=True
for _ in range(400): w.step()
k=source_kernel(w,(xs[0]-5,N/2),1.0)   # overnight geometry: source 5 cells before bead 0
for amp in (0.05,0.2):
    runs=[]
    for s in range(6):
        bits=np.random.default_rng(s).choice([-1.,1.],size=400)
        v=fork_world(w); tr=[]
        for t in range(4800):
            v.step(external_source=k*float(amp*bits[t//12]))
            if t%600==599:
                th=v.theta.numpy(); tr.append((wrap(th[0]-th[3]), wrap(th[1]-th[2])))
        runs.append(tr)
    R=np.array(runs)   # streams x time x (AT,CG)
    spread=[np.abs(np.angle(np.exp(1j*R[:,i,:]).mean(0)*0+np.exp(1j*(R[:,i,:]-np.angle(np.exp(1j*R[:,i,:]).mean(0)))))).mean(0) for i in range(R.shape[1])]
    print(f"amp={amp}: across-stream spread of difference phases (AT, CG) at t=600..4800:")
    print("   ", [tuple(np.round(s,2)) for s in spread])
