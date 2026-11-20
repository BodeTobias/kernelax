# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""A sweep over sodium and potassium conductances of a branched neuron (7 branches of 4 compartments, Hodgkin-Huxley
channels, 10 ms): the cell and the simulate function of Jaxley's tutorial on jit and vmap, jit(vmap(simulate)) over
4096 parameter sets.

    pip install jaxley
    python examples/jaxley_sweep.py         # kernelax.jit
    python examples/jaxley_sweep.py jax     # jax.jit"""
import sys

import jax
import jaxley as jx
import numpy as np
from jax import config, vmap
from jaxley.channels import K, Leak, Na

import kernelax
from timing import timed

config.update("jax_enable_x64", True)

dt = 0.025
t_max = 10.0

comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=4)
cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1, 2, 2])

cell.insert(Na())
cell.insert(K())
cell.insert(Leak())

cell.delete_stimuli()
current = jx.step_current(i_delay=1.0, i_dur=1.0, i_amp=0.1, delta_t=dt, t_max=t_max)
cell.branch(0).loc(0.0).stimulate(current)

cell.delete_recordings()
cell.branch(0).loc(0.0).record()


def simulate(params):
    param_state = None
    param_state = cell.data_set("Na_gNa", params[0], param_state)
    param_state = cell.data_set("K_gK", params[1], param_state)
    return jx.integrate(cell, param_state=param_state, delta_t=dt)


all_params = np.random.default_rng(0).random((4096, 2))
jit = jax.jit if "jax" in sys.argv[1:] else kernelax.jit  # the one change: kernelax.jit instead of jax.jit
timed(jit, vmap(simulate), all_params)
