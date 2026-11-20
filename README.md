# kernelax: fused, vectorized and multithreaded C and CUDA kernels from JAX

[![tests](https://github.com/BodeTobias/kernelax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/BodeTobias/kernelax/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24395595.svg)](https://doi.org/10.5281/zenodo.24395595)

`kernelax.jit` is a drop-in replacement for `jax.jit` for JAX code that does many small operations per sample: a
material law at every integration point, a Newton solve per particle, a 3 x 3 eigenproblem per cell. It compiles such
code into fused kernels, C with SIMD and OpenMP on the CPU and CUDA on NVIDIA GPUs.

## When it helps

XLA fuses elementwise operations, but it splits a program at reductions, contractions, gathers and scatters, loops and
library calls. Every intermediate value that crosses such a boundary is written to memory as an array over the whole
batch and read back, and a `while` loop under `jax.vmap` runs every step over the whole batch until its slowest sample
has converged. When a program consists of many small operations per sample, this memory traffic, not the arithmetic,
dominates the run time, and `jax.jit` often reaches only a few percent of what the hardware can do. kernelax keeps
the intermediate values of a sample in registers and runs loops per sample. Typical cases:

- material laws and their derivatives (`jax.grad`, `jax.jacfwd`) at every integration point,
- local Newton solves, `while` loops and ODE time steps per sample,
- small linear algebra per sample (solves, inverses, eigenvalues of matrices up to about 8 x 8),
- element routines of a finite element assembly, with their gathers and scatter-adds,
- rollouts of simulators and reinforcement learning environments, one per sample (gymnax, pgx, jumanji, Brax), and
  simulations per parameter set (Jaxley, diffrax, dynamiqs).

## Example

```python
import jax, jax.numpy as jnp
import kernelax

jax.config.update("jax_enable_x64", True)


def psi(F):  # strain energy of a Neo-Hooke material at one point (shear modulus 1, Lame constant 10)
    J = jnp.linalg.det(F)
    return 0.5 * (jnp.sum(F * F) - 3) - jnp.log(J) + 5.0 * jnp.log(J) ** 2


stress = jax.vmap(jax.jacrev(psi))  # first Piola-Kirchhoff stress P = dpsi/dF at every point

F = jnp.eye(3) + 0.1 * jax.random.normal(jax.random.key(0), (1_000_000, 3, 3))  # a million points
P_xla = jax.jit(stress)(F)
P = kernelax.jit(stress)(F)  # the first call compiles the kernel (seconds; later processes load it from disk)
```

A call takes 3.9 ms with `kernelax.jit` and 420 ms with `jax.jit` on 8 cores of an AMD EPYC 9474F, and 0.27 ms
against 4.1 ms on an NVIDIA L40S.

## With common libraries

In these [examples](https://github.com/BodeTobias/kernelax/tree/main/examples) the only change is `kernelax.jit` in
place of `jax.jit`. Seconds on 8 cores of an AMD EPYC 9474F, each run in a fresh process; the first call includes
tracing and compilation, for `kernelax.jit` the build of its kernels (empty cache).

| Example | First call, `jax.jit` | `kernelax.jit` | Later calls, `jax.jit` | `kernelax.jit` |
|---|---:|---:|---:|---:|
| [diffrax](https://github.com/BodeTobias/kernelax/blob/main/examples/diffrax_kinetics.py): stiff kinetics (Kvaerno5) for 16 384 sets of rate constants | 14.3 | 3.9 | 14.5 | 0.12 (120×) |
| [optimistix](https://github.com/BodeTobias/kernelax/blob/main/examples/optimistix_bfgs.py): BFGS for 2 million parameter sets | 6.6 | 2.8 | 6.2 | 0.059 (105×) |
| [e3nn](https://github.com/BodeTobias/kernelax/blob/main/examples/e3nn_messages.py): equivariant messages on 4 million edges (float32) | 1.3 | 2.8 | 1.6 | 0.10 (15×) |
| [Jaxley](https://github.com/BodeTobias/kernelax/blob/main/examples/jaxley_sweep.py): the branched neuron of Jaxley's tutorial on `jit` and `vmap` for 4096 pairs of conductances | 3.1 | 4.9 | 2.3 | 0.088 (27×) |

## Installation

```bash
pip install kernelax
```

Requirements: Linux on x86-64 with AVX2 or AVX-512, Python >= 3.12, JAX 0.12 (>= 0.12.1), gcc and g++ >= 12 with OpenMP,
glibc >= 2.35. GPU kernels also need an NVIDIA GPU, JAX with CUDA (`jax[cuda12]` or `jax[cuda13]`) and the CUDA
toolkit's `nvcc`.

## Further API

`kernelax.kernel` compiles a function of one sample explicitly: under `jax.vmap` it runs as one kernel over the
batch, and its derivatives run as kernels too. With `psi` and `F` from the example above:

```python
stress = kernelax.kernel(jax.grad(psi))  # the stress at one point, as a kernel
P = jax.vmap(stress)(F)  # one kernel call over all points
A = jax.vmap(jax.jacfwd(stress))(F)  # its derivative, the tangent dP/dF, is a kernel too
```

`kernelax.export` writes a function of one sample as a C library with Fortran and Java interfaces, e.g. a material
law with its tangent for a finite element code:

```python
import numpy as np

kernelax.export(lambda F: (jax.grad(psi)(F), jax.hessian(psi)(F)), (np.eye(3),), name="neo_hooke", out_dir="neo_hooke")
# then make -C neo_hooke: libneo_hooke.so with neo_hooke_batch(nt, n, F, P, A) for n points on nt threads and
# neo_hooke_point(F, P, A) for one point
```

The [reference](https://github.com/BodeTobias/kernelax/blob/main/docs/reference.md) covers the options,
`report()`, environment variables, numerics (results are close to, but not bitwise equal to, XLA's) and limits.

## AI usage

Parts of the code, tests and documentation were developed with AI assistance.

## Citation

If you use kernelax in your work, please cite it as given in
[CITATION.cff](https://github.com/BodeTobias/kernelax/blob/main/CITATION.cff):

```bibtex
@software{kernelax,
  author  = {Bode, Tobias},
  title   = {kernelax: fused, vectorized and multithreaded C and CUDA kernels from JAX},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.24395595},
  url     = {https://github.com/BodeTobias/kernelax},
}
```

## License

MIT, see [LICENSE](https://github.com/BodeTobias/kernelax/blob/main/LICENSE).
