# kernelax reference

Details behind the [README](https://github.com/BodeTobias/kernelax/blob/main/README.md): the API, how `kernelax.jit`
partitions a program, environment variables, the export of libraries, numerics, platforms and limits.

## API

```python
kernelax.jit(f, *, static_argnums=(), static_argnames=(), donate_argnums=(), donate_argnames=(),
             n_threads=None, keep_nans=False, build_first=None)        # .wait(variants=False), .report()
kernelax.kernel(f, *, static_argnums=(), static_argnames=(), n_threads=None, keep_nans=False)
kernelax.tracing()
kernelax.export(f, example, in_axes=0, *, name, out_dir=".", mode=("batch", "point"),
                isa=("avx2", "avx512", "neon"), keep_nans=False)       # -> the description (NAME.json)
```

`kernelax.jit` and `kernelax.kernel` also take their options alone, as decorators: `@kernelax.jit(n_threads=4)`.
Kernels run on the device of their arguments: C on the CPU, CUDA on an NVIDIA GPU.

- `static_argnums`, `static_argnames`, `donate_argnums`, `donate_argnames`: as in `jax.jit`. Keyword arguments of a
  `kernelax.kernel` function are mapped along their leading axis under `jax.vmap`, as there. Donated buffers go to XLA's
  program and to the one with kernels; the XLA program of a new program's first calls and trials (unless the function
  asks `kernelax.tracing()`) and inputs returned as outputs keep theirs (a donated buffer *may* be reused, as in JAX).
- `n_threads`: threads of a kernel call on the CPU; by default `OMP_NUM_THREADS` (its first entry), at most the CPUs of
  the process's affinity (`taskset`, an MPI rank's binding), else those CPUs; read at import.
- `keep_nans`: by default `x * 0`, `x - x` and `0 / x` fold to 0 also for NaN and inf, unlike `jax.jit` (this keeps
  the code of derivatives small); `keep_nans=True` keeps IEEE's NaN and inf. Off by default: it makes some kernels
  larger and slower (derivatives, complex arithmetic: up to 1.8x).
- `build_first` (default: `KERNELAX_BUILD_FIRST`, 1): a new program's first call builds its kernels and runs with them,
  unless that takes longer than `KERNELAX_BUILD_FIRST_WAIT` (10 s): then it runs in XLA and the kernels follow in a
  thread. Once the variants of K are built, calls alternate between the program with kernels and XLA's program (two
  timed calls each, the first of each untimed); XLA's is kept only if it is more than 5 % faster. `build_first=False`:
  the first calls run in XLA while the kernels are built in a thread, for programs called many times whose first call
  should cost what jax.jit's does; `g.wait()` waits for them (`variants=True`: also for the variants of K, see
  [Internals](#internals)).
- `g.report()`: per signature called, a dict: `signature` (dtypes and shapes of the dynamic arguments, static values),
  `status` (`building`: kernels being built, the calls run in XLA; `trial`: calls alternate between the program with
  kernels and XLA's, timed; `kernels`; `xla`; `failed`), `why` for `kernels` and `xla` (`index`: kernels as chosen
  by an earlier process; `measured`: by the trials, also an earlier process' choice of XLA; `long calls`: the cost
  model's choice without trials, its first timed call took longer than 50 ms (decided so again in every process;
  where the program's first call ran in XLA, as with `build_first=False`, and the call with kernels took more than 0.3
  of that, one call of XLA's program decides; where the kernels were ready for the first call (`build_first`), XLA's
  program never runs, so a long-running program is not compared with it: time `jax.jit` next to it where that
  matters); `cost model`: no region worth a kernel;
  `not built`; choices are kept per thread count of the
  calls; a measured choice of XLA stays until its entry in `index/` is deleted), the plan (`N`: the sample axes'
  lengths, `regions`, `loops`, `xla_heavy`: see `kernelax.tracing()`), `errors` (regions left in XLA, also warned),
  `ms` (the trials' best), `first_ms` (the first call with kernels).
- The vector width and K (sub-blocks interleaved per thread) are chosen automatically (the CPU's widest; K measured);
  there are no options for them. `.lower()` and `.trace()` are not provided (they would give XLA's program without
  kernels).
- `kernelax.tracing()` is true while kernelax traces a function for its kernels, so that a library can take a path
  that suits kernels there and another for XLA, e.g. one `vmap` over all elements of a mesh instead of `lax.map` in
  batches: a kernel computes one element at a time anyway, and the batches would only cost copies of their results (a
  finite element assembly on 1000² quadrilaterals: 330 ms in batches, 22 ms with one `vmap`). It is part of JAX's
  trace context: inner `jax.jit` functions are traced and cached apart under it. kernelax.jit's XLA program (its first
  call while kernels build, the comparison against them, and the program where no region is worth a kernel) runs
  without it, so the cost model counts the heavy operations (linear algebra, gathers, ...) that only XLA's program has
  for every region (`xla_heavy` in the report).

## What `kernelax.jit` runs as kernels

- The parts of a program that are parallel over the rows of its large arrays (the sample axis: the leading length of
  the program's largest arrays, up to three candidates tried, the one that leaves the least data to memory):
  elementwise code, row-wise reductions and products, `vmap`ped linear algebra, `while`, `scan` and `cond`, gathers
  from shared arrays and scatter-adds into them (as in a finite element assembly), also inside loops whose bodies are
  not row-wise as a whole. They run as kernels where a cost model expects them to be faster. The trials then compare
  the whole program with all its kernels against XLA's (see `build_first`); programs whose calls are long keep the
  cost model's choice (`long calls`: trials would run XLA's slower program as long; see `report()`).
- Rollouts as they are written for many environments or parameter sets: a `scan` over the steps with `vmap` over the
  samples inside each step, their random keys split from the step's key (`jax.random.split(key, N)`, whose key `s` is
  `fold_in(key, s)` with threefry and `jax_threefry_partitionable`, JAX's default) or from a key per sample, and `cond`
  or `switch` per sample (under `vmap`: a select of the branches' results) run as one kernel with the loop inside,
  also with their reset before the loop.
- The rest stays in XLA, and regions with more than 400000 operations per sample (counted with their sums over an
  inner axis as loops, which the kernels use from 200000 on). On the CPU also dense layers (a
  region with no loops whose work is half or more in contractions with a shared operand of 4096 entries or more: XLA's
  GEMM is faster) and regions whose largest contraction has more than 2^15 multiply-adds per sample. On the GPU the
  kernels keep dense layers as loops.
- While loops that run until all samples are done with one step counter (equinox's under `vmap`: diffrax,
  optimistix) become one loop per sample where that is proved on the jaxpr.
- A kernel's code does not depend on the sizes of the arrays it gathers from and sums into (passed at run time): an
  assembly over another mesh compiles only its XLA part.
- On the CPU, a program without a sample axis whose time goes into loops over small data (one local problem, a time
  stepping loop) runs as one kernel on one sample, if no other region got a kernel, its loops carry at most 256
  values, its inputs and outputs have at most 2^14 entries, it has at most 200000 operations and neither PRNG keys nor
  side effects.
- Code vmapped twice (elements x quadrature points, particles x neighbours) can take both leading axes as the sample
  axis: one sample per point, row data read per row, sums over the points added per row.
- First calls: a new program's first call builds its kernels (planning, trace, gcc; seconds) and runs with them,
  unless that takes longer than `KERNELAX_BUILD_FIRST_WAIT`: then it runs in XLA while the build goes on in the
  background, as with `build_first=False` from the start: gcc at idle priority (SCHED_IDLE), its time limit counting
  CPU seconds; the variants of K are always built that way (`g.wait()` waits for the kernels). K is chosen at the
  first call with 4096 samples or more; where that call takes longer than 0.25 s, the next calls run the other
  variants, one each, and the fastest is kept (no call is repeated for the choice). A program seen before loads its
  kernels from the index on disk and compiles less XLA than jax.jit (`benchmarks/first_call.py`).
- Threads: kernel calls through XLA run one after the other on one thread of kernelax's. Where the process's affinity
  is restricted (`taskset`, an MPI rank's or a batch system's binding) and the threads are as many as its CPUs, they
  are bound to one CPU each of it, unless `OMP_PROC_BIND` or `GOMP_CPU_AFFINITY` is set; with fewer threads, or where
  the process may run on all CPUs, they stay unbound, so that several such processes do not crowd onto the same first
  CPUs.
- Sums over the samples (reduced outputs, gradients of shared parameters, assemblies) depend on the number of threads
  in their last bits, not on the variant of K the runtime chooses: each thread sums the same samples in the same order
  whatever K.
- The cache directory (`KERNELAX_CACHE_DIR`, else `$XDG_CACHE_HOME/kernelax` or `~/.cache/kernelax`) must belong to
  the user: its libraries are loaded and run (one that others may write to is made private). It grows with the kernels
  built; deleting it is safe while no kernelax process runs (they are built again when needed; `index/` alone holds
  the choices and recorded build failures). As JAX, kernelax is not fork-safe: with `multiprocessing`, use the `spawn`
  or `forkserver` start method.

## Environment variables

| Variable | Default | |
|---|---|---|
| `KERNELAX_CACHE_DIR` | `$XDG_CACHE_HOME/kernelax`, else `~/.cache/kernelax` | kernels, index, runtime |
| `KERNELAX_CC`, `KERNELAX_CXX` | the newest of `gcc`, `g++` and `gcc-NN`, `g++-NN` on the PATH (>= 12) | compilers of the kernels and the runtime (a wrapper script: its text keys the cache) |
| `KERNELAX_JOBS` | CPUs of the affinity | gcc processes at a time |
| `KERNELAX_GCC_TIMEOUT` | 120 | seconds per gcc process (the parts of a large kernel each; background builds: of CPU, and at most 10 times as long in wall-clock time); beyond, the region stays in XLA |
| `KERNELAX_MAX_NODES` | 300000 | scalar operations per kernel; beyond, tracing stops |
| `KERNELAX_NVCC`, `KERNELAX_CUDA_ARCH` | `nvcc`, the first GPU's | CUDA compiler, architecture (`sm_XY`) |
| `KERNELAX_NVCC_JOBS`, `KERNELAX_NVCC_TIMEOUT` | CPUs of the affinity, 120 | GPU kernels built at a time, seconds per kernel |
| `KERNELAX_GPU_MAX_NODES` | 300000 | as `KERNELAX_MAX_NODES`, for GPU kernels |
| `KERNELAX_BUILD_FIRST` | 1 | 0 (or false): `build_first=False` by default |
| `KERNELAX_BUILD_FIRST_WAIT` | 10 | seconds a first call waits for its kernels (else XLA, the kernels in a thread) |
| `KERNELAX_EXIT_WAIT` | 600 | seconds at exit to finish kernel builds for the index (0: off; inf: all of them); compilers still running then are stopped |
| `OMP_NUM_THREADS` | | threads of a kernel call (`n_threads`) |
| `OMP_PROC_BIND`, `GOMP_CPU_AFFINITY` | | set: the kernel threads are bound by OpenMP as they say, not by kernelax |
| `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_ALLOCATOR` | | GPU: the free memory left for kernels' local memory |

For the numeric and boolean `KERNELAX_*` variables an empty value counts as unset, and a value that is not valid
(`KERNELAX_JOBS=-1`) stops `import kernelax` with an error naming it; a compiler that does not run fails at the first
build. `KERNELAX_X64` (default 1) and `KERNELAX_SCALE` (default 1) are read by `benchmarks/` (x64, batch sizes), the
first also by the tests that use `benchmarks/cases.py`.

## Export

`kernelax.export` writes a function of one sample as a library for programs without Python or JAX (FE codes, C, C++,
Fortran, Java), e.g. a material law with its consistent tangent:

```python
kernelax.export(point, (eps, state, params), in_axes=(0, 0, None), name="law", out_dir="law")  # then: make -C law
```

- `law_batch(nt, n, ...)`: n samples at once (one row per sample), on nt OpenMP threads (1 where the host parallelizes
  itself). One C file per instruction set (`law_batch_avx2.c`, `_avx512.c`, `_neon.c`); the dispatcher `law_batch.c`
  runs the variant for the CPU at hand (x86-64-v4: AVX-512, x86-64-v3: AVX2, aarch64: NEON), else the point function per
  sample. `law_isa()` names the variant. Returns 0, 1 if no variant runs on this CPU (and no point function was
  exported), 2 if n < 0, nt < 1 or nt beyond the processors.
- `law_point(...)`: one sample per call, for hosts that integrate point by point (MFront, UMAT): scalar code compiled
  for the CPU (`law.cpp`; no SIMD, OpenMP or libmvec). A Mohr-Coulomb return mapping (implicit tangent, one thread):
  0.74 us per point, the batch kernel 0.39 us.
- Both are reentrant; outputs must not overlap inputs. `law.h` lists every argument with its dtype, C type and shape;
  `law.json` the same for tools (`kernelax.export` returns it). C types: float64 `double`, 64-bit integers `int64_t` /
  `uint64_t` (their bits), float32 `float`, other integers `int32_t` / `uint32_t`; outputs of functions with any 8-byte
  dtype as `double` (integers: exact values; 64-bit ones `int64_t` / `uint64_t`); complex values as [re, im] on a
  trailing axis (as `double _Complex`). The definitions name their arguments by position (`kx_a0`, ...): names such as
  `K` or `W` do not meet the kernel's macros.
- `law.f90`: a Fortran module of `bind(C)` interfaces (arrays column-major there: (n, 6) in C is (6, n) in Fortran).
  `law.java`: a class over the Foreign Function & Memory API (JDK >= 22, `--enable-native-access`), Java arrays
  without a copy, their lengths checked (`IllegalArgumentException`). `Makefile`: `liblaw.so` for the machine it runs on
  (`gcc`/`g++`, or `CC`/`CXX`).
- Needs gcc >= 12 with glibc's libmvec (>= 2.35) and OpenMP for the batch kernel, and a C++ compiler (the library is
  linked with it).
  float16/bfloat16 arguments are not exported.

## Numerics

- float64 and float32 (float16/bfloat16 are computed in float32 and rounded to their format at every equation).
  float32 values are computed in float32, as XLA does (rounded at every jaxpr equation), the others in double. Kernels
  whose arrays all have elements of at most 4 bytes (float32, int32, also bool, 8- and 16-bit types) keep them so (16
  lanes with AVX-512, 8 with AVX2, 4 with NEON, no `jax_enable_x64` needed); arrays of 8-byte dtypes need
  `jax_enable_x64`, as in JAX.
- Integers are exact doubles, or int64 lanes where their bits matter (bitwise operations, shifts, wraparound,
  bitcasts): `jax.random` works per sample from seeds (threefry; `normal` through JAX's `erf_inv`, as XLA).
- Not bitwise equal to XLA (glibc's libmvec, e.g. `erfc` within 6 ulp of XLA's, its own `exp`, `-ffp-contract=fast`,
  divisions by a shared reciprocal; products that error-free transformations need rounded, as in JAX's double-word
  arithmetic of `gamma`, `lgamma`, `digamma`, stay rounded, so these agree with XLA). Kernels compute in the caller's
  floating-point mode: through XLA subnormal results flush to zero as in XLA (libmvec's functions take subnormal
  arguments as they are).
- A division by a value computed at run time is `x * (1/y)` (XLA divides; by a constant it multiplies by the
  reciprocal too): a denominator beyond 2^1022 (float32: 2^126) gives 0 there (its reciprocal is subnormal), and an
  exact quotient can come out one ulp below, so that `floor(x / y)` or `int(x / y)` of an exact multiple is one less in
  a few percent of cases (`jnp.floor_divide` and integer division are exact). True divisions cost a median +0.3 % over
  the benchmark cases, a 3 x 3 inverse +16 %.
- Derivatives of `jnp.linalg.det` (and `jax.scipy.linalg.det`) of 2 x 2 and 3 x 3 matrices: while kernelax traces,
  JAX's rule (an LU solve, `_cofactor_solve`) is replaced by d det = sum(cof(A) * dA) with the cofactors as 2 x 2
  minors; the value is det's own. The gradient of a 3 x 3 det takes 27 instead of 432 operations, its Hessian 9 instead
  of 1575; both are exact for integer matrices, singular ones too (there JAX's Hessian is NaN), and as accurate as
  JAX's rule otherwise. A NaN or inf entry reaches the cofactors that contain it (JAX's rule returns 0 for all).
  `jax.jit` and kernelax.jit's XLA program keep JAX's rule (its LU solve counts in the cost model: `xla_heavy`).
- `x * 0`, `x - x` and `0 / x` fold to 0 also for NaN and inf (small derivative code); `keep_nans=True` keeps them, so
  that NaN and inf propagate as in XLA.
- equinox's runtime errors (`eqx.error_if`, e.g. diffrax's `max_steps` reached or optimistix with `throw=True`) do not
  raise inside a kernel: as with `EQX_ON_ERROR=nan`, the checked value of a failing sample becomes NaN (integers
  their largest value, booleans True), which usually reaches its outputs; the other samples are unaffected. `jax.jit`
  raises (equinox's default).
- Complex numbers become pairs of real values inside kernels (fft by Cooley-Tukey over small twiddle matrices) and pass
  kernel boundaries as [re, im] on a trailing axis. On the GPU, `kernelax.jit` leaves ffts longer than 128 (float32) or
  32 (float64) points in XLA's cuFFT.
- `eig` (eigenvalues, left and right eigenvectors) and `schur` (unsorted, as in XLA) follow LAPACK's dgeev/dgees
  (dgebal's permutations and scaling, dlahqr, dlanv2), of complex matrices zgeev/zgees (zgebal, zlahqr, ztrevc3); where
  rounding takes another iteration path, a 2 x 2 block of T may be rotated otherwise and a complex eigenvector's sign
  differ from LAPACK's. The Schur form of a complex matrix equals LAPACK's up to the phase of each column of Z
  (rounding decides it, in LAPACK too).

## Platforms and limits

- Everything but `while` and long scans (whose xs and ys are small next to their body) is unrolled into scalars,
  except lookups in constant tables and gathers from inputs in memory. Suited to small fixed-size per-sample
  functions, not to large arrays or BLAS-type work: beyond 300000 scalar operations tracing stops with the place that
  causes it (`KERNELAX_MAX_NODES`).
- A region the cost model chose for a kernel but kernelax cannot build (an operation not covered, too large for gcc)
  stays in XLA with a warning; `g.report()` of `g = kernelax.jit(f)` reports the plan, the regions and their errors.
- Compile latency of seconds (large kernels: up to minutes), then from the disk cache.
- Linux only: x86-64 with AVX2/FMA or AVX-512, aarch64 with NEON (experimental); NVIDIA GPUs with CUDA 12 (>= 12.0,
  from compute capability 5.0) or CUDA 13 (from 7.5), as JAX's `jax[cuda12]` / `jax[cuda13]`.
- CPU kernels need gcc and g++ >= 12 and glibc's `libmvec` (>= 2.35; aarch64: >= 2.39). kernelax takes the newest of
  `gcc`/`g++` and `gcc-NN`/`g++-NN` on the PATH: gcc 12-14 spill AVX-512 masks a byte wide and reload them two bytes
  wide (GCC PR target/117159), a pattern the kernels avoid and the tests check in their assembly; gcc 15 also builds
  large kernels faster. `timeout` (coreutils) and `chrt` (util-linux) are used where present (a gcc ends with the
  process; background builds at idle priority). GPU kernels need the toolkit's `nvcc` (`KERNELAX_NVCC`; with gcc as its
  host compiler, at most the version the toolkit supports: CUDA 13.2 up to 15). Exported libraries build with the
  host's compiler (gcc >= 12 as well).
- Tested on Linux x86-64 and one NVIDIA L40S: the measurements and the cost model's thresholds come from an AMD EPYC
  9474F (AVX-512; AVX2 as builds without AVX-512 on it) and the L40S. aarch64 (NEON): the kernels of the coverage cases
  are checked under QEMU against the x86 kernels, never run on ARM hardware, not tuned. The GPU tests pass on the L40S
  with CUDA 12.9 and 13.2. Other NVIDIA GPUs: the kernels compile for all architectures of the toolkit (CUDA 12.9:
  sm_50 to sm_121; 13.2: sm_75 to sm_121), not run there. AMD GPUs (ROCm), macOS and Windows are not supported
  (Windows: WSL2 untested).
- Coverage: `tests/test_coverage.py` runs 266 jnp/linalg/scipy.special/nn/control-flow/AD/random/complex functions per
  sample against `jax.vmap`; `tests/test_cases.py` checks the 37 per-sample functions of `benchmarks/cases.py`
  (mechanics, finance, orbital mechanics, small linear algebra, eig/schur/sqrtm, loops, complex numbers, fft, MLP).
  `benchmarks/libraries.py` runs 41 typical uses of flax, equinox, diffrax, numpyro, blackjax, jax-md, optimistix,
  lineax, TFP, distrax, haiku, jraph, e3nn, evosax and jaxopt, rollouts of the environments of gymnax, pgx, jumanji,
  xland-minigrid, JaxMARL and Brax, and simulations of Jaxley, jaxoplanet and dynamiqs through `kernelax.jit` and
  `kernelax.kernel` against `jax.jit`.
- kernelax interprets jaxprs: it depends on JAX's primitives and their parameters, and follows JAX's latest release
  (now 0.12.1). Its few uses of JAX's internals (`jax._src`), which have no public equivalent, are in
  `src/kernelax/internals.py`; so it requires JAX < 0.13 until a newer release is tested.

## Tips

- Local Newton solves (return mappings, implicit time steps): a tangent by `jax.jacfwd` through the Newton loop
  differentiates every iteration; differentiating the converged point implicitly (`jax.lax.custom_root`) is the same
  tangent at a fraction of the cost (Mohr-Coulomb, per point in a batch: 0.34
  instead of 1.46 us).
- XLA's buffers on the CPU come from glibc's malloc, which maps large ones anew on every call and returns them
  afterwards; the OS zeroes their pages each time. Next to fast kernels that can be most of a call (jax.jit pays it as
  well). `MALLOC_MMAP_THRESHOLD_=4294967296 MALLOC_TRIM_THRESHOLD_=1000000000000` in the process's environment keeps
  freed memory for reuse, and the process keeps its peak memory. Measured on 8-40 cores: `kernelax.jit` on
  `benchmarks/programs.py`, ema 29 -> 1.6 ms, quadrature 4.1 -> 0.66 ms; a finite element tangent assembly on
  1000 x 1000 quadrilaterals, 26-29 -> 22 ms. The trim threshold alone made some programs much slower.

## Internals

1. `frontend.py`: `jax.make_jaxpr` of the per-sample function (AD, `custom_jvp` and while loops are resolved by JAX),
   its jaxpr interpreted over the hash-consed scalar nodes of `dag.py` (CSE, constant folding, algebraic
   simplification); small linear algebra (`linalg.py`: LU, QR, Cholesky, eigh up to 8 x 8, svd by one-sided Jacobi)
   is expanded. Work that only one side of selects needs (a reset, a rarely taken branch) runs inside loops in a block
   that the lanes enter only where one of them takes that side (`dag.guards`). A pick among options at a per-sample
   index (gathers, `dynamic_slice`) is a chain of selects where the options form up to 12 runs of equal neighbours,
   else an array of the options read per lane (GPU: up to 16 options as selects, more from local memory).
2. `emit.py`: the DAG as one C kernel over the batch: blocks of W samples (AVX2: 4, AVX-512: 8; twice as many for
   kernels whose arrays all have elements of up to 4 bytes), one SIMD lane per sample, rows read and written as vectors
   (short rows as contiguous vectors and shuffles, long ones transposed in registers), `while` as a masked SIMD loop,
   blocks spread over OpenMP threads, outputs beyond half the L3 of the threads streamed with non-temporal stores. K
   sub-blocks of W samples are interleaved statement by statement to hide latency; the runtime picks K by measurement
   on the first call with >= 4096 samples (kept in the index per thread count). `exp`, `expm1` and `tanh` of doubles
   are inline, the other elementary functions (all of float32) come from glibc's libmvec. `vec.py` holds the C vector
   types and helpers per type and configuration.
3. `build.py`: gcc, cached on disk, at -O3; kernels with loops of 256 KB to 2 MB of C at -O1 with FMA contraction and
   SLP vectorization, larger ones at -O1 with SLP (gcc's time; as fast or faster there). `cpu.py`: a kernel's variants
   built in parallel and registered with the runtime
   (`runtime.cpp`, g++ against jaxlib's XLA FFI headers: choice of K, FFI handler; built when a kernel is first
   lowered, so that `import kernelax` needs no g++). `reference.py`: the DAG evaluated with numpy (loop counts of
   `benchmarks/run.py`).
4. `primitive.py`: `kernel`, a JAX primitive whose calls lower to an FFI custom call; batching folds vmap's batch into
   the sample axis, JVP and transpose bind the kernels of `jax.jvp` / `jax.vjp` of the per-sample function; cotangents
   of arguments shared by all samples are summed inside the kernel (map-reduce).
5. `auto.py`: `jit`, the planner (regions, cost model, trials, index); `gpu.py`: the CUDA emitter and build;
   `standalone.py`: `export`.

## Tests and benchmarks

```bash
pip install -e .[dev]
pytest tests
python benchmarks/run.py [case ...]   # kernels of benchmarks/cases.py against jax.jit(jax.vmap(f)) and the roofline bound
python benchmarks/auto.py [N]         # kernelax.jit against jax.jit on whole programs (benchmarks/programs.py)
python benchmarks/first_call.py [N]   # their first calls, with an empty and a filled cache
python benchmarks/per_sample.py [N]   # jax.jit(jax.vmap(f)) against kernelax.kernel through the public API, CPU or GPU
python benchmarks/libraries.py --all out.jsonl    # uses of JAX libraries, each in a fresh process with an empty cache
python benchmarks/libraries.py --compare old.jsonl new.jsonl
```

The libraries are installed apart from the environment, at the versions tested:
`pip install --target tmp/pylibs --no-deps -r benchmarks/requirements-libraries.txt`, then `PYTHONPATH=tmp/pylibs`.

Before a release:
- `pytest tests` on the CPU and `pytest tests/test_gpu.py` on a GPU (CUDA 12 and 13).
- `python benchmarks/run.py`: no case slower than in the last release.
- `python benchmarks/libraries.py --all` on the CPU and on a GPU (`KERNELAX_SCALE=4`), compared with the last
  release's results (`--compare`, which marks cases more than 5 % slower, first calls more than 20 %, and changes of
  status): each regression explained or fixed.
- `python examples/compare.py` for the README's table.
