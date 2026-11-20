# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""jax.jit against kernelax.jit on typical uses of JAX libraries: one case per process (a JSON line), or all of them,
each in a fresh process with an empty kernel cache (before a release: against the last release's results).

    python benchmarks/libraries.py case                  # one case
    python benchmarks/libraries.py --all out.jsonl [case ...]   # all (or the given) cases, appended to out.jsonl
    python benchmarks/libraries.py --compare old.jsonl new.jsonl  # new against old: ratios, status, errors

Per case: jax.jit(f) and kernelax.jit(f); per stage g of f.per_sample (the per-sample functions that do its work)
also kernelax.jit(jax.vmap(g)) (where f is not that already) and jax.jit(jax.vmap(kernelax.kernel(g))). Times: the best
of alternating calls, each program called once untimed before its timed call, kernelax.jit's trials decided first.
Cases: flax, equinox, diffrax, numpyro, blackjax, jax-md, optimistix, lineax, TFP, distrax, haiku, jraph, e3nn,
evosax, jaxopt, a GP; environments of gymnax, pgx, jumanji, xland-minigrid, JaxMARL and Brax over 200 steps as users
write the rollout (a scan over the steps, vmap over the environments in it; per sample: one environment's rollout);
Jaxley (a neuron's simulation and its gradient per pair of conductances), jaxoplanet (a light curve per time and the
gradient of a fit), dynamiqs (the Lindblad equation per detuning). The libraries, at the versions tested, are needed
only here, e.g. next to the installed jax without changing it:
    pip install --target tmp/pylibs --no-deps -r benchmarks/requirements-libraries.txt
    PYTHONPATH=tmp/pylibs python benchmarks/libraries.py --all tmp/libraries.jsonl
With KERNELAX_X64=0 the cases run in JAX's default precision (float32 data: kernels of float32 arrays, 16 lanes; the
environments always do). KERNELAX_SCALE=s multiplies the batch sizes by s (batches that fill a GPU)."""
import json, os, sys, time
import warnings
warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", os.environ.get("KERNELAX_X64", "1") == "1")  # KERNELAX_X64=0: JAX's default, float32
X64 = jax.config.jax_enable_x64  # (float32: solver tolerances it can reach)
from kernelax import jit as kernelax_jit

r = np.random.default_rng(0)
S = int(os.environ.get("KERNELAX_SCALE", "1"))  # batch sizes times S


def stage(g, in_axes, args=None, **extra):
    """A per-sample function g whose vmap over in_axes (0 or None per argument) does part of a case's work, on
    the case's arguments or on `args`; extra: operations per sample outside g (atomic: scatter-adds)."""
    return dict(g=g, in_axes=in_axes, args=args, extra=extra)


def tagged(f, *stages):
    f.per_sample = stages
    return f


def vmapped(g, in_axes=0):
    f = tagged(jax.vmap(g, in_axes), stage(g, in_axes))
    f.vmapped = True  # (kernelax.jit of the stage's vmap: f itself)
    return f


def flax_mlp_infer():
    import flax.linen as nn
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.gelu(nn.Dense(64)(x)); x = nn.gelu(nn.Dense(64)(x)); return nn.Dense(1)(x)
    m = MLP(); x = r.standard_normal((100_000 * S, 8))
    p = m.init(jax.random.PRNGKey(0), x[:1])
    f = lambda p, x: m.apply(p, x)
    return tagged(f, stage(f, (None, 0))), (p, x)


def flax_train_step():
    import flax.linen as nn, optax
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.tanh(nn.Dense(64)(x)); x = nn.tanh(nn.Dense(64)(x)); return nn.Dense(1)(x)
    m, opt = MLP(), optax.adam(1e-3)
    x, y = r.standard_normal((4096 * S, 8)), r.standard_normal((4096 * S, 1))
    p = m.init(jax.random.PRNGKey(0), x[:1]); s = opt.init(p)
    def step(p, s, x, y):
        loss, g = jax.value_and_grad(lambda p: jnp.mean((m.apply(p, x) - y) ** 2))(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss
    point = lambda p, s, x, y: jax.value_and_grad(lambda p: jnp.sum((m.apply(p, x) - y) ** 2))(p)  # (summed)
    return tagged(step, stage(point, (None, None, 0, 0))), (p, s, x, y)


def eqx_pinn():
    import equinox as eqx
    net = eqx.nn.MLP(2, 1, 32, 3, activation=jnp.tanh, key=jax.random.PRNGKey(0))
    net = jax.tree.map(lambda a: a.astype(jnp.float64) if eqx.is_array(a) else a, net)
    params, static = eqx.partition(net, eqx.is_array)
    x = r.uniform(-1, 1, (100_000 * S, 2))
    def point(params, z):  # u and the Laplacian of u per point (forward-over-forward), as in a PINN loss
        f = lambda z: eqx.combine(params, static)(z)[0]
        return (jnp.trace(jax.jacfwd(jax.jacfwd(f))(z)) + jnp.sin(z[0])) ** 2
    residual = lambda params, x: jnp.mean(jax.vmap(point, (None, 0))(params, x))
    return tagged(residual, stage(point, (None, 0))), (params, x)


def diffrax_ode():
    import diffrax
    def solve(y0, args):
        f = lambda t, y, a: jnp.stack([a[0] * y[0] - a[1] * y[0] * y[1], a[2] * y[0] * y[1] - a[3] * y[1]])
        sol = diffrax.diffeqsolve(diffrax.ODETerm(f), diffrax.Tsit5(), 0.0, 10.0, 0.05, y0, args=args,
                                  stepsize_controller=diffrax.PIDController(rtol=1e-8 if X64 else 1e-5, atol=1e-8 if X64 else 1e-5), max_steps=4096,
                                  saveat=diffrax.SaveAt(t1=True))
        return sol.ys[-1]
    y0 = r.uniform(0.5, 2.0, (4096 * S, 2)); a = np.array([1.1, 0.4, 0.1, 0.4])
    return vmapped(solve, (0, None)), (y0, a)


def diffrax_stiff():
    import diffrax
    def solve(k):
        f = lambda t, y, k: jnp.stack([-k[0] * y[0] + k[2] * y[1] * y[2], k[0] * y[0] - k[1] * y[1] ** 2 - k[2] * y[1] * y[2],
                                       k[1] * y[1] ** 2])
        sol = diffrax.diffeqsolve(diffrax.ODETerm(f), diffrax.Kvaerno5(), 0.0, 100.0, 1e-4, jnp.array([1.0, 0.0, 0.0]), args=k,
                                  stepsize_controller=diffrax.PIDController(rtol=1e-6 if X64 else 1e-4, atol=1e-9 if X64 else 1e-6), max_steps=4096,
                                  saveat=diffrax.SaveAt(t1=True))
        return sol.ys[-1]
    k = np.array([0.04, 3e7, 1e4]) * r.uniform(0.8, 1.2, (1024 * S, 3))
    return vmapped(solve), (k,)


def numpyro_logp():
    import numpyro, numpyro.distributions as dist
    from numpyro.infer.util import potential_energy
    X = r.standard_normal((500, 5)); yv = (X @ r.standard_normal(5) + 0.3 * r.standard_normal(500) > 0).astype(np.float64)
    def model(X, y):
        w = numpyro.sample("w", dist.Normal(0, 1).expand([5]))
        b = numpyro.sample("b", dist.Normal(0, 1))
        s = numpyro.sample("s", dist.HalfNormal(1.0))
        numpyro.sample("obs", dist.Bernoulli(logits=(X @ w + b) / s), obs=y)
    pe = lambda z: potential_energy(model, (X, yv), {}, z)
    n = 4096 * S
    z = {"w": r.standard_normal((n, 5)), "b": r.standard_normal(n), "s": r.uniform(-1, 1, n)}  # (s: unconstrained)
    return vmapped(jax.value_and_grad(pe)), (z,)


def blackjax_nuts():
    import blackjax
    A = r.standard_normal((10, 10)); P = A @ A.T / 10 + np.eye(10)
    logdensity = lambda x: -0.5 * x @ P @ x + jnp.sum(jnp.log1p(jnp.tanh(x) ** 2))
    nuts = blackjax.nuts(logdensity, step_size=0.3, inverse_mass_matrix=jnp.ones(10))
    def run(x, key):
        st = nuts.init(x)
        def body(st, k):
            st, info = nuts.step(k, st)
            return st, info.num_integration_steps
        st, n = jax.lax.scan(body, st, jax.random.split(key, 10))
        return st.position, n.sum()
    return vmapped(run), (r.standard_normal((512 * S, 10)), jax.random.split(jax.random.PRNGKey(1), 512 * S))


def jaxmd_lj():
    from jax_md import space, energy
    N, box = 4096 * S, 30.0 * np.sqrt(S)  # (the same density)
    disp, shift = space.periodic(box)
    pos = r.uniform(0, box, (N, 2))
    nbr_fn, energy_fn = energy.lennard_jones_neighbor_list(disp, box, sigma=1.0, epsilon=1.0, r_onset=2.0, r_cutoff=2.5)
    nbrs = nbr_fn.allocate(pos)
    def ef(pos, nbrs):
        e, g = jax.value_and_grad(energy_fn)(pos, nbrs)
        return e, -g
    lj = energy.multiplicative_isotropic_cutoff(energy.lennard_jones, 2.0, 2.5)
    def pair(ij, pos):  # one pair of the (ordered sparse) list: its energy and force, scattered to both particles
        e, g = jax.value_and_grad(lambda d: lj(space.distance(d)))(disp(pos[ij[0]], pos[ij[1] % N]))
        return jnp.where(ij[0] < N, e, 0.0), jnp.where(ij[0] < N, g, 0.0)
    return tagged(ef, stage(pair, (0, None), (np.asarray(nbrs.idx).T, pos), atomic=4)), (pos, nbrs)


def optx_root():  # Newton root find per parameter set (equinox while loop under vmap)
    import optimistix as optx
    def solve(p):
        fn = lambda y, p: jnp.stack([y[0] ** 3 + y[0] + 0.3 * jnp.tanh(y[1]) - p[0], y[1] + 0.5 * jnp.tanh(y[0] * y[1]) - p[1]])
        return optx.root_find(fn, optx.Newton(rtol=1e-12 if X64 else 1e-6, atol=1e-12 if X64 else 1e-6), jnp.zeros(2), args=p, throw=False, max_steps=64).value
    return vmapped(solve), (r.uniform(-3, 3, (8192 * S, 2)),)


def optx_minimise():  # BFGS on a Rosenbrock function per parameter set
    import optimistix as optx
    def solve(p):
        fn = lambda y, p: (p[0] - y[0]) ** 2 + p[1] * (y[1] - y[0] ** 2) ** 2
        return optx.minimise(fn, optx.BFGS(rtol=1e-10 if X64 else 1e-6, atol=1e-10 if X64 else 1e-6), jnp.array([-1.0, 1.0]), args=p, throw=False, max_steps=512).value
    p = np.stack([r.uniform(0.5, 1.5, 2048 * S), r.uniform(5, 50, 2048 * S)], 1)
    return vmapped(solve), (p,)


def lineax_solve():  # batched small dense solves through lineax (LU and Cholesky), 50 000 systems 5 x 5
    import lineax as lx
    def solve(A, S, b):
        x = lx.linear_solve(lx.MatrixLinearOperator(A), b, lx.LU()).value
        y = lx.linear_solve(lx.MatrixLinearOperator(S, lx.positive_semidefinite_tag), b, lx.Cholesky()).value
        return x, y
    N = 50_000 * S
    A = r.standard_normal((N, 5, 5)) + 5 * np.eye(5); B = r.standard_normal((N, 5, 5)); P = B @ B.transpose(0, 2, 1) / 5 + np.eye(5)
    return vmapped(solve), (A, P, r.standard_normal((N, 5)))


def tfp_gmm_logprob():  # TFP (substrates.jax): log density of a 2-d Gaussian mixture and its gradient wrt the parameters
    from tensorflow_probability.substrates import jax as tfp
    tfd = tfp.distributions
    X = r.standard_normal((100_000 * S, 2))
    def gmm(params):
        logits, mu, Lraw = params
        L = jnp.tril(Lraw) + 2.0 * jnp.eye(2)
        return tfd.MixtureSameFamily(tfd.Categorical(logits=logits), tfd.MultivariateNormalTriL(loc=mu, scale_tril=L))
    loss = lambda params, X: -jnp.mean(gmm(params).log_prob(X))
    point = lambda params, x: jax.value_and_grad(lambda p: -gmm(p).log_prob(x))(params)  # (summed)
    params = (jnp.zeros(3), jnp.asarray(r.standard_normal((3, 2))), jnp.asarray(0.1 * r.standard_normal((3, 2, 2))))
    return tagged(jax.value_and_grad(loss), stage(point, (None, 0))), (params, X)


def distrax_tanh_policy():  # SAC-style tanh-squashed Gaussian policy: sample and log-prob per state
    import distrax
    def act(mu, log_std, key):
        d = distrax.Transformed(distrax.MultivariateNormalDiag(mu, jnp.exp(log_std)), distrax.Block(distrax.Tanh(), 1))
        return d.sample_and_log_prob(seed=key)
    N = 65_536 * S
    return vmapped(act), (r.standard_normal((N, 4)), r.uniform(-2, 0, (N, 4)), jax.random.split(jax.random.PRNGKey(0), N))


def haiku_per_example_grads():  # per-example gradients of a haiku MLP (as in DP-SGD)
    import haiku as hk
    net = hk.without_apply_rng(hk.transform(lambda x: hk.nets.MLP([32, 32, 1], activation=jax.nn.softplus)(x)))
    x, y = r.standard_normal((4096 * S, 8)), r.standard_normal((4096 * S, 1))
    p = jax.tree.map(lambda a: a.astype(jnp.float64), net.init(jax.random.PRNGKey(0), x[:1]))
    loss = lambda p, x, y: jnp.sum((net.apply(p, x) - y) ** 2)
    def grads(p, x, y):
        g = jax.vmap(jax.grad(loss), (None, 0, 0))(p, x, y)
        norms = jnp.sqrt(sum(jnp.sum(a.reshape(a.shape[0], -1) ** 2, 1) for a in jax.tree.leaves(g)))
        clip = jnp.minimum(1.0, 1.0 / norms)
        return jax.tree.map(lambda a: jnp.mean(a * clip.reshape((-1,) + (1,) * (a.ndim - 1)), 0), g)
    def clipped(p, x, y):  # one example's clipped gradient (summed)
        g = jax.grad(loss)(p, x, y)
        clip = jnp.minimum(1.0, 1.0 / jnp.sqrt(sum(jnp.sum(a ** 2) for a in jax.tree.leaves(g))))
        return jax.tree.map(lambda a: a * clip, g)
    return tagged(grads, stage(clipped, (None, 0, 0))), (p, x, y)


def gymnax_rollout():  # Pendulum-v1: 4096 environments x 200 steps with a linear policy and noise (scan inside vmap)
    import gymnax
    env, ep = gymnax.make("Pendulum-v1")
    def rollout(key, w):
        k0, k1 = jax.random.split(key)
        obs, st = env.reset(k0, ep)
        def step(c, k):
            obs, st, ret = c
            a = jnp.clip(obs @ w + 0.1 * jax.random.normal(k, ()), -2.0, 2.0)[None]
            obs, st, rew, _, _, _ = env.step(k, st, a, ep)
            return (obs, st, ret + rew), None
        return jax.lax.scan(step, (obs, st, 0.0), jax.random.split(k1, 200))[0][2]
    N = 4096 * S
    return vmapped(rollout), (jax.random.split(jax.random.PRNGKey(0), N), r.standard_normal((N, 3)))


def jraph_gnn():  # one GraphNetwork message pass (edge MLP, segment sums, node MLP), 20 000 nodes, 100 000 edges
    import jraph
    nn_, ne, d = 20_000 * S, 100_000 * S, 16
    W = [jnp.asarray(r.standard_normal(s) / np.sqrt(s[0])) for s in [(3 * d, 32), (32, d), (2 * d, 32), (32, d)]]
    mlp = lambda x, a, b: jnp.tanh(x @ a) @ b
    def gn(nodes, edges, senders, receivers):
        g = jraph.GraphsTuple(nodes=nodes, edges=edges, senders=senders, receivers=receivers, globals=None,
                              n_node=jnp.array([nn_]), n_edge=jnp.array([ne]))
        net = jraph.GraphNetwork(update_edge_fn=lambda e, s, rr, gl: mlp(jnp.concatenate([e, s, rr], -1), W[0], W[1]),
                                 update_node_fn=lambda n, se, re, gl: mlp(jnp.concatenate([n, re], -1), W[2], W[3]))
        out = net(g)
        return out.nodes, out.edges
    edge = lambda nodes, e, s, rr: mlp(jnp.concatenate([e, nodes[s], nodes[rr]]), W[0], W[1])  # (summed per receiver)
    node = lambda n, re: mlp(jnp.concatenate([n, re]), W[2], W[3])
    args = (r.standard_normal((nn_, d)), r.standard_normal((ne, d)), r.integers(0, nn_, ne), r.integers(0, nn_, ne))
    return tagged(gn, stage(edge, (None, 0, 0, 0), atomic=d), stage(node, (0, 0), (args[0], np.zeros((nn_, d))))), args


def e3nn_tp():  # NequIP-like message: spherical harmonics of edge vectors (x) neighbour features, per edge
    import e3nn_jax as e3nn
    ne = 100_000 * S
    irr = e3nn.Irreps("8x0e + 8x1o")
    def msg(vec, feat):
        sh = e3nn.spherical_harmonics("0e + 1o + 2e", vec, normalize=True)
        f = e3nn.IrrepsArray(irr, feat)
        tp = e3nn.tensor_product(sh, f, filter_ir_out="0e + 1o + 2e")
        return tp.array * jnp.exp(-jnp.sum(vec ** 2, -1, keepdims=True))
    return tagged(msg, stage(msg, (0, 0))), (r.standard_normal((ne, 3)), r.standard_normal((ne, irr.dim)))


def gp_kernel():  # GP: ARD Matern-5/2 kernel matrices (vmap of vmap) and the predictive mean at 20 000 test points
    def k(x, y, ls):
        d = jnp.sqrt(jnp.sum(((x - y) / ls) ** 2) + 1e-12)
        return (1 + jnp.sqrt(5.0) * d + 5.0 / 3.0 * d ** 2) * jnp.exp(-jnp.sqrt(5.0) * d)
    def predict(Xs, X, alpha, ls):
        K = jax.vmap(jax.vmap(k, (None, 0, None)), (0, None, None))(Xs, X, ls)
        return K @ alpha
    point = lambda xs, X, alpha, ls: jax.vmap(k, (None, 0, None))(xs, X, ls) @ alpha
    return tagged(predict, stage(point, (0, None, None, None))), (r.uniform(0, 1, (20_000 * S, 3)), r.uniform(0, 1, (500, 3)),
                                                                  r.standard_normal(500), np.array([0.3, 0.5, 0.2]))


def evosax_cmaes():  # CMA-ES on Rastrigin, 20 generations, vmapped over 64 independent runs (popsize 32, dim 5)
    from evosax.algorithms import CMA_ES
    dim, pop = 5, 32
    es = CMA_ES(population_size=pop, solution=jnp.zeros(dim))
    params = es.default_params
    f = lambda x: 10 * dim + jnp.sum(x ** 2 - 10 * jnp.cos(2 * jnp.pi * x))
    def run(key, x0):
        k0, k1 = jax.random.split(key)
        st = es.init(k0, x0, params)
        def gen(st, k):
            ka, kt = jax.random.split(k)
            xs, st = es.ask(ka, st, params)
            st, _ = es.tell(kt, xs, jax.vmap(f)(xs), st, params)
            return st, None
        st, _ = jax.lax.scan(gen, st, jax.random.split(k1, 20))
        return st.best_fitness, st.mean
    R = 64 * S
    return vmapped(run), (jax.random.split(jax.random.PRNGKey(0), R), r.uniform(-3, 3, (R, dim)))


def jaxopt_lm():  # Levenberg-Marquardt curve fit (y = a exp(-b t) + c) per curve, 4096 curves of 30 points
    import jaxopt
    t = np.linspace(0, 4, 30)
    def fit(y):
        res = lambda p, y: p[0] * jnp.exp(-p[1] * t) + p[2] - y
        return jaxopt.LevenbergMarquardt(res, maxiter=50, tol=1e-10 if X64 else 1e-6).run(jnp.array([1.0, 1.0, 0.0]), y).params
    n = 4096 * S
    P = np.stack([r.uniform(0.5, 2, n), r.uniform(0.3, 2, n), r.uniform(-0.5, 0.5, n)], 1)
    Y = P[:, :1] * np.exp(-P[:, 1:2] * t) + P[:, 2:] + 0.01 * r.standard_normal((n, 30))
    return vmapped(fit), (Y,)


T = 200  # steps of an environment's rollout


def _env(lib, name):
    """(reset(key) -> state, step(key, state) -> (state, reward)) of a library's environment, with random actions; where
    the library does not reset finished episodes itself (pgx, jumanji), they start anew from reset's state."""
    sel = lambda c, a, b: jax.tree.map(lambda x, y: jnp.where(c, x, y), a, b)
    if lib == "gymnax":
        import gymnax
        env, params = gymnax.make(name)
        space = env.action_space(params)

        def step(k, s):
            ka, ks = jax.random.split(k)
            _, s, rew, *_ = env.step(ks, s, space.sample(ka), params)
            return s, rew
        return (lambda k: env.reset(k, params)[1]), step
    if lib == "pgx":
        import pgx
        env = pgx.make(name)

        def step(k, s):
            ka, ks, kr = jax.random.split(k, 3)
            s = env.step(s, jax.random.categorical(ka, jnp.where(s.legal_action_mask, 0.0, -jnp.inf)), ks)
            return sel(s.terminated | s.truncated, env.init(kr), s), s.rewards.sum()
        return env.init, step
    if lib == "jumanji":
        import jumanji
        env = jumanji.make(name)

        def step(k, s):
            ka, kr = jax.random.split(k)
            nv = jnp.asarray(env.action_spec.num_values)
            s, ts = env.step(s, jax.random.randint(ka, nv.shape, 0, nv))
            return sel(ts.last(), env.reset(kr)[0], s), jnp.sum(ts.reward)
        return (lambda k: env.reset(k)[0]), step
    if lib == "xminigrid":
        import xminigrid
        env, params = xminigrid.make(name)

        def step(k, ts):
            ts = env.step(params, ts, jax.random.randint(k, (), 0, env.num_actions(params)))
            return ts, ts.reward
        return (lambda k: env.reset(params, k)), step
    if lib == "jaxmarl":
        import jaxmarl
        env = jaxmarl.make(name)

        def step(k, s):
            ks, *ka = jax.random.split(k, 1 + env.num_agents)
            _, s, rew, _, _ = env.step(ks, s, {a: env.action_space(a).sample(kk) for a, kk in zip(env.agents, ka)})
            return s, sum(rew[a] for a in env.agents)
        return (lambda k: env.reset(k)[1]), step
    from brax import envs  # brax: "ant/mjx", "ant/positional"
    ename, backend = name.split("/")
    env = envs.get_environment(ename, backend=backend)

    def step(k, s):
        s = env.step(s, jax.random.uniform(k, (env.action_size,), minval=-1.0, maxval=1.0))
        return s, s.reward
    return env.reset, step


def _rollouts(lib, name, n):
    """n environments over T steps, the sum of each one's rewards, as users write it: a scan over the steps with vmap
    over the environments in it, their keys split from the step's key (jax.random.split(k, n)); per sample, one
    environment's rollout (jax.vmap of it: the same work). float32, as these libraries run."""
    jax.config.update("jax_enable_x64", False)
    reset, step1 = _env(lib, name)

    def one(key):
        k0, k1 = jax.random.split(key)
        def step(c, k):
            s, rew = step1(k, c[0])
            return (s, c[1] + rew), None
        return jax.lax.scan(step, (reset(k0), jnp.zeros(())), jax.random.split(k1, T))[0][1]

    def rollout(key):
        k0, k1 = jax.random.split(key)
        def step(c, k):
            s, rew = jax.vmap(step1)(jax.random.split(k, n), c[0])
            return (s, c[1] + rew), None
        return jax.lax.scan(step, (jax.vmap(reset)(jax.random.split(k0, n)), jnp.zeros(n)), jax.random.split(k1, T))[0][1]
    return tagged(rollout, stage(one, (0,), (jax.random.split(jax.random.key(1), n),))), (jax.random.key(0),)


def gymnax_cartpole():
    return _rollouts("gymnax", "CartPole-v1", 8192 * S)


def gymnax_acrobot():
    return _rollouts("gymnax", "Acrobot-v1", 8192 * S)


def gymnax_asterix():
    return _rollouts("gymnax", "Asterix-MinAtar", 8192 * S)


def pgx_connect_four():
    return _rollouts("pgx", "connect_four", 8192 * S)


def pgx_othello():
    return _rollouts("pgx", "othello", 8192 * S)


def pgx_2048():
    return _rollouts("pgx", "2048", 8192 * S)


def jumanji_snake():
    return _rollouts("jumanji", "Snake-v1", 8192 * S)


def jumanji_tsp():
    return _rollouts("jumanji", "TSP-v1", 8192 * S)


def jumanji_tetris():
    return _rollouts("jumanji", "Tetris-v0", 8192 * S)


def jumanji_maze():  # (jax.jit: 30 s per call on 24 cores)
    return _rollouts("jumanji", "Maze-v0", 8192 * S)


def jumanji_cleaner():  # (jax.jit: 30 s per call on 24 cores)
    return _rollouts("jumanji", "Cleaner-v0", 8192 * S)


def xminigrid_doorkey():
    return _rollouts("xminigrid", "MiniGrid-DoorKey-8x8", 8192 * S)


def jaxmarl_mpe():
    return _rollouts("jaxmarl", "MPE_simple_spread_v3", 8192 * S)


def brax_ant_mjx():  # (1024 environments: jax.jit 9 s per call on 24 cores)
    return _rollouts("brax", "ant/mjx", 1024 * S)


def brax_ant_positional():
    return _rollouts("brax", "ant/positional", 1024 * S)


def _jaxley_cell():  # one Hodgkin-Huxley compartment under a step current, 20 ms at dt 0.025: the voltage over time
    import jaxley as jx
    from jaxley.channels import HH
    cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
    cell.insert(HH())
    cell.stimulate(jx.step_current(i_delay=1.0, i_dur=15.0, i_amp=0.1, delta_t=0.025, t_max=20.0), verbose=False)
    cell.record("v", verbose=False)
    cell.make_trainable("HH_gNa", verbose=False)
    cell.make_trainable("HH_gK", verbose=False)
    return lambda g: jx.integrate(cell, params=[{"HH_gNa": g[:1]}, {"HH_gK": g[1:]}], delta_t=0.025)[0]


def _conductances():
    n = 8192 * S
    return np.stack([r.uniform(0.05, 0.5, n), r.uniform(0.01, 0.1, n)], 1)


def jaxley_sweep():  # a parameter sweep: the voltage trace per pair of conductances
    return vmapped(_jaxley_cell()), (_conductances(),)


def jaxley_grad():  # training: loss and gradient per pair of conductances
    sim = _jaxley_cell()
    return vmapped(jax.value_and_grad(lambda g: jnp.mean(sim(g) ** 2))), (_conductances(),)


def _transit():  # a transit's limb-darkened light curve at time t for (period, radius, impact, eccentricity, omega)
    from jaxoplanet.light_curves import limb_dark_light_curve
    from jaxoplanet.orbits.keplerian import Central, System
    def flux(t, p):
        s = System(Central()).add_body(period=p[0], radius=p[1], impact_param=p[2], eccentricity=p[3],
                                       omega_peri=p[4], time_transit=0.0)
        return limb_dark_light_curve(s, jnp.array([0.1, 0.3]))(t)[0]
    return flux, np.linspace(-2.0, 2.0, 100_000 * S), np.array([3.0, 0.1, 0.4, 0.2, 0.5])


def jaxoplanet_flux():
    flux, t, p = _transit()
    return vmapped(flux, (0, None)), (t, p)


def jaxoplanet_grad():  # a fit's squared error per time and its gradient wrt the parameters
    flux, t, p = _transit()
    y = jax.jit(jax.vmap(flux, (0, None)))(t, p * 1.01)
    return vmapped(lambda t, y, p: jax.value_and_grad(lambda p: (flux(t, p) - y) ** 2)(p), (0, 0, None)), (t, y, p)


def _mesolve(H0, Hd, L, rho0, n):  # dynamiqs: the Lindblad equation per detuning (Tsit5), states at 20 times
    import dynamiqs as dq
    tsave = jnp.linspace(0.0, 10.0, 20)
    one = lambda d: dq.mesolve(d * H0 + Hd, L, rho0, tsave, progress_meter=False).states.to_jax()
    return vmapped(one), (np.linspace(-2.0, 2.0, n),)


def dynamiqs_qubit():  # a driven, damped qubit
    import dynamiqs as dq
    return _mesolve(0.5 * dq.sigmaz(), 0.5 * dq.sigmax(), [jnp.sqrt(0.1) * dq.sigmam()], dq.basis_dm(2, 1), 10_000 * S)


def dynamiqs_kerr():  # a driven Kerr oscillator with loss, 5 levels
    import dynamiqs as dq
    a = dq.destroy(5)
    return _mesolve(a.dag() @ a, 0.3 * (a + a.dag()) + 0.05 * a.dag() @ a.dag() @ a @ a, [jnp.sqrt(0.2) * a], dq.fock_dm(5, 0),
                    10_000 * S)


CASES = {f.__name__: f for f in (flax_mlp_infer, flax_train_step, eqx_pinn, diffrax_ode, diffrax_stiff, numpyro_logp,
                                 blackjax_nuts, jaxmd_lj, optx_root, optx_minimise, lineax_solve, tfp_gmm_logprob,
                                 distrax_tanh_policy, haiku_per_example_grads, gymnax_rollout, jraph_gnn, e3nn_tp, gp_kernel,
                                 evosax_cmaes, jaxopt_lm,
                                 gymnax_cartpole, gymnax_acrobot, gymnax_asterix, pgx_connect_four, pgx_othello, pgx_2048,
                                 jumanji_snake, jumanji_tsp, jumanji_tetris, jumanji_maze, jumanji_cleaner,
                                 xminigrid_doorkey, jaxmarl_mpe, brax_ant_mjx, brax_ant_positional, jaxley_sweep,
                                 jaxley_grad, jaxoplanet_flux, jaxoplanet_grad, dynamiqs_qubit, dynamiqs_kerr)}


def settled(g, args, limit=400):
    """kernelax.jit's trials decided: until then its calls alternate between its programs (a timed call could be XLA's)."""
    for _ in range(limit):
        if g.report()[0]["status"] not in ("trial", "building"):
            return
        jax.block_until_ready(g(*args))


def timed(fs, budget=0.5, rounds=3, most=20):
    """The best time per program of fs (name -> (f, args)): alternately, each called once untimed before its timed call."""
    ts = {k: [] for k in fs}
    while len(ts[next(iter(ts))]) < rounds or min(map(sum, ts.values())) < budget and len(ts[next(iter(ts))]) < most:
        for k, (f, a) in fs.items():
            jax.block_until_ready(f(*a))
            t = time.perf_counter(); jax.block_until_ready(f(*a)); ts[k].append(time.perf_counter() - t)
    return {k: min(v) for k, v in ts.items()}


def rel_err(out, ref):
    errs = [float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))) / max(1e-300, float(np.max(np.abs(np.asarray(b, np.float64))))))
            for a, b in zip(jax.tree.leaves(out), jax.tree.leaves(ref)) if np.size(b) and np.issubdtype(np.asarray(b).dtype, np.number)]
    return max(errs) if errs else None


def measure(name):
    """One case: jax.jit(f) against kernelax.jit(f), and per stage kernelax.jit(jax.vmap(g)) (unless that is f) and
    jax.jit(jax.vmap(kernelax.kernel(g))): first calls (empty cache: as the caller's KERNELAX_CACHE_DIR), then times."""
    from kernelax import kernel
    f, args = CASES[name]()
    res, fs = dict(case=name, platform=jax.devices()[0].platform), {}

    def first(key, h, a):
        t0 = time.perf_counter()
        try:
            out = jax.block_until_ready(h(*a))
        except Exception as e:  # noqa: BLE001 (reported: a stage that kernelax.kernel does not cover)
            res[f"{key}_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            return None
        res[f"first_{key}_s"] = time.perf_counter() - t0
        fs[key] = (h, a)
        return out
    ref = first("jax", jax.jit(f), args)
    g = kernelax_jit(f)
    t0 = time.perf_counter()
    res["rel_err"] = rel_err(first("kernelax", g, args), ref)
    g.wait(); res["ready_s"] = time.perf_counter() - t0  # (kernels in use: K = 1; the variants K > 1 follow)
    g.wait(variants=True); res["ready_all_s"] = time.perf_counter() - t0
    autos = [(g, args)]
    for i, st in enumerate(f.per_sample):
        a = args if st["args"] is None else st["args"]
        if not getattr(f, "vmapped", False):
            gv = kernelax_jit(jax.vmap(st["g"], st["in_axes"]))
            if first(f"vmap{i}", gv, a) is not None:
                gv.wait(variants=True)
                autos.append((gv, a))
        first(f"kernel{i}", jax.jit(jax.vmap(kernel(st["g"]), st["in_axes"])), a)
    for h, a in autos:
        settled(h, a)
    res.update({f"{k}_ms": 1e3 * t for k, t in timed(fs).items()})
    res["speedup"] = res["jax_ms"] / res["kernelax_ms"] if "kernelax_ms" in res else None
    plan = g.report()[0]
    res.update(status=f"{plan['status']} {plan.get('why', '')}".strip(), N=plan.get("N"),
               regions=[(x["eqns"], x["chosen"], x.get("work")) for x in plan.get("regions", [])],
               errors=[e[:120] for e in plan.get("errors", [])])
    return res


def run_all(out, names, timeout=3600):
    """Each case in a fresh process with an empty kernel cache, its line appended to `out` (cases there skipped)."""
    import subprocess, tempfile
    done = {json.loads(ln).get("case") for ln in open(out)} if os.path.exists(out) else set()
    for name in names:
        if name in done:
            continue
        with tempfile.TemporaryDirectory() as cache:
            try:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), name], capture_output=True, text=True, timeout=timeout,
                                   env=dict(os.environ, KERNELAX_CACHE_DIR=cache, KERNELAX_EXIT_WAIT="0"))
                line = next((ln for ln in p.stdout.splitlines() if ln.startswith("{")), None) or json.dumps(
                    dict(case=name, error=((p.stderr.strip().splitlines() or ["no output"])[-1])[:300]))
            except subprocess.TimeoutExpired:
                line = json.dumps(dict(case=name, error=f"timeout ({timeout} s)"))
        with open(out, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)


def compare(old, new):
    """new against old per case: times (ratios; * where more than 5 % slower, the first calls 20 %), status, errors."""
    load = lambda p: {d["case"]: d for d in map(json.loads, open(p)) if "case" in d}
    a, b = load(old), load(new)
    print("| case | jax.jit ms | kernelax.jit ms | new/old | vmap | kernel | first call | status |\n|---|---:|---:|---:|---:|---:|---:|---|")
    for c in b:
        o, n = a.get(c, {}), b[c]
        ratio = lambda k, lim: (f"{n[k] / o[k]:.2f}" + ("*" if n[k] > (1 + lim) * o[k] else "")) if o.get(k) and n.get(k) else "-"
        status = n.get("status", n.get("error", "")) + ("" if n.get("status") == o.get("status") else f" (was: {o.get('status', o.get('error', '-'))})")
        errs = "; ".join(f"{k}: {v[:60]}" for k, v in n.items() if k.endswith("error") and v != o.get(k))
        print(f"| {c} | {n.get('jax_ms', float('nan')):.2f} | {n.get('kernelax_ms', float('nan')):.2f} | {ratio('kernelax_ms', 0.05)} | "
              f"{ratio('vmap0_ms', 0.05)} | {ratio('kernel0_ms', 0.05)} | {ratio('first_kernelax_s', 0.2)} | {status}{' ' + errs if errs else ''} |")


if __name__ == "__main__":
    if sys.argv[1] == "--all":  # --all OUT [case ...]
        run_all(sys.argv[2], sys.argv[3:] or list(CASES))
    elif sys.argv[1] == "--compare":  # --compare OLD NEW
        compare(sys.argv[2], sys.argv[3])
    else:
        print(json.dumps(measure(sys.argv[1])), flush=True)
