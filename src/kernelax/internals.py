# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""What kernelax takes from JAX's internals (jax._src or private attributes), in one place. None of these has a
public equivalent in JAX 0.12.1 (checked: jax, jax.core, jax.extend, jax.interpreters). The primitives' expansions are
imported at their first use, so that a JAX version that moved one fails only for the primitive that needs it, naming
it (kernelax.jit then runs that program in XLA; its report() says why). Used at import: JAX's config, the trace
outside any transformation (_EVAL) and the closed-form determinants (closed_form_dets). frontend._jax_own also
recognizes some of JAX's functions by their jax/_src source paths."""

import importlib

import jax
from jax._src import config as _config
from jax.extend import core as jcore

with jax.core.eval_context():
    _EVAL = type(jcore.find_top_trace(()))  # (the trace outside any transformation: EvalTrace)


def top_level():
    """Whether the caller runs outside any JAX trace (jit, vmap, grad, make_jaxpr; also inside
    jax.ensure_compile_time_eval), where kernelax may run its kernels directly."""
    return isinstance(jcore.find_top_trace(()), _EVAL)


def no_constant_folding():
    """A context without jax.ensure_compile_time_eval's eager constant folding: kernelax plans and calls programs
    traced there too (the sample index stays symbolic), as JAX's dispatch calls a primitive outside jit
    (jax._src.dispatch.xla_primitive_callable)."""
    return _config.eager_constant_folding(False)


def closed_form_dets():
    """JAX's determinants of 2 x 2 and 3 x 3 matrices (jnp.linalg.det calls them), custom_jvp functions whose rule
    is read at every call: frontend replaces their rule, JAX's _det_jvp, while kernelax traces. Empty where JAX changed
    them (tests/test_coverage.py then fails: derivatives of det stay JAX's, correct but large)."""
    from jax._src.numpy import linalg
    dets, rule = [getattr(linalg, f"_det_{k}x{k}", None) for k in (2, 3)], getattr(linalg, "_det_jvp", None)
    return dets if all(isinstance(f, jax.custom_jvp) and f.jvp is rule for f in dets) else []


def jit_cache_size(f):
    """A function of no arguments: the number of signatures the jax.jit function f has compiled (kernelax.jit: whether
    a call saw a new one; a marker in the traced function would miss signatures whose trace JAX had cached under an
    enclosing jit)."""
    return f._cache_size


def weakly_typed(x):
    """x as a weakly typed value of its dtype, as a Python scalar is under jax.jit (kernelax.kernel's arguments that
    were Python scalars: x_f32 * 0.5 stays float32); no public function makes a value weakly typed."""
    return jax_function("lax.lax", "_convert_element_type")(x, x.dtype, weak_type=True)


def key_impl(dtype):
    """The PRNG implementation of a key dtype (jax.random.key_impl takes key arrays, not their dtypes)."""
    return dtype._impl


def jax_function(module, *names):
    """JAX's jnp implementation of a primitive that XLA lowers through it (no public function computes it without
    binding the primitive again): the first of `names` in jax._src.`module` (one per JAX version that renamed it)."""
    try:
        found = importlib.import_module("jax._src." + module)
    except ImportError as e:
        raise NotImplementedError(f"kernelax: JAX {jax.__version__} has no jax._src.{module}, whose functions kernelax "
                                  "uses for this primitive (kernelax/internals.py)") from e
    for name in names:
        if hasattr(found, name):
            return getattr(found, name)
    raise NotImplementedError(f"kernelax: JAX {jax.__version__} has none of jax._src.{module}.{'/'.join(names)}, whose "
                              "expansion kernelax uses for this primitive (kernelax/internals.py)")
