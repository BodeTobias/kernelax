# Copyright (C) 2026 Tobias Bode
# SPDX-License-Identifier: MIT

"""Operations on constants by NumPy where the results are XLA's to the bit: the planner's constant folding and the
frontend's concrete values bind thousands of them, and JAX's eager binds cost ~100 us each (with a cold compilation
cache a compile each)."""

import numpy as np

_NP_EW = {"add": np.add, "add_any": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "neg": np.negative,
          "max": np.maximum, "min": np.minimum, "eq": np.equal, "ne": np.not_equal, "lt": np.less, "le": np.less_equal,
          "gt": np.greater, "ge": np.greater_equal, "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
          "not": np.invert}


def _plain(dtype):  # integers, bools, 32- and 64-bit floats
    return dtype.kind in "biu" or dtype.kind == "f" and dtype.itemsize >= 4


def _subnormal(x):  # whether a float array has entries XLA's CPU flushes to zero (FTZ, DAZ)
    return x.dtype.kind == "f" and bool(np.any((x != 0) & (np.abs(x) < np.finfo(x.dtype).tiny)))


def numpy_bind(e, args):
    """The results of an operation on constants by NumPy, where they are XLA's to the bit, or None (JAX's eager binds
    cost ~100 us each; folding a program's constants made thousands): data movement, elementwise arithmetic,
    comparisons and selects without subnormal values (XLA's CPU flushes them), iota, exact conversions."""
    name, p = e.primitive.name, e.params
    if not all(_plain(a.dtype) for a in args) or not all(_plain(np.dtype(v.aval.dtype)) for v in e.outvars):
        return None  # (16-bit floats: NumPy's ml_dtypes round otherwise; keys)
    if name in _NP_EW and all(isinstance(a, np.ndarray) for a in args) and len({a.dtype for a in args}) == 1 and \
            not (name == "div" and args[0].dtype.kind != "f") and not any(map(_subnormal, args)):
        with np.errstate(all="ignore"):
            out = np.asarray(_NP_EW[name](*args), e.outvars[0].aval.dtype)
        return None if _subnormal(out) else [out]
    if name == "select_n" and not any(map(_subnormal, args[1:])):
        return [np.choose(args[0].astype(np.int64), args[1:])] if args[0].shape == args[1].shape or not args[0].shape else None
    if name == "iota" and all(isinstance(d, int) for d in p["shape"]):
        shape = [1] * len(p["shape"])
        shape[p["dimension"]] = p["shape"][p["dimension"]]
        return [np.broadcast_to(np.arange(p["shape"][p["dimension"]], dtype=p["dtype"]).reshape(shape), tuple(p["shape"]))]
    if name == "convert_element_type" and (args[0].dtype.kind in "biu" or np.dtype(p["new_dtype"]).kind == "f" == args[0].dtype.kind):
        out = args[0].astype(p["new_dtype"])  # (integers and bools to anything; floats to floats: rounded as XLA's)
        return None if _subnormal(args[0]) or _subnormal(out) else [out]
    return _moved(e, args)


def concrete(eqn, vals):
    """vals (an equation's concrete operands) as arrays of their avals' dtypes, or None (keys, other dtypes)."""
    try:
        return [np.asarray(v, np.dtype(x.aval.dtype)) for v, x in zip(vals, eqn.invars)]
    except (TypeError, ValueError):
        return None


def _moved(e, args):
    """The results of a data movement on constants, by NumPy (the same entries), or None."""
    name, p = e.primitive.name, e.params
    if name == "broadcast_in_dim" and len(args) == 1 and all(isinstance(d, int) for d in p["shape"]):
        shape = [1] * len(p["shape"])
        for i, d in enumerate(p["broadcast_dimensions"]):
            shape[d] = args[0].shape[i]
        return [np.broadcast_to(args[0].reshape(shape), tuple(p["shape"]))]
    if name == "reshape" and p.get("dimensions") is None and len(args) == 1:
        return [args[0].reshape(tuple(p["new_sizes"]))]
    if name == "squeeze":
        return [np.squeeze(args[0], tuple(p["dimensions"]))]
    if name == "transpose":
        return [np.transpose(args[0], tuple(p["permutation"]))]
    if name == "slice":
        st = p["strides"] or (1,) * args[0].ndim
        return [args[0][tuple(slice(a, b, c) for a, b, c in zip(p["start_indices"], p["limit_indices"], st))]]
    if name == "concatenate":
        return [np.concatenate(args, axis=p["dimension"])]
    return None
