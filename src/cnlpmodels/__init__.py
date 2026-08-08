"""Consume an NLP exposed by a C-ABI shared library as a numpy-backed model.

The producer is typically ExaModels' recorder compiled with
``ExaModels.compile_library`` (Julia), but any library implementing the ABI
works — see ``tests/fixtures/tinyqp.c`` for a complete plain-C reference.
This package needs **no Julia**: the library is self-contained and this
wrapper is ctypes + numpy.

C ABI (for a symbol ``prefix``; 1-based indices, lower-triangle Lagrangian
Hessian of ``obj_weight * f + y' c``, ``int32`` statuses, 0 = success)::

    <prefix>_new(n)               -> id > 0 (0 on failure)
    <prefix>_nvar/_ncon/_nnzj/_nnzh(id) -> int32
    <prefix>_meta(id, x0, lvar, uvar, lcon, ucon)         (double* x 5)
    <prefix>_obj(id, x, out)      / _grad(id, x, g) / _cons(id, x, c)
    <prefix>_jac_structure(id, rows, cols)   (int32*, 1-based)
    <prefix>_jac(id, x, vals)
    <prefix>_hess_structure(id, rows, cols)  (int32*, 1-based)
    <prefix>_hess(id, x, y, obj_weight, vals)

Model protocol (duck-typed, shared with ``examodels``): attributes ``nvar``,
``ncon``, ``nnzj``, ``nnzh``, ``x0``, ``lvar``, ``uvar``, ``lcon``, ``ucon``;
methods ``obj(x)``, ``grad(x)``, ``cons(x)``, ``jac_structure()``,
``jac(x)``, ``hess_structure()``, ``hess(x, y, obj_weight=1.0)``. Structure
indices are returned 0-based here, matching Python convention; the 1-based
shift happens inside this wrapper.
"""
import ctypes
import os

import numpy as np

__all__ = ["load", "CModel", "solve_ipopt"]

from .ipopt import solve_ipopt  # noqa: E402

_c_int = ctypes.c_int32
_c_dbl = ctypes.c_double
_pd = ctypes.POINTER(_c_dbl)
_pi = ctypes.POINTER(_c_int)


def load(path):
    """dlopen the shared library (RTLD_LOCAL | RTLD_DEEPBIND, so its bundled
    runtime does not leak symbols into — or take them from — the process)."""
    flags = os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
    return ctypes.CDLL(os.fspath(path), mode=flags)


def _f(lib, prefix, name, argtypes):
    fn = getattr(lib, f"{prefix}_{name}")
    fn.restype = _c_int
    fn.argtypes = argtypes
    return fn


def _arr(a):
    return a.ctypes.data_as(_pd)


def _iarr(a):
    return a.ctypes.data_as(_pi)


def _check(st, what):
    if st != 0:
        raise RuntimeError(f"{what} returned nonzero status {st}")


class CModel:
    """A model instance of size `n` from a loaded library.

        lib = cnlpmodels.load("liblv.so")
        m = cnlpmodels.CModel(lib, n=1000, prefix="lv")

    Any number of instances may coexist per library.
    """

    def __init__(self, lib, *, n, prefix="rec"):
        self._fn = {
            name: _f(lib, prefix, name, argtypes)
            for name, argtypes in (
                ("new", [_c_int]),
                ("nvar", [_c_int]), ("ncon", [_c_int]),
                ("nnzj", [_c_int]), ("nnzh", [_c_int]),
                ("meta", [_c_int, _pd, _pd, _pd, _pd, _pd]),
                ("obj", [_c_int, _pd, _pd]),
                ("grad", [_c_int, _pd, _pd]),
                ("cons", [_c_int, _pd, _pd]),
                ("jac_structure", [_c_int, _pi, _pi]),
                ("jac", [_c_int, _pd, _pd]),
                ("hess_structure", [_c_int, _pi, _pi]),
                ("hess", [_c_int, _pd, _pd, _c_dbl, _pd]),
            )
        }
        self._id = self._fn["new"](_c_int(int(n)))
        if self._id <= 0:
            raise RuntimeError(f"{prefix}_new({n}) failed (returned {self._id})")
        self.nvar = int(self._fn["nvar"](self._id))
        self.ncon = int(self._fn["ncon"](self._id))
        self.nnzj = int(self._fn["nnzj"](self._id))
        self.nnzh = int(self._fn["nnzh"](self._id))
        self.x0 = np.zeros(self.nvar)
        self.lvar = np.zeros(self.nvar)
        self.uvar = np.zeros(self.nvar)
        self.lcon = np.zeros(self.ncon)
        self.ucon = np.zeros(self.ncon)
        _check(self._fn["meta"](self._id, _arr(self.x0), _arr(self.lvar),
                                _arr(self.uvar), _arr(self.lcon), _arr(self.ucon)),
               "meta")

    def _x(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (self.nvar,):
            raise ValueError(f"x must have shape ({self.nvar},), got {x.shape}")
        return x

    def obj(self, x):
        out = _c_dbl(0.0)
        _check(self._fn["obj"](self._id, _arr(self._x(x)), ctypes.byref(out)), "obj")
        return out.value

    def grad(self, x, out=None):
        g = out if out is not None else np.zeros(self.nvar)
        _check(self._fn["grad"](self._id, _arr(self._x(x)), _arr(g)), "grad")
        return g

    def cons(self, x, out=None):
        c = out if out is not None else np.zeros(self.ncon)
        _check(self._fn["cons"](self._id, _arr(self._x(x)), _arr(c)), "cons")
        return c

    def jac_structure(self):
        rows = np.zeros(self.nnzj, dtype=np.int32)
        cols = np.zeros(self.nnzj, dtype=np.int32)
        _check(self._fn["jac_structure"](self._id, _iarr(rows), _iarr(cols)),
               "jac_structure")
        return rows - 1, cols - 1        # 0-based, per Python convention

    def jac(self, x, out=None):
        v = out if out is not None else np.zeros(self.nnzj)
        _check(self._fn["jac"](self._id, _arr(self._x(x)), _arr(v)), "jac")
        return v

    def hess_structure(self):
        rows = np.zeros(self.nnzh, dtype=np.int32)
        cols = np.zeros(self.nnzh, dtype=np.int32)
        _check(self._fn["hess_structure"](self._id, _iarr(rows), _iarr(cols)),
               "hess_structure")
        return rows - 1, cols - 1        # 0-based, per Python convention

    def hess(self, x, y, obj_weight=1.0, out=None):
        y = np.ascontiguousarray(y, dtype=np.float64)
        if y.shape != (self.ncon,):
            raise ValueError(f"y must have shape ({self.ncon},), got {y.shape}")
        v = out if out is not None else np.zeros(self.nnzh)
        _check(self._fn["hess"](self._id, _arr(self._x(x)), _arr(y),
                                _c_dbl(float(obj_weight)), _arr(v)), "hess")
        return v
