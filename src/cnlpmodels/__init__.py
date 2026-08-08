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
import sys

import numpy as np

__all__ = ["CModel", "lib", "load", "schema", "set_path", "solve_ipopt"]

from .ipopt import solve_ipopt

_c_int = ctypes.c_int32
_c_dbl = ctypes.c_double
_pd = ctypes.POINTER(_c_dbl)
_pi = ctypes.POINTER(_c_int)


_PATHS = []
_LIBS = {}


def set_path(*dirs):
    """Set the library search path for name-based loading (`lib("acopf")`,
    `CModel("acopf", ...)`). Initialized from the colon-separated
    `CNLPMODELS_PATH` environment variable; calling this replaces it."""
    _PATHS[:] = [os.fspath(d) for d in dirs]
    _LIBS.clear()
    return list(_PATHS)


def _paths():
    if not _PATHS:
        env = os.environ.get("CNLPMODELS_PATH", "")
        if env:
            _PATHS[:] = env.split(":")
    return _PATHS


def lib(name):
    """Resolve `lib<name>.so` against the search path — also accepting the
    `<dir>/<name>/lib/` and `<dir>/lib/` layouts `compile_library` produces —
    load it, and cache the handle by name."""
    if name not in _LIBS:
        ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
        fname = f"lib{name}{ext}"
        for d in _paths():
            for cand in (os.path.join(d, fname),
                         os.path.join(d, name, "lib", fname),
                         os.path.join(d, "lib", fname)):
                if os.path.isfile(cand):
                    _LIBS[name] = load(cand)
                    return _LIBS[name]
        where = ":".join(_paths()) or "empty — call set_path() or set CNLPMODELS_PATH"
        raise FileNotFoundError(f"{fname} not found on the cnlpmodels path ({where})")
    return _LIBS[name]


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


def schema(lib, *, prefix="rec"):
    """The library's data schema (ABI v2), as published by `<prefix>_schema`."""
    import json
    fn = getattr(lib, f"{prefix}_schema")
    fn.restype = _c_int
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint8), _c_int]
    need = fn(ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8)), 0)
    buf = (ctypes.c_uint8 * int(need))()
    fn(buf, need)
    return json.loads(bytes(buf).decode())


def _as_col(v, kind):
    dt = np.float64 if kind == "f64" else np.int64
    a = np.ascontiguousarray(v, dtype=dt)
    if a.ndim != 1:
        raise ValueError("columns and arrays must be one-dimensional")
    return a


def _fill_data(lib, prefix, data):
    """Fill a builder from a dict: tables are dicts of columns (numpy arrays),
    arrays are numpy arrays, scalars are numbers. Validated against the
    library's own schema."""
    sch = schema(lib, prefix=prefix)
    fields = {f["name"]: f for f in sch["fields"]}
    unknown = set(data) - set(fields)
    missing = set(fields) - set(data)
    if unknown or missing:
        raise ValueError(f"data keys do not match schema: unknown={sorted(unknown)}, "
                         f"missing={sorted(missing)}")

    def fn(name, argtypes):
        g = getattr(lib, f"{prefix}_{name}")
        g.restype = _c_int
        g.argtypes = argtypes
        return g

    begin = fn("data_begin", [])
    b = begin()
    if b <= 0:
        raise RuntimeError("data_begin failed")
    for name, spec in fields.items():
        v = data[name]
        if spec["kind"] == "scalar":
            if spec["type"] == "f64":
                _check(fn("set_scalar_f64", [_c_int, ctypes.c_char_p, _c_dbl])(
                    b, name.encode(), float(v)), f"set_scalar_f64({name})")
            else:
                _check(fn("set_scalar_i64", [_c_int, ctypes.c_char_p, ctypes.c_longlong])(
                    b, name.encode(), int(v)), f"set_scalar_i64({name})")
        elif spec["kind"] == "array":
            a = _as_col(v, spec["type"])
            setter = f"set_array_{spec['type']}"
            ptr_t = _pd if spec["type"] == "f64" else ctypes.POINTER(ctypes.c_longlong)
            _check(fn(setter, [_c_int, ctypes.c_char_p, ptr_t, _c_int])(
                b, name.encode(), a.ctypes.data_as(ptr_t), len(a)), f"{setter}({name})")
        else:  # table
            cols = {c["name"]: c for c in spec["columns"]}
            if set(v) != set(cols):
                raise ValueError(f"table {name!r}: columns {sorted(v)} != schema {sorted(cols)}")
            lens = {len(np.asarray(col)) for col in v.values()}
            if len(lens) != 1:
                raise ValueError(f"table {name!r}: column lengths differ: {lens}")
            for cname, cspec in cols.items():
                a = _as_col(v[cname], cspec["type"])
                setter = f"set_col_{cspec['type']}"
                ptr_t = _pd if cspec["type"] == "f64" else ctypes.POINTER(ctypes.c_longlong)
                _check(fn(setter, [_c_int, ctypes.c_char_p, ctypes.c_char_p, ptr_t, _c_int])(
                    b, name.encode(), cname.encode(), a.ctypes.data_as(ptr_t), len(a)),
                    f"{setter}({name}.{cname})")
    ready = fn("data_ready", [_c_int])(b)
    if ready != 1:
        raise RuntimeError("library reports data incomplete after all fields were set")
    mid = fn("new_from_data", [_c_int])(b)
    if mid <= 0:
        raise RuntimeError("new_from_data failed")
    return mid


def _instantiate(lib, prefix, args):
    """`args` -> model id. No shape is privileged: an int uses `<prefix>_new`
    when the library exports it (and the builder otherwise); a tuple/list
    binds positionally to the schema's field order; a dict binds by name;
    None (the default) means no instance data — valid when the schema
    requires none."""
    if isinstance(args, bool):
        raise TypeError("args must be an int, tuple, dict, or None — not bool")
    if isinstance(args, int):
        try:
            new = _f(lib, prefix, "new", [_c_int])
        except AttributeError:
            return _instantiate(lib, prefix, (args,))
        mid = new(_c_int(int(args)))
        if mid <= 0:
            raise RuntimeError(f"{prefix}_new({args}) failed (returned {mid})")
        return mid
    if args is None:
        args = {}
    if isinstance(args, (tuple, list)):
        try:
            sch = schema(lib, prefix=prefix)
        except AttributeError:
            raise RuntimeError(
                "this library has no builder surface; positional args need "
                "a published schema") from None
        names = [f["name"] for f in sch["fields"]]
        if len(names) != len(args):
            raise ValueError(
                f"positional args have {len(args)} entries but the schema "
                f"declares {len(names)} fields: {names}")
        return _fill_data(lib, prefix, dict(zip(names, args)))
    if isinstance(args, dict):
        try:
            return _fill_data(lib, prefix, args)
        except AttributeError:
            raise RuntimeError(
                "this library has no builder surface; instantiating it "
                "requires integer args") from None
    raise TypeError(f"unsupported args shape: {type(args).__name__}")


class CModel:
    """A model instance from a loaded library.

        lib = cnlpmodels.load("liblv.so")
        m = cnlpmodels.CModel(lib, args=1000, prefix="lv")     # int -> <prefix>_new
        m = cnlpmodels.CModel(lib, args=(1000,), prefix="lv")  # positional vs schema
        m = cnlpmodels.CModel(lib, args={"n": 1000}, prefix="lv")  # by name

    `args` defaults to None (no instance data; valid when the library's
    schema requires none). Any number of instances may coexist per library.
    """

    def __init__(self, lib, *, args=None, prefix=None):
        if isinstance(lib, str):
            prefix = prefix if prefix is not None else lib
            lib = globals()["lib"](lib)
        prefix = prefix if prefix is not None else "rec"
        self._fn = {
            name: _f(lib, prefix, name, argtypes)
            for name, argtypes in (
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
        self._id = _instantiate(lib, prefix, args)
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
