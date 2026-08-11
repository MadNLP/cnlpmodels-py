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
    `CModel("@acopf", ...)`). Initialized from the colon-separated
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


# `@name` resolves on the search path; any other string is a filesystem path,
# relative to the current directory or absolute, exactly as written.
def _is_name(spec):
    return spec.startswith("@")


# `librosen.so` → `rosen`; a bundle directory or a file not following the
# `lib<name>` convention keeps its stem, and `prefix=` remains the override
# for libraries whose symbols are named independently of the file.
def _default_prefix(spec):
    if _is_name(spec):
        return spec[1:]
    base = os.path.splitext(os.path.basename(spec.rstrip("/")))[0]
    return base[3:] if base.startswith("lib") and len(base) > 3 else base


# A path names a shared library directly, or a bundle DIRECTORY — the layout
# `compile_library` produces — in which case the library is found inside it.
# Returned ABSOLUTE: `dlopen` treats a slash-free relative like `qp.so` as a
# soname to search the system path for, not as a file in the current
# directory — it resolved locally only by environmental accident and failed
# in CI.
def _resolve_path(spec):
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    if os.path.isdir(spec):
        ext = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
        fname = f"lib{os.path.basename(spec.rstrip('/'))}{ext}"
        for cand in (os.path.join(spec, "lib", fname), os.path.join(spec, fname)):
            if os.path.isfile(cand):
                return os.path.abspath(cand)
        raise FileNotFoundError(
            f"no shared library in {spec} (tried lib/{fname} and {fname})")
    raise FileNotFoundError(f"no shared library at {spec}")


# Cached like name-resolution: absolute-path keys cannot collide with bare
# names, so the one registry serves both.
def _resolve_spec(spec):
    if _is_name(spec):
        return lib(spec[1:])
    key = os.path.abspath(spec)
    if key not in _LIBS:
        _LIBS[key] = load(_resolve_path(spec))
    return _LIBS[key]


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


# The schema says what a slot holds; a value of the wrong kind is refused
# rather than coerced into it. Arguments are positional, so a transposition is
# the mistake to expect — and `int(2.0)` would swallow one silently, building a
# different model from the one asked for.
def _as_scalar(name, v, kind):
    if isinstance(v, (bool, np.bool_)):
        raise TypeError(f"field {name!r}: a bool is not a model argument")
    if kind == "i64":
        if not isinstance(v, (int, np.integer)):
            raise TypeError(f"field {name!r} is an int64 in the library's schema, "
                            f"got {type(v).__name__}")
        return int(v)
    if not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(f"field {name!r} is a float64 in the library's schema, "
                        f"got {type(v).__name__}")
    return float(v)


def _as_col(v, kind, name="value"):
    a = np.asarray(v)
    if kind == "i64" and not np.issubdtype(a.dtype, np.integer):
        raise TypeError(f"{name} is an int64 array in the library's schema, "
                        f"got dtype {a.dtype}")
    if kind == "f64" and not np.issubdtype(a.dtype, np.number):
        raise TypeError(f"{name} is a float64 array in the library's schema, "
                        f"got dtype {a.dtype}")
    dt = np.float64 if kind == "f64" else np.int64
    a = np.ascontiguousarray(a, dtype=dt)
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
                    b, name.encode(), _as_scalar(name, v, "f64")),
                    f"set_scalar_f64({name})")
            else:
                _check(fn("set_scalar_i64", [_c_int, ctypes.c_char_p, ctypes.c_longlong])(
                    b, name.encode(), _as_scalar(name, v, "i64")),
                    f"set_scalar_i64({name})")
        elif spec["kind"] == "array":
            a = _as_col(v, spec["type"], name)
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
                a = _as_col(v[cname], cspec["type"], f"{name}.{cname}")
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


def _bind(lib, prefix, args):
    """Positional arguments against the schema's field order. No arguments at
    all is the "no instance data" case, and is checked the same way — a schema
    declaring fields says so here, rather than the library reporting itself
    incomplete several calls later."""
    if not hasattr(lib, f"{prefix}_data_begin"):
        raise RuntimeError(
            "this library has no builder surface: it instantiates from a "
            "single integer, CModel(lib, n)")
    try:
        sch = schema(lib, prefix=prefix)
    except AttributeError:
        raise RuntimeError(
            f"{prefix}_data_begin exists but {prefix}_schema does not, so "
            "there is nothing to bind the arguments against") from None
    names = [f["name"] for f in sch["fields"]]
    if len(names) != len(args):
        raise ValueError(
            f"given {len(args)} argument{'' if len(args) == 1 else 's'} but the "
            f"library's schema declares {len(names)} "
            f"field{'' if len(names) == 1 else 's'}: {names}")
    return dict(zip(names, args))


def _instantiate(lib, prefix, args):
    """Positional arguments -> model id.

    One value per schema field, in the order the library publishes them — the
    same convention the producer side uses (`ExaModel(core, arg1, arg2, ...)`,
    `compile_library(out, core, arg1, ...)`), so a compiled model is consumed
    the way it was written. A lone integer takes the one-knob `<prefix>_new(n)`
    when the library exports it and falls through to the builder otherwise; the
    two surfaces are disjoint in practice, since a producer emits `P_new` and
    no builder precisely when the schema is a single integer scalar."""
    if any(isinstance(a, bool) for a in args):
        raise TypeError("a bool is not a model argument")
    if len(args) == 1 and isinstance(args[0], (int, np.integer)):
        try:
            new = _f(lib, prefix, "new", [_c_int])
        except AttributeError:
            pass
        else:
            mid = new(_c_int(int(args[0])))
            if mid <= 0:
                raise RuntimeError(f"{prefix}_new({args[0]}) failed (returned {mid})")
            return mid
    # A FIXED model — a library whose `<prefix>_nargs()` reports 0 — consumes
    # no instantiation data: `<prefix>_new` keeps its one-integer C signature
    # but ignores the value, so no arguments at all is the natural call. A
    # library that does not declare its arity keeps the old behaviour and
    # falls through to the builder surface (where a schema with no fields is
    # the other legitimate "no instance data" case).
    if not args:
        try:
            nargs = _f(lib, prefix, "nargs", [])
        except AttributeError:
            pass
        else:
            if int(nargs()) == 0:
                new = _f(lib, prefix, "new", [_c_int])
                mid = new(_c_int(0))
                if mid <= 0:
                    raise RuntimeError(f"{prefix}_new(0) failed (returned {mid})")
                return mid
    return _fill_data(lib, prefix, _bind(lib, prefix, args))


class CModel:
    """A model instance from a loaded library.

        lib = cnlpmodels.load("liblv.so")
        m = cnlpmodels.CModel(lib, 1000, prefix="lv")          # lv_new(1000)
        m = cnlpmodels.CModel("@acopf", bus, vmin, 100.0)      # search path
        m = cnlpmodels.CModel("rosen", 1000)                   # ./rosen (file or bundle dir)
        m = cnlpmodels.CModel("/opt/models/rosen", 1000)       # full path

    The arguments are the values the model is instantiated with — one per field
    of the library's schema, positionally, in the order the library publishes
    them, which is the same spelling the producer side uses
    (`ExaModel(core, arg1, ...)`, `compile_library(out, core, arg1, ...)`).

    Each value is a **number**, a **numpy/sequence array of numbers**, or a
    **table** (a dict of equal-length columns, sent to the ABI v2 builder column
    by column and validated against the library's schema). A lone integer uses
    `<prefix>_new(n)` when the library exports it and the builder otherwise;
    with no arguments at all the model is built from no instance data, which is
    valid when the schema declares no fields.

    Any number of instances may coexist per library.
    """

    def __init__(self, lib, *args, prefix=None):
        if isinstance(lib, str):
            prefix = prefix if prefix is not None else _default_prefix(lib)
            lib = _resolve_spec(lib)
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
