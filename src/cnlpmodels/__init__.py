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

__all__ = ["BlockRef", "CModel", "argtype", "available_models", "lib", "load",
           "multipliers",
           "multipliers_L",
           "multipliers_U", "schema", "set_path", "solution", "solve_ipopt"]

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
    load it, and cache the handle by name.

    A leading `@` is accepted and ignored: this function only ever takes a
    name, and the sigil spelling travels from `CModel`'s string argument."""
    name = name.removeprefix("@")
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


def _require_model(lib, model):
    """A name this library does not carry is reported here, clearly.

    `_nvar` is the witness symbol: the ABI requires it of every model however
    the model is instantiated — unlike `_new` (absent from builder-only
    models) or `_data_begin` (absent from one-knob ones). Without this check a
    mistyped name surfaces as a raw ctypes `undefined symbol` error several
    layers down."""
    try:
        getattr(lib, f"{model}_nvar")
    except AttributeError:
        where = getattr(lib, "_name", "this library")
        raise ValueError(
            f"{where} carries no model named {model!r} "
            f"(it exports no {model}_nvar)"
        ) from None


def schema(lib, model=None, *, prefix="rec"):
    """The library's data schema, as published by `<prefix>_schema`.

    In a library carrying several models the schema is per model — name the
    one you want, `schema(lib, "acopf")`, exactly as in `CModel`."""
    import json
    if model is not None:
        _require_model(lib, model)
        prefix = model
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


def argtype(library, model):
    """What a model instantiates from: the types its entry point takes, in
    order, each optionally followed by `|` and a description.

        cnlpmodels.argtype("@grid", "acopf")
        # "int|arg1,Vector{f64}|v0,Table{i::int pd::f64}|bus"

    `""` is a model that takes nothing (or a library that publishes no
    signature); a lone "int" or "string" names the entry point; a longer list
    is a structured model taking one value per field. Every model answers,
    schema or not — which is why this exists rather than reading the schema,
    which only structured models have.
    """
    lib_ = _resolve_spec(library) if isinstance(library, str) else library
    try:
        fp = getattr(lib_, f"{model}_argtype")
    except AttributeError:
        return ""
    fp.restype = _c_int
    fp.argtypes = [ctypes.POINTER(ctypes.c_uint8), _c_int]
    need = int(fp(ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8)), _c_int(0)))
    if need <= 0:
        return ""
    buf = (ctypes.c_uint8 * need)()
    fp(buf, _c_int(need))
    return bytes(buf).decode()


def available_models(library):
    """The models a library carries, by name.

    Every other entry point needs a prefix to start from; this is the one
    question a caller can ask having only a path:

        for name in cnlpmodels.available_models("@grid"):
            m = cnlpmodels.CModel("@grid", name, data)

    A library that publishes no catalogue returns an empty list; selecting a
    model by name still works if you know the name.
    """
    lib_ = _resolve_spec(library) if isinstance(library, str) else library
    try:
        nf = lib_.cnlp_nmodels
    except AttributeError:
        return []
    nf.restype = _c_int
    nf.argtypes = []
    n = int(nf())
    if n <= 0:
        return []
    npf = lib_.cnlp_model_name
    npf.restype = _c_int
    npf.argtypes = [_c_int, ctypes.POINTER(ctypes.c_uint8), _c_int]
    out = []
    for k in range(n):
        need = int(npf(_c_int(k), ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8)), _c_int(0)))
        buf = (ctypes.c_uint8 * need)()
        npf(_c_int(k), buf, _c_int(need))
        out.append(bytes(buf).decode())
    return out


class BlockRef:
    """A named block of a compiled model — a variable, constraint or parameter.

    `offset` is 0-based and `dims` is the block's shape, so a slice of the
    solution can be reshaped the way the model was written rather than handed
    back flat. `index` is the library's own numbering, which is how parameters
    are addressed for `CModel.get_value` / `CModel.set_value`.
    """

    __slots__ = ("dims", "index", "kind", "length", "name", "offset")

    def __init__(self, name, kind, offset, length, dims, index):
        self.name, self.kind = name, kind
        self.offset, self.length, self.dims, self.index = offset, length, dims, index

    def __repr__(self):
        return (f"BlockRef({self.name!r}, {self.kind}, "
                f"{'×'.join(map(str, self.dims))} at {self.offset})")


def _slice(vec, b):
    v = np.asarray(vec)[b.offset:(b.offset + b.length)]
    return v.reshape(b.dims) if len(b.dims) > 1 else v


def solution(x, block):
    """The part of solution vector `x` belonging to variable `block`, reshaped
    to the block's own dimensions."""
    return _slice(x, block)


def multipliers(y, block):
    """The dual variables in `y` for constraint `block`."""
    return _slice(y, block)


def multipliers_L(z, block):
    """The lower-bound dual variables in `z` for variable `block`."""
    return _slice(z, block)


def multipliers_U(z, block):
    """The upper-bound dual variables in `z` for variable `block`."""
    return _slice(z, block)


def _read_layout(lib, prefix, mid):
    """The library's published named blocks, or empty dicts when it publishes
    none — named blocks are an optional part of the ABI, like the builder."""
    try:
        nb = getattr(lib, f"{prefix}_nblocks")
    except AttributeError:
        return {}, {}, {}
    nb.restype = _c_int
    nb.argtypes = [_c_int]
    n = int(nb(_c_int(mid)))
    if n <= 0:
        return {}, {}, {}
    bp = _f(lib, prefix, "block", [_c_int, _c_int, _pi])
    npf = getattr(lib, f"{prefix}_block_name")
    npf.restype = _c_int
    npf.argtypes = [_c_int, _c_int, ctypes.POINTER(ctypes.c_uint8), _c_int]
    vars_, cons_, pars_ = {}, {}, {}
    for k in range(n):
        out = np.zeros(12, dtype=np.int32)
        _check(bp(_c_int(mid), _c_int(k), _iarr(out)), f"{prefix}_block")
        need = int(npf(_c_int(mid), _c_int(k),
                       ctypes.cast(0, ctypes.POINTER(ctypes.c_uint8)), _c_int(0)))
        buf = (ctypes.c_uint8 * need)()
        npf(_c_int(mid), _c_int(k), buf, _c_int(need))
        name = bytes(buf).decode()
        nd = int(out[3])
        b = BlockRef(name,
                     ("var", "con", "par")[int(out[0])],
                     int(out[1]), int(out[2]),
                     tuple(int(out[4 + i]) for i in range(nd)),
                     k)
        {"var": vars_, "con": cons_, "par": pars_}[b.kind][name] = b
    return vars_, cons_, pars_


class CModel:
    """A model instance from a loaded library.

        lib = cnlpmodels.load("liblv.so")
        m = cnlpmodels.CModel(lib, 1000, prefix="lv")          # lv_new(1000)
        m = cnlpmodels.CModel("@acopf", bus, vmin, 100.0)      # search path
        m = cnlpmodels.CModel("rosen", 1000)                   # ./rosen (file or bundle dir)
        m = cnlpmodels.CModel("/opt/models/rosen", 1000)       # full path

    One library may carry **several models**, each under its own symbol
    prefix with its own schema and instances. A leading string argument
    selects one by name — the name is the prefix, mirroring CNLPModels.jl's
    `CNLPModel(lib, :acopf, ...)`; unambiguous, since a model argument is
    never a string. A mistyped name is refused at selection, not as a raw
    `undefined symbol` several calls later:

        m = cnlpmodels.CModel("@grid", "acopf", bus, 100.0)  # acopf_* in libgrid.so
        d = cnlpmodels.CModel("@grid", "dcopf", bus)         # dcopf_*, same file

    The arguments are the values the model is instantiated with — one per field
    of the library's schema, positionally, in the order the library publishes
    them, which is the same spelling the producer side uses
    (`ExaModel(core, arg1, ...)`, `compile_library(out, core, arg1, ...)`).

    Each value is a **number**, a **numpy/sequence array of numbers**, or a
    **table** (a dict of equal-length columns, sent to the builder column
    by column and validated against the library's schema). A lone integer uses
    `<prefix>_new(n)` when the library exports it and the builder otherwise;
    with no arguments at all the model is built from no instance data, which is
    valid when the schema declares no fields.

    Any number of instances may coexist per library.
    """

    def __init__(self, lib, *args, prefix=None):
        # A leading string argument names a MODEL in a library carrying
        # several — the name is the symbol prefix its ABI functions are
        # exported under, mirroring CNLPModels.jl's
        # `CNLPModel(lib, :acopf, ...)`. Unambiguous: a model argument is
        # never a string.
        model = None
        if args and isinstance(args[0], str):
            model, args = args[0], args[1:]
            if prefix is not None and prefix != model:
                raise TypeError(
                    f"both a model name ({model!r}) and prefix= ({prefix!r}) "
                    "were given; they mean the same thing — give one"
                )
            prefix = model
        if isinstance(lib, str):
            prefix = prefix if prefix is not None else _default_prefix(lib)
            lib = _resolve_spec(lib)
        prefix = prefix if prefix is not None else "rec"
        if model is not None:
            _require_model(lib, model)
        # Instantiate before resolving the evaluation table, so a failure to
        # build the model surfaces as what it is — not as a missing evaluation
        # symbol on a library that never got that far. Same order as the Julia
        # consumer.
        self._id = _instantiate(lib, prefix, args)
        self._prefix = prefix
        self._lib = lib
        self._vars, self._cons, self._pars = _read_layout(lib, prefix, self._id)
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

    # ── Named blocks ─────────────────────────────────────────────────────────

    def get_vars(self, name=None):
        """The model's named variable blocks as a dict, or one of them by name.

        A compiled library publishes the names its model was written with, so a
        caller who never sees the Julia source can address a slice of the
        solution by name:

            m = cnlpmodels.CModel("@grid", "acopf", bus)
            m.get_vars()                       # {"pg": BlockRef, ...}
            cnlpmodels.solution(x, m.get_vars("pg"))

        Same spellings as CNLPModels.jl's `get_vars` / `get_cons` / `get_pars`.
        """
        return self._named("var", name)

    def get_cons(self, name=None):
        """The model's named constraint blocks; see `get_vars`."""
        return self._named("con", name)

    def get_pars(self, name=None):
        """The model's named parameter blocks; see `get_vars`."""
        return self._named("par", name)

    def _named(self, kind, name):
        d = {"var": self._vars, "con": self._cons, "par": self._pars}[kind]
        if name is None:
            return dict(d)
        if name in d:
            return d[name]
        # The two ways of being wrong want different fixes: a name of another
        # kind names the accessor that would work, a typo lists what exists.
        for k, other in (("var", self._vars), ("con", self._cons), ("par", self._pars)):
            if name in other:
                what = {"var": "variable (get_vars)", "con": "constraint (get_cons)",
                        "par": "parameter (get_pars)"}[k]
                raise ValueError(f"{name!r} is a {what}, not a {kind}")
        what = {"var": "variable", "con": "constraint", "par": "parameter"}[kind]
        raise ValueError(
            f"this model has no named {what} {name!r}; it has {sorted(d)}"
            + ("" if d else " (none — this library publishes no named blocks)"))

    def get_value(self, block):
        """Read a parameter block's values.

        Unlike the Julia consumer's view, this returns a copy: the values live
        in the library's address space, not the caller's.
        """
        b = self._par_block(block)
        out = np.zeros(b.length)
        fn = _f(self._lib, self._prefix, "get_value", [_c_int, _c_int, _pd, _c_int])
        _check(fn(self._id, _c_int(b.index), _arr(out), _c_int(b.length)),
               f"{self._prefix}_get_value")
        return out.reshape(b.dims) if len(b.dims) > 1 else out

    def set_value(self, block, values):
        """Update a parameter block's values.

        Parameters are model state: the write takes effect for every later
        evaluation of this instance.
        """
        b = self._par_block(block)
        v = np.ascontiguousarray(np.ravel(values), dtype=np.float64)
        if v.size != b.length:
            raise ValueError(
                f"parameter {b.name!r} has {b.length} elements, got {v.size}")
        fn = _f(self._lib, self._prefix, "set_value", [_c_int, _c_int, _pd, _c_int])
        _check(fn(self._id, _c_int(b.index), _arr(v), _c_int(b.length)),
               f"{self._prefix}_set_value")
        return self

    def _par_block(self, block):
        b = self.get_pars(block) if isinstance(block, str) else block
        if b.kind != "par":
            raise ValueError(f"{b.name!r} is a {b.kind}, not a parameter")
        return b
