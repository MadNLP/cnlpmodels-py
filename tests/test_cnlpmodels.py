"""Hermetic tests against the plain-C reference library (no Julia anywhere)."""
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import cnlpmodels

FIX = pathlib.Path(__file__).parent / "fixtures" / "tinyqp.c"


@pytest.fixture(scope="module")
def sopath(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    assert cc, "no C compiler available"
    so = tmp_path_factory.mktemp("lib") / "libtinyqp.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(FIX)], check=True)
    return so


@pytest.fixture(scope="module")
def lib(sopath):
    return cnlpmodels.load(sopath)


@pytest.fixture()
def m(lib):
    return cnlpmodels.CModel(lib, 4, prefix="tq")


def test_meta(m):
    assert (m.nvar, m.ncon, m.nnzj, m.nnzh) == (4, 1, 2, 4)
    assert np.all(m.x0 == 0.0)
    assert np.all(np.isinf(m.lvar)) and np.all(m.lvar < 0)
    assert np.all(np.isinf(m.uvar)) and np.all(m.uvar > 0)
    assert m.lcon.tolist() == [0.0] and m.ucon.tolist() == [0.0]


def test_evaluations_match_closed_form(m):
    x = np.array([0.5, 0.25, 2.0, -1.0])
    assert m.obj(x) == 0.25 + 0.5625 + 1.0 + 4.0
    assert np.array_equal(m.grad(x), 2.0 * (x - 1.0))
    assert m.cons(x).tolist() == [x[0] + x[1] - 1.0]
    rows, cols = m.jac_structure()
    assert rows.tolist() == [0, 0] and cols.tolist() == [0, 1]   # 0-based
    assert m.jac(x).tolist() == [1.0, 1.0]
    hr, hc = m.hess_structure()
    assert hr.tolist() == [0, 1, 2, 3] and hc.tolist() == [0, 1, 2, 3]
    assert m.hess(x, np.array([3.0]), obj_weight=0.7).tolist() == [1.4] * 4


def test_new_failure_raises(lib):
    with pytest.raises(RuntimeError, match="tq_new"):
        cnlpmodels.CModel(lib, 1, prefix="tq")


def test_multiple_instances_are_independent(lib):
    a = cnlpmodels.CModel(lib, 4, prefix="tq")
    x = np.array([0.5, 0.25, 2.0, -1.0])
    before = a.obj(x)
    b = cnlpmodels.CModel(lib, 6, prefix="tq")
    assert b.nvar == 6
    assert a.obj(x) == before


def test_solve_with_scipy(m):
    """Julia-free end-to-end: solve through scipy.optimize (SLSQP).

    min sum((x-1)^2) s.t. x1+x2=1 has optimum 0.5 at (.5, .5, 1, 1)."""
    scipy = pytest.importorskip("scipy.optimize")
    res = scipy.minimize(
        m.obj, m.x0, jac=m.grad, method="SLSQP",
        constraints=[{"type": "eq", "fun": m.cons,
                      "jac": lambda x: np.array([[1.0, 1.0, 0.0, 0.0]])}],
    )
    assert res.success
    assert abs(res.fun - 0.5) < 1e-8
    assert np.allclose(res.x, [0.5, 0.5, 1.0, 1.0], atol=1e-6)


def test_solve_with_cyipopt(m):
    cyipopt = pytest.importorskip("cyipopt")  # noqa: F841
    from cnlpmodels import solve_ipopt
    x, info = solve_ipopt(m, print_level=0)
    assert info["status"] == 0
    assert abs(info["obj_val"] - 0.5) < 1e-8
    assert np.allclose(x, [0.5, 0.5, 1.0, 1.0], atol=1e-6)


def test_name_based_loading(lib, tmp_path):
    import shutil
    shutil.copy(pathlib.Path(lib._name), tmp_path / "libtoy.so")
    cnlpmodels.set_path(tmp_path)
    l1 = cnlpmodels.lib("toy")
    assert cnlpmodels.lib("toy") is l1                      # cached
    m = cnlpmodels.CModel("@toy", 4, prefix="tq")         # name-based
    assert m.nvar == 4
    with pytest.raises(FileNotFoundError):
        cnlpmodels.lib("nonexistent")


def test_lone_integer_takes_the_one_knob_constructor(lib):
    """`tq` declares one scalar field and also exports tq_new."""
    assert [f["name"] for f in cnlpmodels.schema(lib, prefix="tq")["fields"]] == ["n"]
    m = cnlpmodels.CModel(lib, 5, prefix="tq")
    assert m.nvar == 5
    assert m.obj(np.zeros(5)) == 5.0          # sum (0 - 1)^2


def test_arguments_bind_positionally_to_the_schema(lib):
    """`sq` is builder-only (no sq_new) and declares three fields, so the
    arguments bind positionally in the order the schema publishes them."""
    assert [f["name"] for f in cnlpmodels.schema(lib, prefix="sq")["fields"]] == \
        ["n", "s", "w"]
    n, s, w = 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0])
    m = cnlpmodels.CModel(lib, n, s, w, prefix="sq")
    assert (m.nvar, m.ncon) == (4, 1)
    x = np.array([0.5, 0.25, 2.0, -1.0])
    assert m.obj(x) == np.sum(w * (x - s) ** 2)        # min sum w_i (x_i - s)^2
    assert np.array_equal(m.grad(x), 2.0 * w * (x - s))
    assert np.array_equal(m.hess(x, np.array([3.0]), obj_weight=0.7), 1.4 * w)
    assert m.cons(x).tolist() == [x[0] + x[1] - 1.0]


def test_argument_order_is_load_bearing(lib):
    """The same values in the wrong order are refused by the slot they land in
    — n is the int64 scalar, so the float meant for s cannot go there. Coercing
    would build a different model from the one asked for, silently."""
    with pytest.raises(TypeError, match=r"'n' is an int64"):
        cnlpmodels.CModel(lib, 2.0, 4, np.array([1.0, 2.0, 3.0, 4.0]), prefix="sq")
    # The same guard on arrays and table columns, which the fixture's schema
    # has no int64 instance of.
    assert cnlpmodels._as_col([1.0, 2.0], "f64").dtype == np.float64
    with pytest.raises(TypeError, match=r"int64 array"):
        cnlpmodels._as_col(np.array([1.5, 2.5]), "i64", "bus.i")


def test_wrong_arity_names_the_schema(lib):
    with pytest.raises(ValueError, match=r"declares 3 fields"):
        cnlpmodels.CModel(lib, 4, 2.0, prefix="sq")
    with pytest.raises(ValueError, match=r"declares 1 field"):
        cnlpmodels.CModel(lib, 5, 6, prefix="tq")
    # No arguments at all is the no-instance-data case; both schemas want some.
    with pytest.raises(ValueError, match=r"given 0 arguments"):
        cnlpmodels.CModel(lib, prefix="tq")
    with pytest.raises(ValueError, match=r"given 0 arguments"):
        cnlpmodels.CModel(lib, prefix="sq")


def test_inconsistent_structured_data_is_the_librarys_call(lib):
    """w must be as long as n says — the library decides, not this wrapper."""
    with pytest.raises(RuntimeError, match="incomplete"):
        cnlpmodels.CModel(lib, 4, 2.0, np.array([1.0, 2.0]), prefix="sq")


def test_a_bool_is_not_an_argument(lib):
    with pytest.raises(TypeError):
        cnlpmodels.CModel(lib, True, prefix="tq")


def test_fixed_library_needs_no_arguments(lib):
    # `fx` declares zero instantiation arguments (fx_nargs() == 0), so no
    # arguments instantiate it directly through fx_new — whose integer is part
    # of the C signature and ignored.
    m0 = cnlpmodels.CModel(lib, prefix="fx")
    assert (m0.nvar, m0.ncon) == (3, 1)
    assert m0.obj(np.array([0.5, 0.5, 1.0])) == 0.5
    # An explicit integer still works, and lands on the same fixed model.
    m1 = cnlpmodels.CModel(lib, 999, prefix="fx")
    assert m1.nvar == 3


def test_string_is_at_name_on_search_path_or_literal_path(sopath, tmp_path, monkeypatch):
    spec = str(sopath)
    # `@name` resolves on the search path and defaults the prefix to the name.
    assert cnlpmodels._is_name("@toy")
    assert cnlpmodels._default_prefix("@toy") == "toy"

    # Anything else is a filesystem path, exactly as written. A full path to
    # the library file, with the prefix defaulting from the file name
    # (libtinyqp.so → tinyqp), overridable as always:
    assert not cnlpmodels._is_name(spec)
    assert cnlpmodels._default_prefix(spec) == "tinyqp"
    mp = cnlpmodels.CModel(spec, 4, prefix="tq")
    assert mp.nvar == 4
    # The handle is cached by absolute path: one dlopen per library.
    assert cnlpmodels._resolve_spec(spec) is cnlpmodels._resolve_spec(spec)
    # A fixed model by path needs nothing beyond the path.
    mf = cnlpmodels.CModel(spec, prefix="fx")
    assert mf.nvar == 3

    # A path to a bundle DIRECTORY finds the library inside it, and the
    # prefix defaults from the directory name.
    bdir = tmp_path / "toyqp2"
    (bdir / "lib").mkdir(parents=True)
    shutil.copy(sopath, bdir / "lib" / "libtoyqp2.so")
    assert cnlpmodels._default_prefix(str(bdir)) == "toyqp2"
    md = cnlpmodels.CModel(str(bdir), 4, prefix="tq")
    assert md.nvar == 4

    # A bare string without `@` is a file in the current directory — NOT a
    # search-path name.
    shutil.copy(sopath, tmp_path / "qp.so")
    monkeypatch.chdir(tmp_path)
    mc = cnlpmodels.CModel("qp.so", 4, prefix="tq")
    assert mc.nvar == 4
    with pytest.raises(FileNotFoundError, match="no shared library at"):
        cnlpmodels.CModel("toy", 4)

    # A path that is not there fails as a path, never as a search-path miss.
    with pytest.raises(FileNotFoundError, match="no shared library at"):
        cnlpmodels.CModel(str(sopath.parent / "libnope.so"), 1)


def test_search_path_initializes_from_the_environment(monkeypatch, tmp_path):
    cnlpmodels.set_path()                      # empty the registry
    monkeypatch.setenv("CNLPMODELS_PATH", f"{tmp_path}:/nonexistent")
    assert cnlpmodels._paths() == [str(tmp_path), "/nonexistent"]
    cnlpmodels.set_path()                      # leave no residue for later tests


def test_a_directory_without_a_library_says_what_it_tried(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shared library in"):
        cnlpmodels.CModel(str(tmp_path), 1)


def test_value_guards_name_the_field_and_kind():
    with pytest.raises(TypeError, match="bool is not a model argument"):
        cnlpmodels._as_scalar("n", True, "i64")
    with pytest.raises(TypeError, match="is a float64"):
        cnlpmodels._as_scalar("s", "x", "f64")
    with pytest.raises(TypeError, match="float64 array"):
        cnlpmodels._as_col(np.array(["a"]), "f64", "w")
    with pytest.raises(ValueError, match="one-dimensional"):
        cnlpmodels._as_col(np.ones((2, 2)), "f64", "w")


def test_fill_data_validates_keys_against_the_schema(lib):
    with pytest.raises(ValueError, match="do not match schema"):
        cnlpmodels._fill_data(lib, "sq", {"n": 4})


def test_a_table_field_binds_column_by_column(lib):
    # min Σ_j w_j (x_{i_j} - 1)^2 over 3 vars: rows (1, 1.0), (2, 2.0),
    # (3, 3.0), (1, 0.5) — variable 1 carries weight 1.5.
    pts = {"i": np.array([1, 2, 3, 1]), "w": np.array([1.0, 2.0, 3.0, 0.5])}
    m = cnlpmodels.CModel(lib, pts, prefix="tb")
    assert (m.nvar, m.ncon, m.nnzj, m.nnzh) == (3, 0, 0, 3)
    x = np.array([0.0, 2.0, 4.0])
    assert m.obj(x) == 1.5 * 1.0 + 2.0 * 1.0 + 3.0 * 9.0
    assert m.grad(x).tolist() == [2 * 1.5 * -1.0, 2 * 2.0 * 1.0, 2 * 3.0 * 3.0]
    hr, hc = m.hess_structure()
    assert hr.tolist() == [0, 1, 2] and hc.tolist() == [0, 1, 2]
    assert m.hess(x, np.zeros(0), obj_weight=0.5).tolist() == [1.5, 2.0, 3.0]

    # Validation happens before any column crosses the boundary.
    with pytest.raises(ValueError, match="columns"):
        cnlpmodels.CModel(lib, {"i": [1], "x": [1.0]}, prefix="tb")
    with pytest.raises(ValueError, match="column lengths differ"):
        cnlpmodels.CModel(lib, {"i": [1, 2], "w": [1.0]}, prefix="tb")
    # A column the library refuses surfaces as its setter's status.
    with pytest.raises(RuntimeError, match="set_col_i64"):
        cnlpmodels.CModel(lib, {"i": [1] * 9, "w": [1.0] * 9}, prefix="tb")


def test_each_builder_failure_names_its_stage(lib):
    with pytest.raises(RuntimeError, match="data_begin failed"):
        cnlpmodels.CModel(lib, 4, prefix="bf")
    with pytest.raises(RuntimeError, match="new_from_data failed"):
        cnlpmodels.CModel(lib, 4, prefix="nf")
    with pytest.raises(RuntimeError, match="no builder surface"):
        cnlpmodels.CModel(lib, 2.0, prefix="fx")
    with pytest.raises(RuntimeError, match="ns_schema does not"):
        cnlpmodels.CModel(lib, 4, prefix="ns")
    # A lone integer for a builder-only library falls through to the schema.
    with pytest.raises(ValueError, match="declares 3 fields"):
        cnlpmodels.CModel(lib, 4, prefix="sq")


def test_a_failing_fixed_constructor_is_reported(lib):
    with pytest.raises(RuntimeError, match=r"zz_new\(0\) failed"):
        cnlpmodels.CModel(lib, prefix="zz")


def test_evaluation_shape_guards(lib):
    m = cnlpmodels.CModel(lib, 4, prefix="tq")
    with pytest.raises(ValueError, match=r"x must have shape \(4,\)"):
        m.obj(np.zeros(3))
    with pytest.raises(ValueError, match=r"y must have shape \(1,\)"):
        m.hess(np.zeros(4), np.zeros(2))


# ── Selecting a model by name in a library carrying several ──────────────────
# The fixture carries several models in ONE shared library; a leading string
# argument names one, and the name is the symbol prefix — the same selection
# spelling as CNLPModels.jl's `CNLPModel(lib, :tq, ...)`.


def test_model_selection_by_name(lib):
    x = np.array([0.5, 0.25, 2.0, -1.0])
    n, s, w = 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0])

    m = cnlpmodels.CModel(lib, "tq", 4)
    assert m.nvar == 4
    ms = cnlpmodels.CModel(lib, "sq", n, s, w)   # builder-only sibling
    assert m.obj(x) == ((x - 1.0) ** 2).sum()
    assert ms.obj(x) == (w * (x - s) ** 2).sum()

    # Instances of DIFFERENT models coexist as freely as instances of one.
    m6 = cnlpmodels.CModel(lib, "tq", 6)
    assert m6.nvar == 6
    assert m.obj(x) == ((x - 1.0) ** 2).sum()
    assert ms.obj(x) == (w * (x - s) ** 2).sum()


def test_unknown_model_name_is_refused_clearly(lib):
    # A mistyped name is reported at selection, with the witness spelled out —
    # not as a raw ctypes `undefined symbol` several calls later.
    with pytest.raises(ValueError, match=r"carries no model named 'nosuch'"):
        cnlpmodels.CModel(lib, "nosuch", 4)
    with pytest.raises(ValueError, match=r"carries no model named"):
        cnlpmodels.schema(lib, "nosuch")


def test_model_name_and_prefix_must_agree(lib):
    with pytest.raises(TypeError, match=r"give one"):
        cnlpmodels.CModel(lib, "tq", 4, prefix="sq")
    assert cnlpmodels.CModel(lib, "tq", 4, prefix="tq").nvar == 4


def test_schema_by_model_name(lib):
    sch = cnlpmodels.schema(lib, "sq")
    assert [f["name"] for f in sch["fields"]] == ["n", "s", "w"]


def test_at_sigil_is_accepted_by_lib(lib, tmp_path):
    import shutil
    shutil.copy(pathlib.Path(lib._name), tmp_path / "libtoy9.so")
    cnlpmodels.set_path(tmp_path)
    assert cnlpmodels.lib("@toy9") is cnlpmodels.lib("toy9")     # one cache entry
    m = cnlpmodels.CModel("@toy9", "tq", 4)                      # with model selection
    assert m.nvar == 4


# ── Named blocks ─────────────────────────────────────────────────────────────
# The fixture publishes a layout: `x` (variable), `budget` (constraint), `w`
# (parameter) — the same surface a compiled ExaModels library publishes.


def test_named_blocks(lib):
    m = cnlpmodels.CModel(lib, 4, prefix="tq")
    assert sorted(m.get_vars()) == ["x"]
    assert sorted(m.get_cons()) == ["budget"]
    assert sorted(m.get_pars()) == ["w"]

    b = m.get_vars("x")
    assert (b.kind, b.offset, b.length, b.dims) == ("var", 0, 4, (4,))
    # Lengths follow the INSTANCE, not the library.
    assert cnlpmodels.CModel(lib, 7, prefix="tq").get_vars("x").length == 7

    # Result extraction reshapes the block's own slice.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cnlpmodels.solution(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert cnlpmodels.multipliers(np.array([7.0]), m.get_cons("budget")).tolist() == [7.0]
    assert cnlpmodels.multipliers_L(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert cnlpmodels.multipliers_U(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_named_block_lookup_errors(lib):
    m = cnlpmodels.CModel(lib, 4, prefix="tq")
    with pytest.raises(ValueError, match=r"is a parameter \(get_pars\), not a var"):
        m.get_vars("w")
    with pytest.raises(ValueError, match=r"is a variable \(get_vars\), not a par"):
        m.get_pars("x")
    with pytest.raises(ValueError, match=r"no named variable 'nope'"):
        m.get_vars("nope")


def test_parameter_values(lib):
    m = cnlpmodels.CModel(lib, 4, prefix="tq")
    p = m.get_pars("w")
    m.set_value(p, [3.0, 4.0])
    assert m.get_value(p).tolist() == [3.0, 4.0]
    assert m.get_value("w").tolist() == [3.0, 4.0]       # by name too
    with pytest.raises(ValueError, match="has 2 elements, got 1"):
        m.set_value(p, [1.0])
    with pytest.raises(ValueError, match="not a parameter"):
        m.get_value(m.get_vars("x"))


def test_library_without_named_blocks(lib):
    # Named blocks are optional in the ABI: `sq` publishes none, and everything
    # else about it still works.
    ms = cnlpmodels.CModel(lib, 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0]), prefix="sq")
    assert ms.get_vars() == {}
    with pytest.raises(ValueError, match="publishes no named blocks"):
        ms.get_vars("x")
    assert ms.nvar == 4
