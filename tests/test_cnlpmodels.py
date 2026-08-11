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
