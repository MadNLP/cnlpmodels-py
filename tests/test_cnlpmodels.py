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
def lib(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    assert cc, "no C compiler available"
    so = tmp_path_factory.mktemp("lib") / "libtinyqp.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(so), str(FIX)], check=True)
    return cnlpmodels.load(so)


@pytest.fixture()
def m(lib):
    return cnlpmodels.CModel(lib, args=4, prefix="tq")


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
        cnlpmodels.CModel(lib, args=1, prefix="tq")


def test_multiple_instances_are_independent(lib):
    a = cnlpmodels.CModel(lib, args=4, prefix="tq")
    x = np.array([0.5, 0.25, 2.0, -1.0])
    before = a.obj(x)
    b = cnlpmodels.CModel(lib, args=6, prefix="tq")
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
    m = cnlpmodels.CModel("toy", args=4, prefix="tq")          # name-based
    assert m.nvar == 4
    with pytest.raises(FileNotFoundError):
        cnlpmodels.lib("nonexistent")


def test_args_shapes_are_uniform(lib):
    """int, positional tuple, and dict all instantiate; None needs an
    argument-free schema; wrong arity names the schema."""
    m_int = cnlpmodels.CModel(lib, args=5, prefix="tq")
    m_tup = cnlpmodels.CModel(lib, args=(5,), prefix="tq")
    m_dct = cnlpmodels.CModel(lib, args={"n": 5}, prefix="tq")
    assert m_int.nvar == m_tup.nvar == m_dct.nvar == 5
    x = np.zeros(5)
    assert m_int.obj(x) == m_tup.obj(x) == m_dct.obj(x)
    with pytest.raises(ValueError):
        cnlpmodels.CModel(lib, args=(5, 6), prefix="tq")
    with pytest.raises(ValueError, match="missing"):
        cnlpmodels.CModel(lib, prefix="tq")   # schema requires n
