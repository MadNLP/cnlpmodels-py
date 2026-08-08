# cnlpmodels

Consume an NLP exposed by a C-ABI shared library — typically compiled from an
ExaModels tape with `ExaModels.compile_library` (Julia) — as a numpy-backed
Python model. **No Julia required**: the library is self-contained, and this
package is ctypes + numpy.

```python
import cnlpmodels

lib = cnlpmodels.load("liblv.so")
m = cnlpmodels.CModel(lib, n=1000, prefix="lv")   # lv_new(1000) → instance
m.obj(m.x0), m.grad(m.x0), m.cons(m.x0)           # numpy in, numpy out
```

Pair with any Python solver: `scipy.optimize` works out of the box (see
`tests/test_cnlpmodels.py::test_solve_with_scipy`), `cyipopt` via the
`[ipopt]` extra. Any number of model instances may coexist per library.

The C ABI and the duck-typed model protocol (shared with `examodels`) are
documented in the module docstring; `tests/fixtures/tinyqp.c` is a complete
plain-C reference implementation, and the test suite is fully hermetic —
it compiles that fixture and needs no Julia toolchain.

Sibling packages: `examodels` (modeling + tape recording + compilation;
needs the Julia runtime) · `CNLPModels.jl` (the same consumer for Julia
hosts).
