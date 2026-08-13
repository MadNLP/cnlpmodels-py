# cnlpmodels

[![CI](https://github.com/madsuite-org/cnlpmodels-py/actions/workflows/ci.yml/badge.svg)](https://github.com/madsuite-org/cnlpmodels-py/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/madsuite-org/cnlpmodels-py/branch/master/graph/badge.svg)](https://codecov.io/gh/madsuite-org/cnlpmodels-py)

Load a nonlinear program (NLP) exposed by a **shared library through a plain
C interface** and use it as a numpy-backed model in Python — evaluate it, or
solve it with any solver that takes dense/sparse NLP callbacks (Ipopt via
`cyipopt` works out of the box; `scipy.optimize` too).

No language runtime is required beyond Python itself: the wrapper is
ctypes + numpy, and any library implementing the ABI below works, whatever
language produced it. `tests/fixtures/tinyqp.c` in this repository is a
complete reference implementation in ~100 lines of plain C, and the test
suite compiles and exercises it end to end.

```python
import cnlpmodels

cnlpmodels.set_path("/opt/models")          # or the CNLPMODELS_PATH env variable

m = cnlpmodels.CModel("@mymodel", 1000)     # resolves libmymodel.so
# m = cnlpmodels.CModel("/path/to/mymodel", 1000)   # or a literal path
x, info = cnlpmodels.solve_ipopt(m)         # cyipopt, if installed
```

`m` exposes `nvar`, `ncon`, `nnzj`, `nnzh`, `x0`, `lvar`, `uvar`, `lcon`,
`ucon`, and methods `obj(x)`, `grad(x)`, `cons(x)`, `jac_structure()`,
`jac(x)`, `hess_structure()`, `hess(x, y, obj_weight=1.0)` — numpy in,
numpy out, structure indices 0-based per Python convention. Any number of
model instances may coexist per library.

## The C ABI

For a symbol prefix `P` (the name-based loader defaults it to the library
name), the library exports the functions below. Conventions:

- every function returns `int32` **status**: `0` = success (except `P_new`,
  `P_data_begin`, `P_new_from_data`, `P_schema`, which return positive
  values as described);
- indices are **1-based inside the ABI** (this wrapper shifts to 0-based);
- the Hessian is the **lower triangle** of `obj_weight * ∇²f(x) + Σᵢ yᵢ ∇²cᵢ(x)`;
- buffers are caller-allocated dense `double` / `int32` / `int64` arrays.

### Instantiation

```c
int32_t P_new(int32_t n);          // → model id (>0), 0 on failure
```

### Metadata and evaluation (per instance `id`)

```c
int32_t P_nvar(int32_t id);  P_ncon(id);  P_nnzj(id);  P_nnzh(id);
int32_t P_meta(int32_t id, double* x0, double* lvar, double* uvar,
               double* lcon, double* ucon);                        // ±INFINITY ok
int32_t P_obj (int32_t id, const double* x, double* out);
int32_t P_grad(int32_t id, const double* x, double* g);            // length nvar
int32_t P_cons(int32_t id, const double* x, double* c);            // length ncon
int32_t P_jac_structure(int32_t id, int32_t* rows, int32_t* cols); // nnzj, 1-based
int32_t P_jac (int32_t id, const double* x, double* vals);         // nnzj
int32_t P_hess_structure(int32_t id, int32_t* rows, int32_t* cols);// nnzh, 1-based
int32_t P_hess(int32_t id, const double* x, const double* y,
               double obj_weight, double* vals);                   // nnzh
```

### Structured instantiation (optional, "ABI v2")

Libraries built from structured data publish a JSON schema and take the data
through a builder — tables cross the boundary **as columns**:

```c
int32_t P_schema(uint8_t* buf, int32_t len);        // returns needed length
int32_t P_data_begin(void);                         // → builder id
int32_t P_set_scalar_f64(int32_t b, const char* field, double v);       // _i64 too
int32_t P_set_array_f64 (int32_t b, const char* field, const double* v, int32_t len);
int32_t P_set_col_f64   (int32_t b, const char* table, const char* col,
                         const double* v, int32_t len);                 // _i64 too
int32_t P_data_ready    (int32_t b);                // 1 iff complete and consistent
int32_t P_new_from_data (int32_t b);                // → model id
```

From Python the arguments are positional, one per schema field, in the order
the library publishes them — a table is a dict of equal-length columns:

```python
m = cnlpmodels.CModel("@mymodel",
    {"i": np.array([1, 2]), "pd": np.array([0.4, 0.3])},   # a table (columns)
    np.array([0.9, 0.9]),                                  # an array
    100.0,                                                 # a scalar
)
m = cnlpmodels.CModel("@scalable", 1000)  # one integer: <prefix>_new when
                                          # exported, else the builder
```

which is the same spelling the producer side uses — `ExaModel(core, arg1,
arg2, ...)` to instantiate a recipe, `compile_library(out, core, arg1, ...)`
to compile one — so a model is consumed the way it was written. A library
compiled from a recipe names its fields `arg1`, `arg2`, ... for that reason.
Each value is checked against the kind and type the schema declares for its
slot; nothing is coerced across it.

### Several models in one library

One shared library may export any number of models, each under its own symbol
prefix with its own schema and its own instances. A leading string argument
selects one by name — the name **is** the prefix its ABI functions are
exported under (unambiguous, since a model argument is never a string):

```python
m = cnlpmodels.CModel("@grid", "acopf", bus, 100.0)  # acopf_* inside libgrid.so
d = cnlpmodels.CModel("@grid", "dcopf", bus)         # dcopf_* in the same file
sch = cnlpmodels.schema(lib, "acopf")                # schemas are per model
```

A mistyped name is refused at selection, with the witness symbol named, rather
than surfacing as a raw `undefined symbol` several calls later. Omitting the
name keeps the single-model spelling, where the prefix falls back to the
library name — a one-model library is unaffected. This is the same selection
spelling as CNLPModels.jl's `CNLPModel(lib, :acopf, ...)`.

## Implementing a compatible library

1. Export the functions above with C linkage under one prefix.
2. Keep instances behind integer ids (a static table suffices — see the
   reference implementation).
3. Return `0`/positive ids on success, nonzero/`0` on failure; never let an
   exception cross the boundary.
4. Validate against `tests/fixtures/tinyqp.c` and this test suite, which
   compiles that file and checks every function against closed-form values,
   including a solve to the model's known optimum. The file carries two
   models: `tq_`, instantiated from one integer, and `sq_`, which has no
   one-integer constructor and is built from a three-field schema through the
   builder.

Sibling package:
[`CNLPModels.jl`](https://github.com/madsuite-org/CNLPModels.jl) — the same
consumer for Julia.
