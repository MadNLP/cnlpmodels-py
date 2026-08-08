"""Solve a protocol model with Ipopt through cyipopt (optional extra)."""
import numpy as np

__all__ = ["solve_ipopt"]


def solve_ipopt(model, x0=None, **options):
    """Solve `model` (anything implementing the cnlpmodels protocol) with
    Ipopt. Returns `(x, info)` as given by cyipopt. Options are passed to
    Ipopt verbatim (e.g. ``print_level=0``, ``tol=1e-8``)."""
    import cyipopt  # deferred: [ipopt] extra

    class _Problem:
        def objective(self, x):
            return model.obj(x)

        def gradient(self, x):
            return model.grad(x)

        def constraints(self, x):
            return model.cons(x)

        def jacobian(self, x):
            return model.jac(x)

        def jacobianstructure(self):
            return model.jac_structure()

        def hessianstructure(self):
            return model.hess_structure()

        def hessian(self, x, lagrange, obj_factor):
            return model.hess(x, lagrange, obj_weight=obj_factor)

    nlp = cyipopt.Problem(
        n=model.nvar, m=model.ncon, problem_obj=_Problem(),
        lb=model.lvar, ub=model.uvar, cl=model.lcon, cu=model.ucon,
    )
    for k, v in options.items():
        nlp.add_option(k, v)
    start = model.x0 if x0 is None else np.ascontiguousarray(x0, dtype=np.float64)
    return nlp.solve(start)
