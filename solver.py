import numpy as np
from highspy import Highs
from models import LPModel
from utils import build_sparse_matrix


def solve_model(model: LPModel) -> dict:
    var_order = list(model.variables.keys())
    n_vars = len(var_order)

    # Extract vectors
    cost_vector = np.array([model.variables[v]["cost"] for v in var_order], dtype=np.float64)
    lb = np.array([model.variables[v].get("min", 0.0) for v in var_order], dtype=np.float64)
    ub = np.array([model.variables[v].get("max", 1.0) for v in var_order], dtype=np.float64)

    # Matrix assembly
    A, rhs_lo, rhs_hi, constraint_names = build_sparse_matrix(model.constraints, model.variables, var_order)
    starts = A.indptr.astype(np.int32)
    index = A.indices.astype(np.int32)
    values = A.data.astype(np.float64)

    # HiGHS solve
    solver = Highs()
    solver.addRows(len(rhs_lo), np.array(rhs_lo), np.array(rhs_hi),
                   0, np.array([], dtype=np.int32),
                   np.array([], dtype=np.int32),
                   np.array([], dtype=np.float64))
    solver.addCols(n_vars, cost_vector, lb, ub, len(values), starts, index, values)
    solver.run()

    status = solver.getModelStatus()
    if str(status) != "HighsModelStatus.kOptimal":
        return {
            "vars": {},
            "cost": 0.0,
            "infeasible": True,
            "debug": {
                "reason": "HiGHS returned infeasible model",
                "model_status": str(status)
            }
        }

    # Parse solution
    sol = solver.getSolution()
    x = sol.col_value
    var_result = {var_order[i]: x[i] for i in range(n_vars)}
    total_cost = float(np.dot(cost_vector, x))

    # Debug: constraint residuals
    constraint_residuals = {}
    for i, name in enumerate(constraint_names):
        value = sum(model.variables[var].get(name, 0.0) * var_result.get(var, 0.0) for var in var_order)
        constraint_residuals[name] = {
            "value": round(value, 6),
            "min": round(rhs_lo[i], 6),
            "max": round(rhs_hi[i], 6)
        }

    # Debug: virtual ingredient usage
    virtuals_used = {
        var: round(val, 6) for var, val in var_result.items()
        if var.startswith("virtual_") and val > 1e-6
    }

    # Try to reconstruct where it's failing
    contribs = {}
    for cname in model.constraints.keys():
        contribs[cname] = {
            var: model.variables[var].get(cname, 0.0)
            for var in var_order
        }

    return {
        "vars": {},
        "cost": 0.0,
        "infeasible": True,
        "debug": {
            "reason": "HiGHS returned infeasible model",
            "model_status": str(status),
            "constraints": model.constraints,
            "variable_bounds": {var: {"min": model.variables[var].get("min", 0), "max": model.variables[var].get("max", 1)} for var in var_order},
            "contributions": contribs
        }
    }
