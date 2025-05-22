import numpy as np
from highspy import Highs
from models import LPModel
from utils import build_sparse_matrix
from utils import analyze_infeasible, analyze_infeasible_detailed

def solve_model(model: LPModel) -> dict:
    var_order = list(model.variables.keys())
    n_vars = len(var_order)

    cost_vector = np.array([model.variables[v]["cost"] for v in var_order], dtype=np.float64)
    lb = np.array([model.variables[v].get("min", 0.0) for v in var_order], dtype=np.float64)
    ub = np.array([model.variables[v].get("max", 1.0) for v in var_order], dtype=np.float64)

    A, rhs_lo, rhs_hi, constraint_names = build_sparse_matrix(model.constraints, model.variables, var_order)
    starts = A.indptr.astype(np.int32)
    index = A.indices.astype(np.int32)
    values = A.data.astype(np.float64)

    solver = Highs()
    solver.addRows(len(rhs_lo), np.array(rhs_lo), np.array(rhs_hi),
                   0, np.array([], dtype=np.int32),
                   np.array([], dtype=np.int32),
                   np.array([], dtype=np.float64))
    solver.addCols(n_vars, cost_vector, lb, ub, len(values), starts, index, values)

    solver.run()
    status = solver.getModelStatus()

    # Always collect contributions (for debug)
    contributions = {}
    for c_name in model.constraints.keys():
        contributions[c_name] = {
            var: model.variables[var].get(c_name, 0.0)
            for var in var_order
        }

    # Debug variable bounds
    variable_bounds = {
        var: {
            "min": model.variables[var].get("min", 0.0),
            "max": model.variables[var].get("max", 1.0)
        } for var in var_order
    }

    if str(status) != "HighsModelStatus.kOptimal":
        debug = {
            "reason": "HiGHS returned infeasible model",
            "model_status": str(status),
            "constraints": {k: v.dict() for k, v in model.constraints.items()},
            "variable_bounds": variable_bounds,
            "contributions": contributions
        }

        debug["hint_summary"] = analyze_infeasible(debug)
        debug["hint_ranked"] = analyze_infeasible_detailed(debug)

        return {
            "vars": {},
            "cost": 0.0,
            "infeasible": True,
            "debug": debug
        }


    sol = solver.getSolution()
    x = sol.col_value
    var_result = {var_order[i]: x[i] for i in range(n_vars)}
    total_cost = float(np.dot(cost_vector, x))

    # Residuals per constraint
    constraint_residuals = {}
    for i, cname in enumerate(constraint_names):
        actual = sum(model.variables[var].get(cname, 0.0) * var_result[var] for var in var_order)
        constraint_residuals[cname] = {
            "value": round(actual, 6),
            "min": round(rhs_lo[i], 6),
            "max": round(rhs_hi[i], 6),
            "status": "ok" if rhs_lo[i] <= actual <= rhs_hi[i] else "violated"
        }

    # Virtual usage
    virtuals_used = {
        var: round(val, 6) for var, val in var_result.items()
        if var.startswith("virtual_") and val > 1e-6
    }

    return {
        "vars": var_result,
        "cost": round(total_cost, 6),
        "infeasible": False,
        "debug": {
            "constraint_residuals": constraint_residuals,
            "virtuals_used": virtuals_used
        }
    }
