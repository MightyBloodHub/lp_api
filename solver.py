import numpy as np
import os
import subprocess
import tempfile
from highspy import Highs
from models import LPModel
from utils import build_sparse_matrix


def apply_relaxations(model: LPModel, suggestions: dict) -> LPModel:
    """Return a new model with constraint bounds relaxed."""

    import copy

    relaxed = copy.deepcopy(model)

    for cname, adjustments in suggestions.get("hint_relaxations", {}).items():
        constraint = relaxed.constraints.get(cname)
        if not constraint:
            continue
        if "min" in adjustments and constraint.min is not None:
            new_min = constraint.min - adjustments["min"]
            constraint.min = max(0.0, new_min)
        if "max" in adjustments and constraint.max is not None:
            constraint.max += adjustments["max"]
        relaxed.constraints[cname] = constraint
        print(f"Relaxed {cname}: min -> {constraint.min}, max -> {constraint.max}")

    return relaxed


def _suggest_relaxations(model: LPModel):
    """Return minimal constraint relaxations using slack variables."""
    var_order = list(model.variables.keys())
    n_vars = len(var_order)

    cost_vector = np.zeros(n_vars, dtype=np.float64)
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

    slack_info = {}
    slack_idx = n_vars
    for row_idx, cname in enumerate(constraint_names):
        con = model.constraints[cname]
        if con.min is not None:
            solver.addCol(1.0, 0.0, np.inf, 1,
                          np.array([row_idx], dtype=np.int32),
                          np.array([1.0], dtype=np.float64))
            slack_info[f"{cname}.min"] = slack_idx
            slack_idx += 1
        if con.max is not None:
            solver.addCol(1.0, 0.0, np.inf, 1,
                          np.array([row_idx], dtype=np.int32),
                          np.array([-1.0], dtype=np.float64))
            slack_info[f"{cname}.max"] = slack_idx
            slack_idx += 1

    solver.run()
    status = solver.getModelStatus()
    if str(status) != "HighsModelStatus.kOptimal":
        return {}

    sol = solver.getSolution().col_value
    relaxations = {}
    for key, idx in slack_info.items():
        cname, bound = key.split(".")
        relaxations.setdefault(cname, {"min": 0.0, "max": 0.0})
        if bound == "min":
            relaxations[cname]["min"] = float(sol[idx])
        else:
            relaxations[cname]["max"] = float(sol[idx])

    # Drop zeros for readability
    relaxations = {
        c: {k: round(v, 6) for k, v in b.items() if v > 1e-8}
        for c, b in relaxations.items() if any(v > 1e-8 for v in b.values())
    }
    return relaxations

def solve_model(model: LPModel, iteration_depth: int = 0) -> dict:
    if iteration_depth > 1:
        return {
            "vars": {},
            "cost": 0.0,
            "infeasible": True,
            "debug": {"reason": "Failed after relaxation attempt"},
        }
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

    # Save model and run HiGHS using CLI for IIS detection when infeasible
    with tempfile.TemporaryDirectory() as tmpdir:
        lp_path = os.path.join(tmpdir, "model.lp")
        solver.writeModel(lp_path)

        solver.run()
        status = solver.getModelStatus()

        iis_constraints = []
        iis_vars = []
        if str(status) == "HighsModelStatus.kInfeasible":
            iis_path = os.path.join(tmpdir, "model.iis")
            try:
                subprocess.run(
                    ["highs", "--read", lp_path, "--iis_find", "true", "--write_iis", iis_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

                with open(iis_path, "r") as f:
                    iis_data = f.read()

                for line in iis_data.splitlines():
                    if line.startswith("row"):
                        parts = line.split()
                        if len(parts) > 2 and parts[2].lower() == "in":
                            iis_constraints.append(parts[1])
                    elif line.startswith("col"):
                        parts = line.split()
                        if len(parts) > 2 and parts[2].lower() == "in":
                            iis_vars.append(parts[1])
            except subprocess.CalledProcessError:
                iis_constraints, iis_vars = [], []
        else:
            iis_constraints, iis_vars = [], []

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

    # Compute achievable ranges per constraint
    constraint_bounds = {}
    for cname in model.constraints.keys():
        min_val = sum(
            variable_bounds[v]["min"] * contributions[cname].get(v, 0)
            for v in variable_bounds
        )
        max_val = sum(
            variable_bounds[v]["max"] * contributions[cname].get(v, 0)
            for v in variable_bounds
        )
        constraint_bounds[cname] = {"min": min_val, "max": max_val}

    if str(status) != "HighsModelStatus.kOptimal":
        col_names = list(model.variables.keys())
        iis_variables = [v for v in iis_vars if v in col_names]

        ranked = []
        for cname in iis_constraints:
            bounds_for_c = constraint_bounds[cname]
            spec = model.constraints[cname].dict()
            req_min = spec.get("min", -float("inf"))
            req_max = spec.get("max", float("inf"))
            gap = max(req_min - bounds_for_c["max"], bounds_for_c["min"] - req_max, 0)

            culprit = max(
                {v: contributions[cname].get(v, 0) for v in iis_variables}.items(),
                key=lambda x: x[1],
                default=(None, 0)
            )[0]

            if culprit:
                fix = "raise" if req_min > bounds_for_c["max"] else "lower"
                coeff = contributions[cname].get(culprit, 0) or 1
                delta = gap / coeff if coeff else 0
                ranked.append({
                    "constraint": cname,
                    "gap": round(gap, 6),
                    "fix": f"{fix} {culprit}.{'max' if fix == 'raise' else 'min'}",
                    "delta": round(delta, 6)
                })

        if not iis_constraints:
            for cname, spec in model.constraints.items():
                cb = constraint_bounds.get(cname, {})
                req_min = spec.dict().get("min", -float("inf"))
                req_max = spec.dict().get("max", float("inf"))
                gap = max(req_min - cb.get("max", 0), cb.get("min", 0) - req_max, 0)
                if gap > 0:
                    culprit = max(
                        contributions[cname].items(),
                        key=lambda x: x[1],
                        default=(None, 0)
                    )[0]
                    if culprit:
                        fix = "raise" if req_min > cb.get("max", 0) else "lower"
                        coeff = contributions[cname].get(culprit, 0) or 1
                        delta = gap / coeff if coeff else 0
                        ranked.append({
                            "constraint": cname,
                            "gap": round(gap, 6),
                            "fix": f"{fix} {culprit}.{'max' if fix == 'raise' else 'min'}",
                            "delta": round(delta, 6)
                        })

        relax_suggestions = _suggest_relaxations(model)
        debug = {
            "reason": "HiGHS returned infeasible model",
            "model_status": str(status),
            "constraints": {k: v.dict() for k, v in model.constraints.items()},
            "variable_bounds": variable_bounds,
            "contributions": contributions,
            "hint_summary": (
                "IIS-based bottlenecks detected" if iis_constraints else
                "No IIS detected. Heuristic fallback used."
            ) + " Review constraint limits or variable bounds.",
            "hint_ranked": ranked[:3],
            "hint_relaxations": relax_suggestions,
        }

        if (
            str(status) == "HighsModelStatus.kInfeasible"
            and model.allow_relaxation
            and iteration_depth == 0
            and relax_suggestions
        ):
            relaxed_model = apply_relaxations(model, {"hint_relaxations": relax_suggestions})
            return solve_model(relaxed_model, iteration_depth=iteration_depth + 1)

        return {
            "vars": {},
            "cost": 0.0,
            "infeasible": True,
            "debug": debug,
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
