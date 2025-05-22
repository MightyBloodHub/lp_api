import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
from typing import List, Dict


def build_sparse_matrix(constraints: Dict, variables: Dict[str, Dict], var_order: List[str]):
    n_constraints = len(constraints)
    n_vars = len(var_order)
    lower_bounds = []
    upper_bounds = []
    constraint_names = list(constraints.keys())

    mat = dok_matrix((n_constraints, n_vars), dtype=np.float64)

    for row_idx, name in enumerate(constraint_names):
        con = constraints[name]
        if con.equal is not None:
            lower_bounds.append(con.equal)
            upper_bounds.append(con.equal)
        else:
            lower_bounds.append(con.min if con.min is not None else -1e20)
            upper_bounds.append(con.max if con.max is not None else 1e20)

        for col_idx, var in enumerate(var_order):
            val = 1.0 if name == "totalMix" else variables[var].get(name, 0.0)
            mat[row_idx, col_idx] = val

    sparse = csr_matrix(mat).tocsc()
    return sparse, lower_bounds, upper_bounds, constraint_names

def analyze_infeasible(debug: dict) -> str:
    """Return a plain-language hint about the tightest blocking constraint."""
    constraints    = debug["constraints"]
    bounds         = debug["variable_bounds"]
    coeffs         = debug["contributions"]

    best_gap, culprit, hint = -1e9, None, ""
    for cname, spec in constraints.items():
        # Build min/max achievable values given var bounds
        min_val = sum(bounds[v]["min"] * coeffs[cname].get(v, 0) for v in bounds)
        max_val = sum(bounds[v]["max"] * coeffs[cname].get(v, 0) for v in bounds)

        # Required interval
        req_min = spec.get("min", -float("inf"))
        req_max = spec.get("max",  float("inf"))
        equal   = spec.get("equal")

        feas = (equal is not None and min_val <= equal <= max_val) or (
            req_min <= max_val and req_max >= min_val
        )
        if feas:
            continue  # constraint is potentially satisfiable

        # Measure severity: distance from feasible region
        gap = (
            req_min - max_val if max_val < req_min else
            min_val - req_max if min_val > req_max else
            abs(equal - max_val)  # equal constraint gap
        )
        if gap > best_gap:
            best_gap, culprit = gap, cname

    if culprit:
        # Pick variable with the largest coefficient for that constraint
        var = max(coeffs[culprit], key=coeffs[culprit].get)
        if constraints[culprit].get("min") is not None:
            hint = f"{culprit} is too low; raise max of '{var}' or lower the min."
        elif constraints[culprit].get("max") is not None:
            hint = f"{culprit} is too high; lower max of '{var}' or relax the max."
        elif constraints[culprit].get('equal') is not None:
            hint = f"{culprit} can’t hit equality; adjust bounds on '{var}'."

    return hint or "No obvious single-constraint bottleneck detected."
def analyze_infeasible_detailed(debug: dict, top_k: int = 3) -> list:
    constraints = debug["constraints"]
    bounds = debug["variable_bounds"]
    coeffs = debug["contributions"]

    suggestions = []

    for cname, spec in constraints.items():
        # Max achievable value for this constraint
        max_val = sum(bounds[v]["max"] * coeffs[cname].get(v, 0) for v in bounds)
        min_val = sum(bounds[v]["min"] * coeffs[cname].get(v, 0) for v in bounds)

        req_min = spec.get("min", -float("inf"))
        req_max = spec.get("max", float("inf"))
        equal = spec.get("equal")

        gap = None
        direction = None
        # Constraint too low
        if equal is not None:
            if not (min_val <= equal <= max_val):
                gap = equal - max_val if equal > max_val else min_val - equal
                direction = "equal"
        elif max_val < req_min:
            gap = req_min - max_val
            direction = "low"
        elif min_val > req_max:
            gap = min_val - req_max
            direction = "high"

    if gap is not None or (req_min > max_val or req_max < min_val):
        if gap is None:
            soft_gap = max(req_min - max_val, min_val - req_max)
            gap = round(soft_gap, 6)
            direction = "low" if req_min > max_val else "high"

        # Find best variable to adjust
        best_var = max(
            coeffs[cname].items(),
            key=lambda kv: kv[1] if kv[1] > 0 else -float("inf"),
            default=(None, 0)
        )[0]

        if best_var:
            if direction == "low":
                fix = f"raise {best_var}.max"
            elif direction == "high":
                fix = f"lower {best_var}.min"
            elif direction == "equal":
                fix = f"adjust {best_var}.bounds to allow {equal}"
            suggestions.append({
                "constraint": cname,
                "gap": gap,
                "fix": fix
            })

    return sorted(suggestions, key=lambda x: abs(x["gap"]), reverse=True)[:top_k]
