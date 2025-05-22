import numpy as np
from highspy import Highs
from models import LPModel, LPSolution
from utils import build_sparse_matrix



def solve_model(model: LPModel) -> LPSolution:
    var_order = list(model.variables.keys())
    n_vars = len(var_order)

    cost_vector = np.array([model.variables[v]["cost"] for v in var_order], dtype=np.float64)
    lb = np.array([model.variables[v].get("min", 0.0) for v in var_order], dtype=np.float64)
    ub = np.array([model.variables[v].get("max", 1.0) for v in var_order], dtype=np.float64)

    A, rhs_lo, rhs_hi, _ = build_sparse_matrix(model.constraints, model.variables, var_order)
    starts = A.indptr.astype(np.int32)
    index = A.indices.astype(np.int32)
    values = A.data.astype(np.float64)

    solver = Highs()
    solver.addRows(len(rhs_lo), np.array(rhs_lo), np.array(rhs_hi), 0,
                   np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
    solver.addCols(n_vars, cost_vector, lb, ub, len(values), starts, index, values)

    solver.run()
    status = solver.getModelStatus()

    if str(status) != "HighsModelStatus.kOptimal":
        return LPSolution(vars={}, cost=0.0, infeasible=True)

    sol = solver.getSolution().col_value
    var_result = {var_order[i]: sol[i] for i in range(n_vars)}
    total_cost = float(np.dot(cost_vector, sol))

    return LPSolution(vars=var_result, cost=total_cost, infeasible=False)

