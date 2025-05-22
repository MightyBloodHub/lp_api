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

