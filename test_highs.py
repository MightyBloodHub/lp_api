from highspy import Highs
import numpy as np
from scipy.sparse import csr_matrix

# Define ingredients and their properties
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.003,  "p": 0.0065, "lys": 0.029,  "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.51,  "me": 2800, "fat": 0.18,  "ca": 0.075,  "p": 0.012,  "lys": 0.028,  "met": 0.0075},
    "premix":   {"cost": 0.9,   "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.15,   "p": 0.1185,"lys": 0.026,  "met": 0.085},
    "limestone":{"cost": 0.015, "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.38,   "p": 0.0,   "lys": 0.0,    "met": 0.0}
}

order = ["corn", "soybean", "bsf", "premix", "limestone"]
n = len(order)

# Define constraints
constraints = [
    ("cp", 0.22, 0.24),
    ("fat", 0.045, 0.06),
    ("ca", 0.009, 0.011),
    ("p", 0.0045, 0.0055),
    ("lys", 0.011, 0.013),
    ("met", 0.005, 0.0065),
    ("me", 2900, 3100),  # energy handled separately
    ("total", 1.0, 1.0)
]

rows = []
lower_bounds = []
upper_bounds = []

for name, lb, ub in constraints:
    if name == "total":
        row = [1.0] * n
    elif name == "me":
        row = [ingredients[i]["me"] / 1000.0 for i in order]  # normalize ME
        lb /= 1000.0
        ub /= 1000.0
    else:
        row = [ingredients[i][name] for i in order]
    rows.append(row)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

# Build sparse matrix (transpose to get column-wise)
dense = np.array(rows)
sparse = csr_matrix(dense.T)

starts = sparse.indptr.astype(np.int32)
index = sparse.indices.astype(np.int32)
values = sparse.data.astype(np.float64)

# Variable bounds
lb = np.zeros(n)
ub = np.ones(n)

# Fixed values
premix_i = order.index("premix")
limestone_i = order.index("limestone")
bsf_i = order.index("bsf")

lb[premix_i] = ub[premix_i] = 0.005
lb[limestone_i] = ub[limestone_i] = 0.047
ub[bsf_i] = 0.04

costs = np.array([ingredients[i]["cost"] for i in order])

# Solve LP
model = Highs()
model.addCols(n, costs, lb, ub, len(values), starts, index, values)
model.addRows(len(lower_bounds), np.array(lower_bounds), np.array(upper_bounds), 0,
              np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
model.run()

solution = model.getSolution().col_value

print("\nFeed Mix:")
for i, v in enumerate(solution):
    print(f"  {order[i]}: {v:.4f}")
print(f"\nCost per kg: {np.dot(costs, solution):.4f}")
