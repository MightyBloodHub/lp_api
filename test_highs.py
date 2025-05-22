from highspy import Highs
import numpy as np
from scipy.sparse import csc_matrix

# -------------------------------
# Feed ingredient definitions
# -------------------------------
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.0030, "p": 0.0065, "lys": 0.0290, "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.510, "me": 2800, "fat": 0.180, "ca": 0.0750, "p": 0.0120, "lys": 0.0280, "met": 0.0075},
    "premix":   {"cost": 0.900, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.1500, "p": 0.1185, "lys": 0.0260, "met": 0.0850},
    "limestone":{"cost": 0.015, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.3800, "p": 0.0000, "lys": 0.0000, "met": 0.0000}
}
order = list(ingredients.keys())
n_vars = len(order)

# -------------------------------
# Starter phase constraints
# -------------------------------
constraints = [
    ("cp",   0.22, 0.24),
    ("fat",  0.045, 0.06),
    ("ca",   0.009, 0.011),
    ("p",    0.0045, 0.0055),
    ("lys",  0.011, 0.013),
    ("met",  0.005, 0.0065),
    ("me",   2900, 3100),  # kcal/kg, will scale to kcal/g
    ("total", 1.0, 1.0)
]

rows = []
lower_bounds = []
upper_bounds = []

# -------------------------------
# Build constraint matrix (dense)
# -------------------------------
for key, lb, ub in constraints:
    if key == "total":
        row = [1.0] * n_vars
    elif key == "me":
        row = [ingredients[i]["me"] / 1000.0 for i in order]
        lb /= 1000.0
        ub /= 1000.0
    else:
        row = [ingredients[i][key] for i in order]

    rows.append(row)
    lower_bounds.append(lb)
    upper_bounds.append(ub)

A_dense = np.array(rows, dtype=np.float64)
costs = np.array([ingredients[i]["cost"] for i in order], dtype=np.float64)

# -------------------------------
# Bounds per ingredient
# -------------------------------
lb = np.zeros(n_vars, dtype=np.float64)
ub = np.ones(n_vars, dtype=np.float64)

ub[order.index("bsf")] = 0.04
lb[order.index("premix")] = ub[order.index("premix")] = 0.005
lb[order.index("limestone")] = ub[order.index("limestone")] = 0.047

# -------------------------------
# Convert dense A to valid CSC
# -------------------------------
# Build sparse matrix safely
sparse_safe = lil_matrix((len(constraints), n_vars), dtype=np.float64)
for row_idx, (key, lb, ub) in enumerate(constraints):
    if key == "total":
        sparse_safe[row_idx, :] = 1.0
    elif key == "me":
        sparse_safe[row_idx, :] = [ingredients[i]["me"] / 1000.0 for i in order]
    else:
        sparse_safe[row_idx, :] = [ingredients[i][key] for i in order]

sparse = sparse_safe.tocsc()
starts = sparse.indptr.astype(np.int32)
index = sparse.indices.astype(np.int32)
values = sparse.data.astype(np.float64)
# -------------------------------
# Debugging output
# -------------------------------
print("Sparse matrix shape:", sparse.shape)
print("Nonzeros:", len(values))
print("starts:", starts)
print("index:", index)
print("costs:", costs)
print("lb:", lb)
print("ub:", ub)
print("Row bounds:", list(zip(lower_bounds, upper_bounds)))
assert starts[-1] == len(values), "Mismatch: last index of starts != number of values"

# -------------------------------
# Build and solve LP
# -------------------------------
model = Highs()
model.addCols(n_vars, costs, lb, ub, len(values), starts, index, values)
model.addRows(len(lower_bounds),
              np.array(lower_bounds, dtype=np.float64),
              np.array(upper_bounds, dtype=np.float64),
              0,
              np.array([], dtype=np.int32),
              np.array([], dtype=np.int32),
              np.array([], dtype=np.float64))
model.run()

sol = model.getSolution()
x = sol.col_value

# -------------------------------
# Print results
# -------------------------------
print("\nFeed Mix:")
for i, val in enumerate(x):
    print(f"  {order[i]}: {val:.4f}")
print(f"\nCost per kg: {np.dot(costs, x):.4f}")
