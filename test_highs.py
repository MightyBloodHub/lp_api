from highspy import Highs
import numpy as np
from scipy.sparse import csc_matrix

# Ingredient profiles
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.003,  "p": 0.0065, "lys": 0.029,  "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.51,  "me": 2800, "fat": 0.18,  "ca": 0.075,  "p": 0.012,  "lys": 0.028,  "met": 0.0075},
    "premix":   {"cost": 0.9,   "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.15,   "p": 0.1185,"lys": 0.026,  "met": 0.085},
    "limestone":{"cost": 0.015, "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.38,   "p": 0.0,   "lys": 0.0,    "met": 0.0}
}

order = ["corn", "soybean", "bsf", "premix", "limestone"]
n = len(order)

# Constraint specs
nutrients = [
    ("cp", 0.22, 0.24),
    ("fat", 0.045, 0.06),
    ("ca", 0.009, 0.011),
    ("p", 0.0045, 0.0055),
    ("lys", 0.011, 0.013),
    ("met", 0.005, 0.0065),
    ("me", 2900, 3100),
    ("total", 1.0, 1.0)
]

rows = []
lowers = []
uppers = []

# Assemble constraint matrix
for name, lb, ub in nutrients:
    if name == "total":
        row = [1.0] * n
    elif name == "me":
        row = [ingredients[i]["me"] / 1000.0 for i in order]
        lb /= 1000.0
        ub /= 1000.0
    else:
        row = [ingredients[i][name] for i in order]
    rows.append(row)
    lowers.append(lb)
    uppers.append(ub)

A = np.array(rows)
costs = np.array([ingredients[i]["cost"] for i in order])
lb = np.zeros(n)
ub = np.ones(n)

# Apply fixed limits
lb[order.index("premix")] = ub[order.index("premix")] = 0.005
lb[order.index("limestone")] = ub[order.index("limestone")] = 0.047
ub[order.index("bsf")] = 0.04

# Convert A to CSC matrix
sparse = csc_matrix(A.T)  # ← correct orientation: constraints as rows
starts = sparse.indptr.astype(np.int32)
index = sparse.indices.astype(np.int32)
values = sparse.data.astype(np.float64)

# Build and solve model
model = Highs()
print("Sparse matrix shape:", sparse.shape)
print("Nonzeros:", len(values))
print("starts:", starts)
print("index:", index)

model.addCols(n, costs, lb, ub, len(values), starts, index, values)
model.addRows(len(lowers), np.array(lowers), np.array(uppers),
              0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))
model.run()

sol = model.getSolution()
x = sol.col_value

print("\nFeed Mix:")
for i, v in enumerate(x):
    print(f"  {order[i]}: {v:.4f}")
print(f"\nCost per kg: {np.dot(costs, x):.4f}")
