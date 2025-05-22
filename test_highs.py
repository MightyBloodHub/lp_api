from highspy import Highs
import numpy as np
from scipy.sparse import csr_matrix

# Ingredient data
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.003,  "p": 0.0065, "lys": 0.029,  "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.51,  "me": 2800, "fat": 0.18,  "ca": 0.075,  "p": 0.012,  "lys": 0.028,  "met": 0.0075},
    "premix":   {"cost": 0.9,   "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.15,   "p": 0.1185,"lys": 0.026,  "met": 0.085},
    "limestone":{"cost": 0.015, "cp": 0.0,   "me": 0,    "fat": 0.0,   "ca": 0.38,   "p": 0.0,   "lys": 0.0,    "met": 0.0}
}

order = ["corn", "soybean", "bsf", "premix", "limestone"]
n = len(order)

# Constraints for Starter phase
constraints = {
    "cp":   (0.22, 0.24),
    "me":   (2900, 3100),
    "fat":  (0.045, 0.06),
    "ca":   (0.009, 0.011),
    "p":    (0.0045, 0.0055),
    "lys":  (0.011, 0.013),
    "met":  (0.005, 0.0065),
    "total": (1.0, 1.0),
    "premix_fixed": 0.005,
    "limestone_fixed": 0.047,
    "bsf_max": 0.04
}

# Assemble constraint matrix row-wise
rows = []
row_lowers = []
row_uppers = []

def add_row(nutrient, lb, ub):
    rows.append([ingredients[i][nutrient] for i in order])
    row_lowers.append(lb)
    row_uppers.append(ub)

# Add nutrient constraints
add_row("cp", *constraints["cp"])
add_row("fat", *constraints["fat"])
add_row("ca", *constraints["ca"])
add_row("p", *constraints["p"])
add_row("lys", *constraints["lys"])
add_row("met", *constraints["met"])

# Energy constraint (scale to per-kg)
add_row("me", constraints["me"][0]/1000.0, constraints["me"][1]/1000.0)

# Total mix = 1
add_row(None, 1.0, 1.0)
rows[-1] = [1.0] * n

# Convert to sparse column-wise matrix
dense = np.array(rows, dtype=np.float64)
sparse = csr_matrix(dense.T)
starts = np.array(sparse.indptr, dtype=np.int32)
index = np.array(sparse.indices, dtype=np.int32)
values = np.array(sparse.data, dtype=np.float64)

# LP bounds
costs = np.array([ingredients[i]["cost"] for i in order])
lower_bounds = np.zeros(n)
upper_bounds = np.ones(n)
upper_bounds[order.index("premix")] = constraints["premix_fixed"]
lower_bounds[order.index("premix")] = constraints["premix_fixed"]
upper_bounds[order.index("limestone")] = constraints["limestone_fixed"]
lower_bounds[order.index("limestone")] = constraints["limestone_fixed"]
upper_bounds[order.index("bsf")] = constraints["bsf_max"]

# LP model
model = Highs()

# Add cols (ingredients with nutrient matrix)
model.addCols(
    n,
    costs,
    lower_bounds,
    upper_bounds,
    len(values),
    starts,
    index,
    values
)

# Add rows (constraints only with bounds)
model.addRows(
    len(row_lowers),
    np.array(row_lowers, dtype=np.float64),
    np.array(row_uppers, dtype=np.float64),
    0,
    np.array([], dtype=np.int32),
    np.array([], dtype=np.int32),
    np.array([], dtype=np.float64)
)

# Solve
model.run()
sol = model.getSolution()

# Output
print("\nFeed Mix:")
for i, val in enumerate(sol.col_value):
    print(f"  {order[i]}: {val:.4f}")
print(f"\nCost per kg: {np.dot(costs, sol.col_value):.4f}")
