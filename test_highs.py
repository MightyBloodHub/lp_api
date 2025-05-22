from highspy import Highs
import numpy as np

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

# Constraints for "Starter" phase
constraints = {
    "cp":   (0.22, 0.24),
    "me":   (2900, 3100),
    "fat":  (0.045, 0.06),
    "ca":   (0.009, 0.011),
    "p":    (0.0045, 0.0055),
    "lys":  (0.011, 0.013),
    "met":  (0.005, 0.0065),
    "total": (1.0, 1.0),  # sum to 1
    "premix_fixed": 0.005,
    "limestone_fixed": 0.047,
    "bsf_max": 0.04
}

# Matrix assembly
rows = []
row_lowers = []
row_uppers = []

def add_row(coeffs, lb, ub):
    rows.append(coeffs)
    row_lowers.append(lb)
    row_uppers.append(ub)

# Nutrient constraints
for key in ["cp", "fat", "ca", "p", "lys", "met"]:
    vec = [ingredients[i][key] for i in order]
    lb, ub = constraints[key]
    add_row(vec, lb, ub)

# Energy is handled separately due to units
me_vec = [ingredients[i]["me"] for i in order]
me_vec = [v / 1000.0 for v in me_vec]  # scale to per-kg
add_row(me_vec, constraints["me"][0]/1000.0, constraints["me"][1]/1000.0)

# Total mix must equal 1
add_row([1.0] * n, 1.0, 1.0)

# LP setup
model = Highs()

costs = np.array([ingredients[i]["cost"] for i in order])
lower_bounds = np.zeros(n)
upper_bounds = np.ones(n)
upper_bounds[order.index("premix")] = constraints["premix_fixed"]
lower_bounds[order.index("premix")] = constraints["premix_fixed"]
upper_bounds[order.index("limestone")] = constraints["limestone_fixed"]
lower_bounds[order.index("limestone")] = constraints["limestone_fixed"]
upper_bounds[order.index("bsf")] = constraints["bsf_max"]

# Flatten matrix for sparse input
starts = [i for i in range(n+1)]
index = np.arange(len(rows), dtype=np.int32).repeat(n)
values = np.array([row[i] for row in rows for i in range(n)], dtype=np.float64)

# Add cols (ingredients)
model.addCols(n, costs, lower_bounds, upper_bounds, len(values),
              np.array(starts, dtype=np.int32), index, values)

# Convert row bounds
row_lowers = np.array(row_lowers, dtype=np.float64)
row_uppers = np.array(row_uppers, dtype=np.float64)

# Must add rows first with dummy matrix
model.addRows(
    len(rows),
    row_lowers,
    row_uppers,
    0,
    np.array([], dtype=np.int32),
    np.array([], dtype=np.int32),
    np.array([], dtype=np.float64)
)

# Now add columns with actual matrix values
model.addCols(
    n,
    np.array([ingredients[i]["cost"] for i in order]),
    lower_bounds,
    upper_bounds,
    len(values),
    np.array(starts, dtype=np.int32),
    index,
    values
)

# Solve
model.run()
sol = model.getSolution()

# Output
print("\nFeed Mix:")
for i, val in enumerate(sol.col_value):
    print(f"  {order[i]}: {val:.4f}")
print(f"\nCost per kg: {np.dot(costs, sol.col_value):.4f}")
