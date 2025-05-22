from highspy import Highs
import numpy as np
from scipy.sparse import lil_matrix

# --------------------------------------
# Ingredient definitions
# --------------------------------------
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.0030, "p": 0.0065, "lys": 0.0290, "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.510, "me": 2800, "fat": 0.180, "ca": 0.0750, "p": 0.0120, "lys": 0.0280, "met": 0.0075},
    "premix":   {"cost": 0.900, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.1500, "p": 0.1185, "lys": 0.0260, "met": 0.0850},
    "limestone":{"cost": 0.015, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.3800, "p": 0.0000, "lys": 0.0000, "met": 0.0000}
}
order = list(ingredients.keys())
n_vars = len(order)

# --------------------------------------
# Nutrient constraints for Starter phase
# --------------------------------------
constraints = [
    ("cp",   0.22, 0.24),
    ("fat",  0.045, 0.06),
    ("ca",   0.009, 0.011),
    ("p",    0.0045, 0.0055),
    ("lys",  0.011, 0.013),
    ("met",  0.005, 0.0065),
    ("me",   2900, 3100),
    ("total", 1.0, 1.0)
]

lower_bounds = []
upper_bounds = []

# --------------------------------------
# Sparse matrix assembly (safe method)
# --------------------------------------
print("\n🔍 Building sparse matrix with lil_matrix...")
sparse_safe = lil_matrix((len(constraints), n_vars), dtype=np.float64)

for row_idx, (nutrient, lb_val, ub_val) in enumerate(constraints):
    if nutrient == "total":
        sparse_safe[row_idx, :] = 1.0
    elif nutrient == "me":
        sparse_safe[row_idx, :] = [ingredients[i]["me"] / 1000.0 for i in order]
        lb_val /= 1000.0
        ub_val /= 1000.0
    else:
        sparse_safe[row_idx, :] = [ingredients[i][nutrient] for i in order]

    lower_bounds.append(lb_val)
    upper_bounds.append(ub_val)

# Convert to CSC
sparse = sparse_safe.tocsc()
starts = sparse.indptr.astype(np.int32)
index = sparse.indices.astype(np.int32)
values = sparse.data.astype(np.float64)

# --------------------------------------
# Variable bounds & cost vector
# --------------------------------------
lb_array = np.zeros(n_vars, dtype=np.float64)
ub_array = np.ones(n_vars, dtype=np.float64)

# Apply fixed and max constraints
ub_array[order.index("bsf")] = 0.04
premix_idx = order.index("premix")
limestone_idx = order.index("limestone")
lb_array[premix_idx] = ub_array[premix_idx] = 0.005
lb_array[limestone_idx] = ub_array[limestone_idx] = 0.047

cost_vector = np.array([ingredients[i]["cost"] for i in order], dtype=np.float64)

# --------------------------------------
# Deep debugging before solve
# --------------------------------------
print("\n📊 Sparse Matrix Summary")
print("-" * 40)
print("Shape:", sparse.shape)
print("Nonzeros:", len(values))
print("Starts (column ptrs):", starts.tolist())
print("Index (row indices):", index.tolist())
print("Values:", values.tolist())
print("\n🧮 Cost vector:", cost_vector.tolist())
print("📌 Lower bounds:", lb_array.tolist())
print("📌 Upper bounds:", ub_array.tolist())
print("📐 Constraint bounds:", list(zip(lower_bounds, upper_bounds)))

assert starts[-1] == len(values), "ERROR: CSC index mismatch!"
assert sparse.shape == (len(constraints), n_vars), "ERROR: Matrix shape mismatch!"

# --------------------------------------
# Build and solve LP model
# --------------------------------------
model = Highs()
model.addCols(n_vars, cost_vector, lb_array, ub_array, len(values), starts, index, values)
model.addRows(len(lower_bounds),
              np.array(lower_bounds, dtype=np.float64),
              np.array(upper_bounds, dtype=np.float64),
              0,
              np.array([], dtype=np.int32),
              np.array([], dtype=np.int32),
              np.array([], dtype=np.float64))

print("\n🚀 Solving LP...")
model.run()
status = model.getModelStatus()
print("✅ Model status:", status)

# --------------------------------------
# Output results
# --------------------------------------
sol = model.getSolution()
x = sol.col_value

if len(x) == 0:
    print("❌ No solution found. LP was likely infeasible.")
else:
    print("\n🥣 Feed Mix Solution:")
    for i, val in enumerate(x):
        print(f"  {order[i]}: {val:.4f}")
    print(f"\n💰 Cost per kg: {np.dot(cost_vector, x):.4f}")
