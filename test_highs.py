from highspy import Highs
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix

# ----------------------------------------
# Feed ingredient definitions
# ----------------------------------------
ingredients = {
    "corn":     {"cost": 0.104, "cp": 0.085, "me": 3350, "fat": 0.038, "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018},
    "soybean":  {"cost": 0.118, "cp": 0.475, "me": 2750, "fat": 0.015, "ca": 0.0030, "p": 0.0065, "lys": 0.0290, "met": 0.0065},
    "bsf":      {"cost": 0.602, "cp": 0.510, "me": 2800, "fat": 0.180, "ca": 0.0750, "p": 0.0120, "lys": 0.0280, "met": 0.0075},
    "premix":   {"cost": 0.900, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.1500, "p": 0.1185, "lys": 0.0260, "met": 0.0850},
    "limestone":{"cost": 0.015, "cp": 0.000, "me":    0, "fat": 0.000, "ca": 0.3800, "p": 0.0000, "lys": 0.0000, "met": 0.0000}
}
order = list(ingredients.keys())
n_vars = len(order)

# ----------------------------------------
# Constraints for Starter phase
# ----------------------------------------
constraints = [
    ("cp",   0.22, 0.24),
    ("fat",  0.045, 0.06),
    ("ca",   0.009, 0.011),
    ("p",    0.0045, 0.0055),
    ("lys",  0.011, 0.013),
    ("met",  0.005, 0.0065),
    ("me",   2800, 3100),
    ("total", 1.0, 1.0)
]

lower_bounds = []
upper_bounds = []

# ----------------------------------------
# Sparse matrix assembly (column-safe)
# ----------------------------------------
print("\n🔍 Building sparse matrix with dok_matrix (column-safe)...")
sparse_safe = dok_matrix((len(constraints), n_vars), dtype=np.float64)

for row_idx, (nutrient, lb_val, ub_val) in enumerate(constraints):
    lower_bounds.append(lb_val / 1000.0 if nutrient == "me" else lb_val)
    upper_bounds.append(ub_val / 1000.0 if nutrient == "me" else ub_val)
    for col_idx, ing in enumerate(order):
        if nutrient == "total":
            sparse_safe[row_idx, col_idx] = 1.0
        elif nutrient == "me":
            sparse_safe[row_idx, col_idx] = ingredients[ing]["me"] / 1000.0
        else:
            sparse_safe[row_idx, col_idx] = ingredients[ing][nutrient]

sparse = csr_matrix(sparse_safe)
sparse.sum_duplicates()
sparse = sparse.tocsc()

starts = sparse.indptr.astype(np.int32)
index = sparse.indices.astype(np.int32)
values = sparse.data.astype(np.float64)

# ----------------------------------------
# Cost vector and variable bounds
# ----------------------------------------
cost_vector = np.array([ingredients[i]["cost"] for i in order], dtype=np.float64)
lb_array = np.zeros(n_vars, dtype=np.float64)
ub_array = np.ones(n_vars, dtype=np.float64)
ub_array[order.index("bsf")] = 0.15
# lb_array[order.index("premix")] = ub_array[order.index("premix")] = 0.005
# lb_array[order.index("limestone")] = ub_array[order.index("limestone")] = 0.047

# ----------------------------------------
# Debugging summary
# ----------------------------------------
print("\n📊 Sparse Matrix Summary")
print("-" * 40)
print("Shape:", sparse.shape)
print("Nonzeros:", len(values))
print("Starts:", starts.tolist())
print("Index:", index.tolist())
print("Values:", values.tolist())
print("Cost vector:", cost_vector.tolist())
print("Lower bounds:", lb_array.tolist())
print("Upper bounds:", ub_array.tolist())
print("Constraint bounds:", list(zip(lower_bounds, upper_bounds)))

assert starts[-1] == len(values), "❌ CSC index mismatch!"
assert all(starts[i] <= starts[i+1] for i in range(len(starts)-1)), "❌ starts not monotonic!"
assert sparse.shape == (len(constraints), n_vars), "❌ Matrix shape mismatch!"

# ----------------------------------------
# Build and solve model
# ----------------------------------------
model = Highs()

# Add rows first
model.addRows(
    len(lower_bounds),
    np.array(lower_bounds, dtype=np.float64),
    np.array(upper_bounds, dtype=np.float64),
    0,
    np.array([], dtype=np.int32),
    np.array([], dtype=np.int32),
    np.array([], dtype=np.float64)
)

# Then add columns with matrix
model.addCols(
    n_vars,
    cost_vector,
    lb_array,
    ub_array,
    len(values),
    starts,
    index,
    values
)

print("\n🚀 Solving LP...")
status = model.run()
model_status = model.getModelStatus()
print("🔧 Solver status:", status)
print("✅ Model status:", model_status)

# ----------------------------------------
# Output solution
# ----------------------------------------
sol = model.getSolution()
x = sol.col_value

if not x or len(x) == 0:
    print("❌ No solution found. LP may be infeasible.")
else:
    print("\n🥣 Feed Mix Solution:")
    for i, val in enumerate(x):
        print(f"  {order[i]}: {val:.4f}")
    print(f"\n💰 Cost per kg: {np.dot(cost_vector, x):.4f}")
