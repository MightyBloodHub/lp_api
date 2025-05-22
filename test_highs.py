from highspy import Highs
import numpy as np

model = Highs()

# Maximize 1*x1 + 2*x2 → minimize -1*x1 -2*x2
# Subject to: x1 + x2 ≤ 10

# Variables
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint bounds
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Constraint matrix (CSC: column-based)
start = np.array([0, 1, 2], dtype=np.int32)     # col start index
index = np.array([0, 0], dtype=np.int32)        # row indices
value = np.array([1.0, 1.0], dtype=np.float64)  # matrix values

# Add all columns with matrix
model.addCols(
    num_col=2,
    costs=obj,
    lower=col_lower,
    upper=col_upper,
    num_new_nz=len(value),
    starts=start,
    indices=index,
    values=value
)

# Add rows with only bounds—NO matrix again!
model.addRows(
    num_new_row=1,
    lower=row_lower,
    upper=row_upper,
    num_new_nz=0,
    starts=np.array([], dtype=np.int32),
    indices=np.array([], dtype=np.int32),
    values=np.array([], dtype=np.float64)
)

# Solve
model.run()

# Show result
print("Solution:", model.getSolution().col_value)
