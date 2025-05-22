from highspy import Highs
import numpy as np

model = Highs()

# Define problem: maximize 1*x1 + 2*x2 => minimize -1*x1 -2*x2
# subject to: x1 + x2 <= 10

# Objective
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: one row (x1 + x2 <= 10)
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Sparse matrix in CSC format (2 cols, 1 row)
start = np.array([0, 1, 2], dtype=np.int32)    # col 0 starts at 0, col 1 at 1
index = np.array([0, 0], dtype=np.int32)       # both entries in row 0
value = np.array([1.0, 1.0], dtype=np.float64) # coefficients

# Add columns (with matrix data)
model.addCols(2, obj, col_lower, col_upper, len(value), start, index, value)

# Add row bounds (NO matrix this time!)
model.addRows(1, row_lower, row_upper)

# Solve
model.run()

# Output
print("Solution:", model.getSolution().col_value)
