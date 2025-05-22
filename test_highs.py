from highspy import Highs
import numpy as np

model = Highs()

# LP: maximize 1*x1 + 2*x2  → minimize -1*x1 -2*x2
# subject to: x1 + x2 <= 10

# Problem dimensions
num_vars = 2
num_constraints = 1
num_nz = 2

# Objective
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint bounds: row 0 => x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Sparse constraint matrix in CSC format (linking cols to rows)
start = np.array([0, 1, 2], dtype=np.int32)    # start of each column
index = np.array([0, 0], dtype=np.int32)       # row indices
value = np.array([1.0, 1.0], dtype=np.float64) # matrix values

# Add both cols and rows with full linkage
model.addCols(num_vars, obj, col_lower, col_upper, num_nz, start, index, value)
model.addRows(num_constraints, row_lower, row_upper, num_nz, start, index, value)

# Solve
model.run()

# Result
print("Solution:", model.getSolution().col_value)
