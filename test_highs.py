from highspy import Highs
import numpy as np

model = Highs()

# LP dimensions
num_vars = 2
num_constraints = 1
num_nz = 2  # non-zero values in matrix

# Objective: maximize 1*x1 + 2*x2 => minimize -1*x1 -2*x2
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Matrix in CSC (column format)
start = np.array([0, 1, 2], dtype=np.int32)     # start index per column
index = np.array([0, 0], dtype=np.int32)        # row indices for col0, col1
value = np.array([1.0, 1.0], dtype=np.float64)  # coefficients

# Add all variables and constraints at once
model.addCols(num_vars, obj, col_lower, col_upper, num_nz, start, index, value)
model.addRows(num_constraints, row_lower, row_upper, 0, np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64))

# Solve
model.run()

# Show result
solution = model.getSolution().col_value
print("Solution:", solution)
