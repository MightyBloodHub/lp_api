from highspy import Highs
import numpy as np

model = Highs()

# Problem dimensions
num_vars = 2
num_constraints = 1

# Objective: maximize 1*x1 + 2*x2 → minimize -1*x1 -2*x2
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Sparse matrix (row-wise): one row, two entries
row_start = np.array([0, 2], dtype=np.int32)  # row 0 starts at 0, ends at 2
row_index = np.array([0, 1], dtype=np.int32)  # uses col 0 and col 1
row_value = np.array([1.0, 1.0], dtype=np.float64)

# Add variables
model.addCols(num_vars, obj, col_lower, col_upper)

# Add constraint rows (with sparse matrix)
model.addRows(num_constraints, row_lower, row_upper, len(row_value), row_start, row_index, row_value)

# Solve
model.run()

# Output
print("Solution:", model.getSolution().col_value)
