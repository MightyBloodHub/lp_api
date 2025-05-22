from highspy import Highs
import numpy as np

model = Highs()

# Define problem dimensions
num_vars = 2
num_constraints = 1

# Objective function: maximize 1*x1 + 2*x2 (minimize -1*x1 -2*x2)
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Sparse constraint matrix (CSC format)
start = np.array([0, 1, 2], dtype=np.int32)    # start indices for each column
index = np.array([0, 0], dtype=np.int32)       # row indices for each nonzero
value = np.array([1.0, 1.0], dtype=np.float64) # values of coefficients

# Add columns (variables)
model.addCols(num_vars, obj, col_lower, col_upper, len(value), start, index, value)

# Add rows (constraints)
model.addRows(num_constraints, row_lower, row_upper)

# Solve
model.run()

# Print solution
solution = model.getSolution().col_value
print("Solution:", solution)
