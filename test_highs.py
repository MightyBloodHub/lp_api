from highspy import Highs
import numpy as np

model = Highs()

# Define the problem
num_vars = 2
num_constraints = 1

# Objective function: maximize 1*x1 + 2*x2
# HiGHS solves minimization by default, so use -1 and -2
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Matrix in CSC (Compressed Sparse Column) format
start = np.array([0, 1, 2], dtype=np.int32)   # column pointers
index = np.array([0, 0], dtype=np.int32)      # row indices
value = np.array([1.0, 1.0], dtype=np.float64)  # coefficients

# Load problem into model
model.addCols(num_vars, obj, col_lower, col_upper, start, index, value)
model.addRows(num_constraints, row_lower, row_upper)

# Solve
model.run()

# Get and print solution
sol = model.getSolution().col_value
print("Solution:", sol)
