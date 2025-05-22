from highspy import Highs, HighsLp
import numpy as np

# Define LP
lp = HighsLp()

# Objective: maximize 1*x1 + 2*x2 → minimize -1*x1 -2*x2
lp.col_cost = np.array([-1.0, -2.0], dtype=np.float64)
lp.col_lower = np.array([0.0, 0.0], dtype=np.float64)
lp.col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint: x1 + x2 <= 10 → row_lower = -inf, row_upper = 10
lp.row_lower = np.array([-1e20], dtype=np.float64)
lp.row_upper = np.array([10.0], dtype=np.float64)

# Sparse matrix in CSC format
lp.a_matrix.start = np.array([0, 1, 2], dtype=np.int32)  # column pointers
lp.a_matrix.index = np.array([0, 0], dtype=np.int32)     # row indices
lp.a_matrix.value = np.array([1.0, 1.0], dtype=np.float64)  # values

# Solve
solver = Highs()
solver.passModel(lp)
solver.run()

# Print result
print("Solution:", solver.getSolution().col_value)
