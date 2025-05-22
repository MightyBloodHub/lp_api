from highspy import Highs, HighsLp, HighsMatrix
import numpy as np

# Define LP model
lp = HighsLp()

# Objective: Maximize 1*x1 + 2*x2 (we'll convert to minimize -1*x1 - 2*x2)
lp.col_cost = np.array([-1.0, -2.0], dtype=np.float64)  # minimize -obj = maximize obj
lp.col_lower = np.array([0.0, 0.0], dtype=np.float64)
lp.col_upper = np.array([1e20, 1e20], dtype=np.float64)

lp.row_lower = np.array([-1e20], dtype=np.float64)
lp.row_upper = np.array([10.0], dtype=np.float64)

# Constraint matrix: x1 + x2 ≤ 10
# A matrix in Compressed Sparse Column (CSC) format
A = HighsMatrix()
A.num_col = 2
A.num_row = 1
A.start = np.array([0, 1, 2], dtype=np.int32)  # col 0 starts at 0, col 1 at 1
A.index = np.array([0, 0], dtype=np.int32)    # row indices
A.value = np.array([1.0, 1.0], dtype=np.float64)  # coefficients

lp.a_matrix = A

# Solve
solver = Highs()
solver.passModel(lp)
solver.run()

solution = solver.getSolution().col_value
print("Solution:", solution)
