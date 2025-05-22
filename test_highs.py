from highspy import Highs
import numpy as np

model = Highs()

# Problem definition
# Maximize 1*x1 + 2*x2  → minimize -1*x1 - 2*x2
# Subject to x1 + x2 ≤ 10

# Define objective
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Sparse matrix for cols: column-wise CSC
col_start = np.array([0, 1, 2], dtype=np.int32)   # 2 columns
col_index = np.array([0, 0], dtype=np.int32)      # row indices
col_value = np.array([1.0, 1.0], dtype=np.float64)

# Add variables with matrix
model.addCols(
    2, obj, col_lower, col_upper,
    len(col_value), col_start, col_index, col_value
)

# Define constraint bounds: row 0 => x1 + x2 <= 10
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Sparse matrix for rows: same data but in row format (only 1 row)
row_start = np.array([0, 2], dtype=np.int32)
row_index = np.array([0, 1], dtype=np.int32)
row_value = np.array([1.0, 1.0], dtype=np.float64)

# Add row with matrix
model.addRows(
    1, row_lower, row_upper,
    len(row_value), row_start, row_index, row_value
)

# Solve
model.run()

# Output
print("Solution:", model.getSolution().col_value)
