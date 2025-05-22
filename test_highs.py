from highspy import Highs
import numpy as np

model = Highs()

# LP: maximize 1*x1 + 2*x2 → minimize -1*x1 - 2*x2
# subject to: x1 + x2 <= 10

# Variables
obj = np.array([-1.0, -2.0], dtype=np.float64)
col_lower = np.array([0.0, 0.0], dtype=np.float64)
col_upper = np.array([1e20, 1e20], dtype=np.float64)

# Constraint bounds (row 0: x1 + x2 ≤ 10)
row_lower = np.array([-1e20], dtype=np.float64)
row_upper = np.array([10.0], dtype=np.float64)

# Matrix in CSC format
start = np.array([0, 1, 2], dtype=np.int32)     # start of each column
index = np.array([0, 0], dtype=np.int32)        # row indices
value = np.array([1.0, 1.0], dtype=np.float64)  # coefficients

# Add full model in one step
model.passModelFromColwise(
    num_col=2,
    num_row=1,
    num_nz=2,
    col_cost=obj,
    col_lower=col_lower,
    col_upper=col_upper,
    row_lower=row_lower,
    row_upper=row_upper,
    start=start,
    index=index,
    value=value
)

# Solve
model.run()

# Output
print("Solution:", model.getSolution().col_value)
