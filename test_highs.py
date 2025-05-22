from highspy import Highs

model = Highs()

# Define objective coefficients, lower bounds, upper bounds
obj = [1.0, 2.0]
lower_bounds = [0.0, 0.0]
upper_bounds = [1e20, 1e20]

# Add variables (columns) with bounds and objective
model.addCols(obj, lower_bounds, upper_bounds)

# Add constraint: x1 + x2 ≤ 10
coefficients = [1.0, 1.0]
column_indices = [0, 1]
model.addRow(column_indices, coefficients, -1e20, 10)

# Run solver
model.run()

# Print solution
solution = model.getSolution().col_value
print("Solution:", solution)
