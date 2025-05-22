from highspy import Highs

model = Highs()

# Add two variables with bounds only
model.addVar(0, 1e20)  # x1
model.addVar(0, 1e20)  # x2

# Set objective coefficients
model.changeObjectiveCoefficient(0, 1)  # Coeff for x1
model.changeObjectiveCoefficient(1, 2)  # Coeff for x2

# Add constraint: x1 + x2 ≤ 10
model.addRow([0, 1], [1.0, 1.0], -1e20, 10)

# Run solver
model.run()

# Print result
solution = model.getSolution().col_value
print("Solution:", solution)
