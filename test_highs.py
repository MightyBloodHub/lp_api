import highspy
import numpy as np

h = highspy.Highs()
inf = highspy.kHighsInf

# Define variables
h.addVar(0, 4)   # x0: lower bound 0, upper bound 4
h.addVar(1, inf) # x1: lower bound 1, upper bound infinity

# Set objective coefficients
h.changeColCost(0, 1)  # Coefficient for x0
h.changeColCost(1, 1)  # Coefficient for x1

# Add constraints
# Constraint: x1 <= 7
h.addRow(-inf, 7, 1, [1], [1])

# Constraint: 5 <= x0 + 2*x1 <= 15
h.addRow(5, 15, 2, [0, 1], [1, 2])

# Constraint: 6 <= 3*x0 + 2*x1
h.addRow(6, inf, 2, [0, 1], [3, 2])

# Solve the model
h.run()

# Retrieve and print the solution
solution = h.getSolution()
print("Optimal solution:", solution.col_value)
