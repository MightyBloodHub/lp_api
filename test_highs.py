from highspy import Highs, HighsLp

lp = HighsLp()

# Define objective function: max 1*x1 + 2*x2
lp.num_col = 2
lp.col_cost = [1.0, 2.0]
lp.col_lower = [0.0, 0.0]
lp.col_upper = [1e20, 1e20]

# Define constraint: x1 + x2 <= 10
lp.num_row = 1
lp.row_lower = [-1e20]
lp.row_upper = [10.0]
lp.a_matrix.start = [0, 1, 2]
lp.a_matrix.index = [0, 0]
lp.a_matrix.value = [1.0, 1.0]

solver = Highs()
solver.passModel(lp)
solver.run()

sol = solver.getSolution().col_value
print("Solution:", sol)
