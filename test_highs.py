from highspy import Highs  # ✅ correct

model = Highs()
model.addVar(0, 1e20, 1)
model.addVar(0, 1e20, 2)
model.addConstr([1, 1], "<=", 10)
model.run()
print(model.getSolution())

