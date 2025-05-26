from fastapi import FastAPI
from typing import List
from models import LPRequest, LPSolution
from solver import solve_model


app = FastAPI()


@app.post("/solve")
def solve(request: LPRequest):
    results: List[LPSolution] = []
    for model in request.models:
        solution = solve_model(model)
        results.append(solution)
    return {"solutions": results}

