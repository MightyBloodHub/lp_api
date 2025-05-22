from fastapi import FastAPI, Request
from pydantic import BaseModel
from highs import Highs

app = FastAPI()

class LPRequest(BaseModel):
    objective: dict
    constraints: dict
    variables: dict

@app.post("/solve")
async def solve_lp(data: LPRequest):
    # Example response; real LP parsing comes next
    return {"message": "LP received", "objective": data.objective}
