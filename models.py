from pydantic import BaseModel
from typing import Dict, Optional, List, Literal


class ConstraintRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    equal: Optional[float] = None


class LPModel(BaseModel):
    optimize: str
    opType: Literal["min", "max"]
    constraints: Dict[str, ConstraintRange]
    variables: Dict[str, Dict[str, float]]
    allow_relaxation: bool = False


class LPRequest(BaseModel):
    models: List[LPModel]


class LPSolution(BaseModel):
    vars: Dict[str, float]
    cost: float
    infeasible: bool

