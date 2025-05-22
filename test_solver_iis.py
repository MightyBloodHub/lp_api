from solver import solve_model
from models import LPModel

def test_infeasible_model_generates_iis_hint():
    model = LPModel(
        optimize="cost",
        opType="min",
        constraints={
            "cp": { "min": 0.3 },    # too high for available vars
            "totalMix": { "equal": 1.0 }
        },
        variables={
            "corn": {
                "cost": 0.1, "cp": 0.08, "min": 0, "max": 1, "totalMix": 1.0
            },
            "soybean": {
                "cost": 0.12, "cp": 0.2, "min": 0, "max": 0.5, "totalMix": 1.0
            }
        }
    )

    result = solve_model(model)

    assert result["infeasible"] is True
    assert "hint_summary" in result["debug"]
    assert isinstance(result["debug"]["hint_ranked"], list)
    assert len(result["debug"]["hint_ranked"]) > 0

    print("IIS Hints:", result["debug"]["hint_ranked"])
    print(result)
