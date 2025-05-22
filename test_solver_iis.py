from solver import solve_model
from models import LPModel

def test_real_model_iis():
    model = LPModel(
        optimize="cost",
        opType="min",
        constraints={
            "cp": { "min": 0.22, "max": 0.24 },
            "fat": { "min": 0.045, "max": 0.06 },
            "ca": { "min": 0.009, "max": 0.011 },
            "p": { "min": 0.0045, "max": 0.0055 },
            "lys": { "min": 0.011, "max": 0.013 },
            "met": { "min": 0.005, "max": 0.0065 },
            "me": { "min": 2.8, "max": 3.1 },
            "totalMix": { "min": 1.0, "max": 1.05 }
        },
        variables={
            "corn": {
                "cost": 0.104, "cp": 0.085, "me": 3.35, "fat": 0.038,
                "ca": 0.0002, "p": 0.0025, "lys": 0.0026, "met": 0.0018,
                "min": 0, "max": 1, "totalMix": 1.0
            },
            "soybean": {
                "cost": 0.118, "cp": 0.475, "me": 2.75, "fat": 0.015,
                "ca": 0.003, "p": 0.0065, "lys": 0.029, "met": 0.0065,
                "min": 0, "max": 1, "totalMix": 1.0
            },
            "bsf": {
                "cost": 0.602, "cp": 0.51, "me": 2.8, "fat": 0.18,
                "ca": 0.075, "p": 0.012, "lys": 0.028, "met": 0.0075,
                "min": 0, "max": 0.08, "totalMix": 1.0
            },
            "premix": {
                "cost": 0.9, "me": 0, "cp": 0.0, "fat": 0.0,
                "ca": 0.15, "p": 0.1185, "lys": 0.026, "met": 0.085,
                "min": 0.0, "max": 0.0, "totalMix": 1.0
            },
            "limestone": {
                "cost": 0.015, "cp": 0.0, "me": 0, "fat": 0.0,
                "ca": 0.38, "p": 0.0, "lys": 0.0, "met": 0.0,
                "min": 0, "max": 0.1, "totalMix": 1.0
            },
            "virtual_energy_boost": {
                "cost": 99, "me": 4.0, "min": 0, "max": 1, "totalMix": 1.0
            },
            "virtual_protein_boost": {
                "cost": 99, "cp": 0.01, "min": 0, "max": 1, "totalMix": 1.0
            },
            "virtual_aa_boost": {
                "cost": 99, "lys": 0.001, "met": 0.0005,
                "min": 0, "max": 1, "totalMix": 1.0
            },
            "virtual_mineral_boost": {
                "cost": 99, "ca": 0.003, "p": 0.002,
                "min": 0, "max": 1, "totalMix": 1.0
            }
        }
    )

    result = solve_model(model)

    assert result["infeasible"] is True
    assert "hint_ranked" in result["debug"]
    assert isinstance(result["debug"]["hint_ranked"], list)
    assert "hint_relaxations" in result["debug"]
    assert isinstance(result["debug"]["hint_relaxations"], dict)
    print("IIS-Based Hints:", result["debug"]["hint_ranked"])
    print("Relaxations:", result["debug"]["hint_relaxations"])
    print("Summary:", result["debug"]["hint_summary"])
    print("Full debug:", result["debug"])

if __name__ == "__main__":
    test_real_model_iis()
