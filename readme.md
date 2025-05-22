# Linear Programming API

This repository exposes a simple FastAPI service for solving linear programming (LP) problems using the [HiGHS](https://www.highs.dev/) solver. It accepts a list of LP models in JSON format and returns the optimal solution or diagnostic information when the model is infeasible.

## Features

- **Multiple models**: send several LP models in one request and receive a solution for each.
- **Sparse matrix generation**: constraints and variables are transformed into SciPy sparse matrices for efficiency.
- **Infeasibility hints**: if a model cannot be solved, the service attempts to detect an Irreducible Infeasible Set (IIS) using the `highs` CLI and provides suggestions and possible constraint relaxations.

## Installation

1. Install Python 3.8+.
2. Install the required packages:

```bash
pip install fastapi uvicorn highspy numpy scipy
```

3. Make sure the `highs` binary is available in your `PATH`. On many Linux systems it can be installed via the package manager (e.g. `apt-get install highs`).

## Running the server

Run the application with Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000` by default.

## Request format

Send a `POST` request to `/solve` with a JSON body matching the following structure:

```json
{
  "models": [
    {
      "optimize": "cost",
      "opType": "min",
      "constraints": {
        "totalMix": {"equal": 1},
        "c1": {"min": 2}
      },
      "variables": {
        "x": {"c1": 1, "cost": 5, "min": 0},
        "y": {"c1": 1, "cost": 3, "min": 0}
      }
    }
  ]
}
```

- `optimize` is the name of the coefficient (usually `cost`) to optimize.
- `opType` is either `"min"` or `"max"`.
- `constraints` maps constraint names to ranges consisting of `min`, `max` and/or `equal` values.
- `variables` contains variable coefficients for each constraint along with `cost` and optional `min`/`max` bounds.

## Response format

The response contains a list of solutions corresponding to the supplied models:

```json
{
  "solutions": [
    {
      "vars": {"x": 0.5, "y": 0.5},
      "cost": 4.0,
      "infeasible": false,
      "debug": {
        "constraint_residuals": {"totalMix": {"value": 1, "min": 1, "max": 1, "status": "ok"}},
        "virtuals_used": {}
      }
    }
  ]
}
```

When the model is infeasible the `debug` section includes details such as the reason, IIS results and ranked hints on which variable bounds to adjust. Minimal constraint relaxation suggestions are also provided when possible.

## Code overview

- `main.py` defines the FastAPI endpoint `/solve` that iterates over the supplied models and returns their solutions.
- `models.py` declares the Pydantic data structures used for requests and responses.
- `solver.py` builds the sparse matrix representation, invokes the HiGHS solver and generates diagnostics. The key functions are `solve_model` and `_suggest_relaxations`.
- `utils.py` contains helpers for building the sparse constraint matrix.

## License

This project is provided as-is without any warranty. Feel free to use it as a starting point for your own LP-based services.
