# Test Suite

This repo uses three testing layers:

- `tests/unit`: fast offline checks for config, loading, ranking, and feature extraction
- `tests/integration`: offline end-to-end checks for CLI, app context loading, and evaluation helpers
- `tests/playwright`: browser smoke tests for a locally running Streamlit app
- `tests/manual`: human-review tasks and checklists

## Install

```powershell
pip install pytest playwright
playwright install
```

## Run

```powershell
pytest tests/unit
pytest tests/integration
pytest tests/playwright -m ui
```

`tests/unit` and `tests/integration` are offline-safe by default.

`tests/playwright` assumes:

- the Streamlit app is already running
- `CAR_RETRIEVAL_UI_URL` points to it if not using `http://127.0.0.1:8501`
- a valid local environment is available if searches need the real embedding model

## Notes

- Synthetic fixture cars live under `tests/fixtures/sample_dataset`
- Fixture assets are intentionally tiny and deterministic
- `evaluate.py` remains the repo’s retrieval evaluation harness; these tests verify that it behaves correctly on fixture data
