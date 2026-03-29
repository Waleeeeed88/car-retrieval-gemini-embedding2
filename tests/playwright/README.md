# Playwright Smoke Tests

These tests validate the Streamlit app as a browser user would see it.

## Start the app first

```powershell
cd car_retrieval
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Default URL:

```text
http://127.0.0.1:8501
```

Override with:

```powershell
$env:CAR_RETRIEVAL_UI_URL="http://127.0.0.1:8501"
```

## Run

```powershell
pytest tests/playwright -m ui
```

## What this browser suite checks

- page loads
- advanced settings are visible
- query mode selector renders
- text input renders
- result images and details are visible after search
- upload inputs accept local files

## What it does not judge

- semantic relevance of results
- business quality of finance/spec snippets
- whether a top result is the best possible car

Use `tests/manual` for those checks.
