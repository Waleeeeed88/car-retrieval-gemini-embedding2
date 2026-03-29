# Playwright Agent Testing for `car_retrieval`

## Why this repo needs Playwright plus humans

This repo is retrieval-first:

- it builds a multimodal index
- it ranks cars by similarity
- it exposes the workflow through Streamlit

That means quality is split across two layers:

1. deterministic behavior
2. perceived result quality

Deterministic behavior can be automated:

- the app loads
- the form submits
- uploads work
- results render
- detail sections open
- car images appear

Perceived result quality still needs people:

- was the returned car actually the right type?
- was the image useful?
- were the spec and finance snippets helpful?
- did the result feel obviously wrong or misleading?

## What Playwright should automate

Playwright is best used here for:

- page smoke tests
- query mode switching
- upload workflows
- result rendering checks
- image visibility checks
- regression checks after UI changes

It is not the best tool for deciding semantic relevance by itself.

## Recommended workflow

### 1. Run offline code tests first

```powershell
pytest tests/unit
pytest tests/integration
```

### 2. Start the local app

```powershell
cd car_retrieval
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Optional:

```powershell
$env:CAR_RETRIEVAL_UI_URL="http://127.0.0.1:8501"
```

### 3. Run browser smoke tests

```powershell
pytest tests/playwright -m ui
```

### 4. Run human review tasks

Use:

- `tests/manual/golden_tasks.md`
- `tests/manual/human_review_checklist.md`

## Human-in-the-loop review rules

For each golden task, the reviewer should score:

- top-1 relevance
- top-3 usefulness
- image correctness
- finance/spec usefulness
- obvious UI or content issues

Suggested pass rule:

- top result is clearly on-target, or
- one of the top 3 is clearly correct and the UI gives enough context to identify it quickly

Suggested fail rule:

- wrong vehicle class dominates results
- images are missing or obviously wrong
- uploaded-file flows break
- spec or finance detail is empty when it should not be

## How this differs from ordinary browser tests

Ordinary browser tests verify:

- visible elements
- navigation
- clicks and uploads

AI-agent tests add:

- realistic task phrasing
- retrieval-quality review
- repeated regression checks over a fixed task set
- human judgment where exact assertions are too brittle
