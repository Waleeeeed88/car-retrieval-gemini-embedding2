# Comprehensive Test Plan - Car Retrieval

Date: 2026-03-31
Project: car_retrieval
Document status: Draft for active use

## 1. Purpose

This plan defines how we test the car retrieval system end to end:

- indexing multimodal car data
- embedding text, image, and PDF sources
- ranking by vector similarity
- serving results through CLI and Streamlit UI
- evaluating retrieval quality and robustness

The goal is to catch functional regressions early, keep retrieval behavior stable, and make release decisions with clear evidence.

## 2. System Overview

Main workflow:

1. Dataset is read from `dataset/`.
2. `embed_index.py` extracts signals, builds embeddings, and writes artifacts.
3. Artifacts are stored in `data/` (`embeddings.npy`, `index_metadata.json`, `embeddings.json`, CSV metadata).
4. `search.py` and `app.py` load artifacts and run top-k vector ranking.
5. `evaluate.py` computes baseline and robustness metrics.

Key modules under test:

- `config.py`
- `embed_index.py`
- `search.py`
- `app.py`
- `evaluate.py`
- `utils/embedding_utils.py`
- `utils/feature_utils.py`
- `utils/io_utils.py`
- `utils/retrieval_utils.py`
- `utils/transform_utils.py`

## 3. Test Objectives

1. Verify data ingestion and index artifact creation are correct and reproducible.
2. Verify retrieval ranking logic is correct (filters, exclusion, ordering, top-k).
3. Verify evaluation metrics are computed correctly for baseline and robustness runs.
4. Verify UI and upload flows function as expected.
5. Verify failures are visible and diagnosable (clear errors, warnings, logs).
6. Reduce risk from external model/API behavior through deterministic fixture tests.

## 4. Scope

### In Scope

- Functional testing of indexing, retrieval, evaluation, and UI smoke behavior
- Unit, integration, UI smoke, and manual quality review
- Path/config behavior and artifact integrity checks
- Regression testing before release

### Out of Scope

- Hard performance SLA benchmarking at production scale
- Deep security penetration testing
- Full semantic quality automation (human review still required)
- Vector database ANN benchmarking (current system uses in-memory exact scan)

## 5. Test Strategy

### 5.1 Unit Tests (fast, offline)

Purpose:

- verify core logic in isolation
- verify deterministic behavior without external API dependency

Coverage focus:

- config path resolution
- feature extraction and image profile helpers
- file I/O helpers
- ranking utility behavior and correct-rank logic

Primary path: `tests/unit`

### 5.2 Integration Tests (offline end to end)

Purpose:

- verify module interaction across realistic workflows

Coverage focus:

- CLI search flow
- app context data loading
- baseline evaluation output structure and value bounds

Primary path: `tests/integration`

### 5.3 UI Smoke Tests (Playwright)

Purpose:

- validate browser-level behavior and critical user paths

Coverage focus:

- page load
- form interactions
- text search flow
- upload modes (image/pdf/finance file)
- result and image rendering

Primary path: `tests/playwright`

### 5.4 Manual Quality Review

Purpose:

- evaluate semantic relevance and usefulness where strict assertions are brittle

References:

- `tests/manual/golden_tasks.md`
- `tests/manual/human_review_checklist.md`

### 5.5 Exploratory and Regression

- exploratory checks after major ranking or feature extraction changes
- full regression pass before release candidate tag

## 6. Environments and Dependencies

### Local Test Environment

- OS: Windows (primary dev environment)
- Python virtual environment: `.venv`
- Dependencies installed from `requirements.txt`
- Additional test deps: `pytest`, `playwright`

### Required Runtime Inputs

- Fixture dataset in `tests/fixtures/sample_dataset` for deterministic tests
- Optional real dataset in `dataset/` for manual and robustness evaluation
- `GEMINI_API_KEY` only when running real embedding operations

## 7. Test Data

### Fixture Data

- synthetic, deterministic fixture cars in `tests/fixtures/sample_dataset`
- small file fixtures in `tests/fixtures/sample_files`

### Real Data

- external dataset under `dataset/`
- used for manual validation and realistic retrieval behavior

### Data Integrity Checks

- vector count equals metadata count
- each metadata row has unique `item_id`
- expected modality rows exist per car when source files exist

## 8. Entry and Exit Criteria

### Entry Criteria

- repository is checked out and install completes
- virtual environment activated
- test dependencies installed
- fixture data available

### Exit Criteria (Release Gate)

- all unit tests pass
- all integration tests pass
- UI smoke tests pass or approved exceptions are documented
- no open Critical defects
- High defects have explicit risk acceptance and owner
- manual golden-task review completed for release candidate

## 9. Defect Severity and Triage

- Critical: data loss, system unusable, blocking core flow
- High: incorrect ranking/evaluation logic, broken upload/query mode, major regression
- Medium: non-blocking functional bug, degraded UX, incomplete details rendering
- Low: minor copy/layout/reporting issues

Triage cadence:

- pre-merge triage for failing automated checks
- release triage for unresolved High/Medium defects

## 10. Coverage Matrix (What Must Be Tested)

### Configuration

- relative and absolute path resolution
- default values and env override behavior

### Indexing

- car directory discovery
- missing files handled with warnings
- item record construction per modality
- artifact write correctness and row alignment

### Embeddings and Fusion

- text/image/pdf embedding paths invoked correctly
- weighted/average fusion behavior
- normalization assumptions preserved

### Retrieval

- candidate filtering by modality
- `exclude_item_id` behavior
- ranking order consistency
- top-k slicing correctness

### Evaluation

- per-query metric computation
- summary aggregation math
- baseline by-modality summaries
- robustness delta fields and variant summaries

### UI

- query mode switch behavior
- file upload acceptance
- result list rendering
- details panel content visibility

## 11. Test Execution Commands

Run from repository root after activating `.venv`.

### Core Automated Suite

```powershell
pytest tests/unit
pytest tests/integration
pytest tests/playwright -m ui
```

### Useful Variants

```powershell
pytest
pytest -m unit
pytest -m integration
pytest -m "not ui"
pytest -q
pytest -vv
pytest -x --maxfail=1
pytest -k search
```

### Targeted Examples

```powershell
pytest tests/integration/test_evaluate_baseline.py -q
pytest tests/integration/test_search_cli.py -q
pytest tests/playwright/test_upload_flow.py -m ui -q
```

### App Prerequisite for UI Tests

```powershell
streamlit run app.py
```

Optional URL override:

```powershell
$env:CAR_RETRIEVAL_UI_URL="http://127.0.0.1:8501"
```

## 12. Manual Golden-Task Plan

For each golden task:

1. Execute query/input mode.
2. Capture top-1 and top-3 relevance judgment.
3. Verify image correctness and details usefulness.
4. Record pass/fail and notes in review log.

Pass guidance:

- top result clearly on target, or
- at least one of top 3 clearly correct and easily identifiable

## 13. Metrics and Reporting

Track per run:

- unit pass/fail count
- integration pass/fail count
- UI smoke pass/fail count
- open defects by severity
- manual golden-task pass rate

From `evaluate.py`, track:

- top_1_exact_car_match_rate
- top_3_exact_car_match_rate
- average_rank_of_correct_car
- same_category_top1_rate
- robustness deltas by variant

Suggested release thresholds (starter baseline):

- no regression larger than 5 percentage points in top-1 on baseline set
- no catastrophic regression in robustness summaries
- no Critical defects

## 14. Risks and Mitigations

1. External embedding API instability
   - Mitigation: use fixture-based tests for deterministic checks; isolate live API checks.

2. Data drift in external dataset
   - Mitigation: lock fixture coverage; rerun baseline evaluation after dataset updates.

3. Semantic quality not fully captured by automated tests
   - Mitigation: manual golden tasks and release checklist.

4. UI flakiness in browser automation
   - Mitigation: keep UI tests smoke-level and stable; gate semantics via manual review.

## 15. Release Test Workflow

1. Run unit and integration tests.
2. Rebuild index if indexing logic changed.
3. Run evaluation baseline and compare summaries with previous known-good run.
4. Run UI smoke tests.
5. Complete manual golden-task review.
6. Triage unresolved defects.
7. Approve or block release based on exit criteria.

## 16. Ownership

- Engineering owner: maintain test suite and fix failing automation.
- QA/reviewer owner: execute manual checklist and sign off quality judgment.
- Release owner: enforce exit criteria and risk acceptance decisions.

## 17. Maintenance of This Plan

Update this document when any of the following changes:

- new query mode/modality
- new evaluation metric
- new major dependency or infrastructure
- changed release gates or severity policy
- changed folder or command structure
