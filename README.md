# Car Retrieval With Gemini Embedding 2

This project builds a multimodal retrieval system for a car dataset using the official Google Gen AI Python SDK (`google-genai`) and the Gemini model `gemini-embedding-2-preview`.

The repository is designed as a standalone handoff. A friend or teammate should be able to clone this repo, point it at a compatible `dataset/` folder, build embeddings, run searches, evaluate retrieval quality, and continue extending the system without needing the rest of the original workspace.

## What This Project Does

- reads a per-car dataset made of `summary.md`, `finance.md`, `spec.pdf`, and one canonical image
- creates multimodal embeddings for text, PDFs, and images
- enriches embeddings with structured vehicle features before indexing
- stores vectors and metadata on disk as NumPy, JSON, and CSV
- supports cosine-similarity search from the CLI or a simple Streamlit GUI
- includes baseline evaluation and robustness testing helpers

## Core Idea

The system maps text, images, and PDFs into one shared embedding space. That means all of these retrieval patterns are possible:

- text to image
- text to PDF
- text to finance summary
- image to summary
- PDF to images
- finance markdown to the matching car

## Current Dataset Assumption

This repo expects an external dataset with this layout:

```text
dataset/
  toyota_camry_2026/
    metadata.json
    summary.md
    finance.md
    spec.pdf
    images/
      front.jpg
  subaru_outback_2025/
    ...
```

Important:

- the dataset is not committed in this repo
- each car currently uses one canonical local image: `images/front.jpg`
- by default the code looks for `../dataset` relative to this repo root
- you can override the dataset path with `CAR_DATASET_ROOT` or `--dataset-root`

## Why The Embeddings Are Feature-Aware

The indexer does not embed raw files only. It also derives structured context and folds that into the embedding text so searches about real car attributes work better.

Current derived signals include:

- identity: make, model, year, market
- vehicle descriptors: category, drivetrain, fuel type, price range when available
- finance signals: MSRP, APR notes, payment notes, incentives, warranty
- spec facts: horsepower, torque, mpg, range, battery size, seating, cargo, towing when found
- color signals: color mentions in text plus dominant-color guesses from the stored image
- image traits: simple visual descriptors such as brightness, saturation, and image orientation

For PDF and image items, the pipeline fuses:

1. the native multimodal Gemini embedding
2. a text embedding of the structured feature context

That keeps the original modality signal while making color and feature queries more searchable.

## Repository Layout

```text
car_retrieval/
  .env.example
  .gitignore
  README.md
  requirements.txt
  config.py
  embed_index.py
  search.py
  evaluate.py
  app.py
  data/
    query_variants.example.json
  utils/
    __init__.py
    embedding_utils.py
    feature_utils.py
    io_utils.py
    retrieval_utils.py
    transform_utils.py
```

Generated files such as `embeddings.npy`, `index_metadata.json`, and evaluation outputs are intentionally ignored by Git.

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
cd car_retrieval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
cd car_retrieval
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your local `.env`

```bash
cp .env.example .env
```

Set at least:

```env
GEMINI_API_KEY=your_gemini_api_key_here
CAR_DATASET_ROOT=../dataset
CAR_RETRIEVAL_OUTPUT_DIR=./data
CAR_RETRIEVAL_TOP_K=5
```

Notes:

- `.env` is ignored by Git
- do not commit real API keys
- the code loads environment variables with `python-dotenv`

## Main Workflows

### Build the index

```bash
python embed_index.py
```

Optional:

```bash
python embed_index.py --dataset-root ../dataset --output-dir ./data --verbose
```

What it writes:

- `data/embeddings.npy`
- `data/index_metadata.json`
- `data/index_metadata.csv`
- `data/embeddings.json`

### Search from the CLI

Text query:

```bash
python search.py --text "blue AWD sedan with good mpg and low monthly payment"
```

Image query:

```bash
python search.py --image ../dataset/subaru_forester_2025/images/front.jpg
```

PDF query:

```bash
python search.py --pdf ../dataset/toyota_camry_2026/spec.pdf
```

Finance markdown query:

```bash
python search.py --finance-file ../dataset/toyota_camry_2026/finance.md
```

Restrict retrieval to one modality:

```bash
python search.py --text "best APR offer" --modality finance
python search.py --text "hybrid sedan" --modality summary
python search.py --image ../path/to/query.jpg --modality image
python search.py --pdf ../path/to/query.pdf --modality pdf
```

### Use the GUI

```bash
streamlit run app.py
```

The GUI supports:

- asking text questions
- uploading an image
- uploading a PDF
- uploading a finance markdown file
- filtering retrieval to one modality
- showing the actual car image for text queries
- showing finance or spec details when the query clearly asks for them

### Run evaluation

Baseline:

```bash
python evaluate.py
```

Robustness:

```bash
python evaluate.py --robustness
```

## Evaluation Metrics

- `Top-1 exact car match`: top result belongs to the same `car_id`
- `Top-3 exact car match`: at least one of the top 3 results belongs to the same `car_id`
- `Average rank of correct car`: average rank of the first correct result
- `Same-category top-1 rate`: top result shares the same category/body type when available

## PDF Handling

Gemini Embedding 2 supports PDFs directly, but public Gemini API documentation limits each PDF embedding request to 6 pages. This project handles that by:

1. splitting each PDF into 6-page chunks
2. embedding each chunk separately
3. averaging and renormalizing the chunk vectors

That keeps PDF retrieval aligned with the model constraint while still storing one final vector per brochure.

## How To Extend The Project

Good next steps for a teammate:

### 1. Improve the dataset

- add richer metadata fields such as explicit `category`, `drivetrain`, `fuel_type`, `body_style`, `trim`, `msrp_numeric`, `horsepower`, `range_miles`, `colors_available`
- replace the one-image-per-car setup with multiple real views per car
- add true query sets for finance-heavy, spec-heavy, and visual queries

### 2. Improve feature extraction

- replace regex-based spec extraction with stronger structured parsers
- add OCR or layout-aware PDF parsing for harder brochures
- improve color detection to focus more tightly on the car body rather than background pixels
- add trim-level or package-level extraction

### 3. Improve retrieval quality

- add hybrid ranking that combines cosine similarity with metadata filters
- add reranking on top retrieved candidates
- experiment with weighting summary, finance, PDF, and image results differently
- add car-level aggregation instead of item-level ranking in the CLI

### 4. Improve the UI

- add a chat-style answer generator over top retrieved results
- add image thumbnails for every result
- add expandable panels for spec facts and finance evidence
- add saved search history

### 5. Productionize

- move artifacts to object storage or a vector database
- add background indexing jobs
- add unit tests around feature extraction and ranking
- log request cost and latency

## Troubleshooting

### `GEMINI_API_KEY is not set`

- make sure `.env` exists
- make sure the key is named exactly `GEMINI_API_KEY`
- restart the shell after editing if needed

### `Run embed_index.py first`

- the saved artifacts are missing
- run `python embed_index.py`

### Search quality seems weak

- confirm the dataset path points to the correct dataset
- rebuild the index after changing feature extraction logic
- inspect `data/index_metadata.json` and `data/embeddings.json`
- test with smaller, clearer queries first

### A PDF embedding fails

- the client retries transient Gemini API failures
- if a brochure still fails repeatedly, rerun indexing later
- inspect the PDF itself for corruption or unusual structure

## Important Files

- `config.py`: settings, paths, model name, defaults
- `embed_index.py`: main indexing pipeline
- `search.py`: CLI search entry point
- `evaluate.py`: evaluation and robustness scripts
- `app.py`: Streamlit GUI
- `utils/embedding_utils.py`: Gemini embedding client wrapper with retry logic
- `utils/feature_utils.py`: color, feature, spec-fact, and image-trait extraction
- `utils/io_utils.py`: file I/O, JSON helpers, PDF splitting, artifact persistence
- `utils/retrieval_utils.py`: cosine ranking helpers
- `utils/transform_utils.py`: robustness transforms for images and text

## Notes For The Next Maintainer

- this repo intentionally starts file-based instead of database-backed
- the dataset sits outside the repo on purpose
- generated embedding artifacts are not versioned
- the current app is a simple retrieval UI, not a full answer-generation system
- if you change feature extraction logic, rebuild the index before judging the result quality

## Current Model And SDK

- SDK: `google-genai`
- model: `gemini-embedding-2-preview`
- client style: `from google import genai` and `client.models.embed_content(...)`

That is the current official SDK pattern used throughout this repository.
