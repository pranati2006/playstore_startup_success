# Startup App Success Analyzer

A complete ML/NLP project that evaluates a startup app idea against **live Google Play competitors**.

The project combines:

1. Live Google Play metadata collection
2. Semantic embeddings of full app descriptions
3. Cosine-similarity competitor retrieval
4. Market feature engineering from similar apps
5. A success target based on installs + reviews
6. Multiple ML classifiers
7. Automatic best-model selection
8. A Streamlit UI for startup idea analysis

---

## 1. Project idea

A founder enters a proposed app description such as:

> An AI-powered fitness coach that creates personalized workouts, tracks nutrition and adapts routines from user progress.

The system:

```text
Startup description
        |
        v
Sentence Transformer embedding
        |
        v
Semantic nearest neighbors
        |
        v
Comparable Google Play apps
        |
        +--> competitor installs/reviews/ratings
        |
        +--> competition density
        |
        +--> monetization patterns
        |
        v
Trained ML model
        |
        v
Success-profile score
```

The system does **not** use the proposed startup's installs or reviews, because those do not exist before launch.

---

## 2. What "success" means

For each historical Google Play app, the project calculates:

```text
Install percentile = percentile rank of log(1 + installs)
Review percentile  = percentile rank of log(1 + reviews)

Success score =
    0.70 * Install percentile
  + 0.30 * Review percentile
```

Percentiles are calculated within the app's genre when that genre has enough observations. Otherwise, global percentiles are used.

The default label is:

```text
success = 1 when success_score >= 0.75
success = 0 otherwise
```

This makes success **relative to comparable market performance**, rather than using an arbitrary rule such as "1 million installs".

You can change the weights and threshold in `src/config.py`.

---

## 3. Important interpretation

The output is best interpreted as:

> "How closely does this proposed app resemble the profile of currently successful Google Play apps, given its concept, launch settings and semantic competitive market?"

It is **not** a causal forecast or guarantee that a startup will achieve a specific number of installs.

Current Play Store data is a snapshot and contains survivorship, age, marketing, brand and geography effects that cannot be fully observed from public metadata.

---

## 4. Data collected

The scraper stores available fields such as:

- App ID
- App title
- Full description
- Short summary
- Genre/category
- Installs
- Real installs when exposed by the scraper
- Rating
- Rating count
- Review count
- Free/paid
- Price
- In-app purchases
- Ads
- Content rating
- Developer
- Release date
- Last update
- Version
- Store URL

The exact fields Google exposes can change over time. The code tolerates many missing values.

---

## 5. Data collection strategy

`google-play-scraper` search results are limited, so the project uses a broad keyword list.

`keywords.txt` contains queries such as:

```text
fitness
education
finance
productivity
travel
shopping
health
ai assistant
ai fitness
...
```

For every query, the collector:

1. discovers app IDs,
2. removes duplicate app IDs,
3. fetches full app details,
4. periodically checkpoints the CSV.

To improve dataset size and diversity, add more query terms to `keywords.txt`.

For a serious report, aim for **several thousand unique apps** rather than only a few hundred.

---

## 6. Semantic embeddings

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Descriptions are encoded into normalized dense vectors.

Normalized embeddings allow cosine similarity to be computed efficiently.

For every historical app, its own row is excluded when building semantic-neighborhood features.

---

## 7. Market features

For the top-K semantically similar apps, the project derives:

- mean semantic similarity
- median competitor installs
- median competitor reviews
- mean competitor rating
- percentage of competitors that are free
- percentage offering IAP
- percentage containing ads
- percentage with historical success label
- number of very close competitors (`similarity >= 0.70`)

These features represent the app's **semantic market environment**.

---

## 8. Candidate features

The ML model receives:

### Text representation

- full semantic embedding

### Proposed launch settings

- category/genre
- content rating
- free/paid
- price
- IAP
- ads

### Description engineering

- character count
- word count
- summary length

### Semantic market features

- competitor statistics listed above

The candidate's own installs and reviews are intentionally excluded from model inputs.

---

## 9. Models compared

The training script evaluates:

- Logistic Regression
- Random Forest
- Extra Trees

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix

The model with the highest ROC-AUC (F1 as tie-breaker) is selected. For evaluation, test apps build their semantic-market features only from the training pool, so held-out app outcomes do not leak through neighborhood statistics. The selected model is then retrained on all historical apps for deployment.

The UI returns one final prediction from the selected model, while `artifacts/model_metrics.json` preserves every model's result for the report.

---

## 10. Folder structure

```text
playstore_startup_success/
|
|-- app.py
|-- run_pipeline.py
|-- keywords.txt
|-- requirements.txt
|-- sample_startup.json
|
|-- src/
|   |-- config.py
|   |-- scrape_playstore.py
|   |-- prepare_data.py
|   |-- embeddings.py
|   |-- features.py
|   |-- train.py
|   `-- predict.py
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- artifacts/
|
|-- notebooks/
|   `-- 01_end_to_end.ipynb
|
`-- tests/
    `-- test_prepare_data.py
```

---

## 11. Installation

Python 3.11+ is recommended.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Sentence Transformer model is downloaded the first time embeddings are generated, so internet access is required for that first run.

---

## 12. Run the entire project

```bash
python run_pipeline.py
```

This executes:

```text
scrape
  -> clean + success target
  -> embeddings
  -> semantic market features
  -> train/evaluate models
  -> save best model
```

Then launch:

```bash
streamlit run app.py
```

---

## 13. Run each stage manually

### Step 1 - collect live data

```bash
python -m src.scrape_playstore --country us --lang en
```

You can use India as the Play Store market:

```bash
python -m src.scrape_playstore --country in --lang en
```

### Step 2 - prepare data

```bash
python -m src.prepare_data
```

### Step 3 - generate semantic embeddings

```bash
python -m src.embeddings
```

### Step 4 - build semantic-market features

```bash
python -m src.features --top-k 30
```

### Step 5 - compare ML models

```bash
python -m src.train
```

### Step 6 - start UI

```bash
streamlit run app.py
```

---

## 14. Using an existing CSV instead of scraping again

If `data/raw/playstore_apps.csv` already exists:

```bash
python run_pipeline.py --skip-scrape
```

This is useful during experimentation because embeddings/training can be rerun without repeatedly requesting Google Play pages.

---

## 15. Recommended experimental methodology

For an academic submission, report:

1. Dataset size before and after cleaning
2. Missing-value handling
3. Success-label distribution
4. Description length distribution
5. Install and review distributions using log scale
6. Semantic embedding method
7. Similarity retrieval examples
8. Engineered competitor features
9. Train/test split
10. Model metrics
11. Confusion matrices
12. Selected model
13. Example startup predictions
14. Limitations and future work

---

## 16. Avoiding target leakage

Incorrect design:

```text
Features:
    installs
    reviews

Target:
    success based on installs + reviews
```

That lets the model see the answer.

This project instead uses:

```text
Candidate features:
    description embedding
    category
    price/free
    IAP
    ads
    semantic competitor statistics

Target:
    historical app success derived from
    that app's installs + reviews
```

For market features, only **other semantic neighbors** are used.

---

## 17. Stronger future version

After the baseline works, possible extensions include:

- larger and more balanced keyword coverage
- multilingual embeddings
- country-specific models
- time-aware snapshots
- app-age-normalized success labels
- calibrated probabilities
- gradient boosting / XGBoost
- SHAP explanations
- FAISS or vector database retrieval for very large datasets
- review sentiment as a competitor-market feature
- developer reputation features
- separate models by game/app category

A time-aware dataset with repeated Play Store snapshots would be the biggest improvement because it could support true future-performance forecasting.

---

## 18. Ethical and practical note

Use moderate request rates and respect the terms and policies applicable to the services and data sources you use. Public app metadata can change, disappear or be localized by country.

---

## 19. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Google Play Scraper
- Streamlit
- Joblib
- Matplotlib

---

## 20. Project outcome

The final system demonstrates:

- live data collection
- NLP
- semantic embeddings
- vector similarity
- feature engineering
- classification
- model comparison
- evaluation
- model persistence
- interactive deployment

This makes the project substantially richer than a standard tabular classification assignment.
