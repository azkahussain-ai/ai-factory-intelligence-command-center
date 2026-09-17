---
title: AI Factory Intelligence Command Center
emoji: 🏭
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.63.0"
app_file: app.py
pinned: false
---

# AI Factory 2.0 — AI Factory Intelligence Command Center

Decision-support platform for a synthetic manufacturing fleet, integrating
production/tabular data, sensor time-series, and maintenance text into a
traceable pipeline. Later stages add computer vision, NLP, RAG, a
multi-agent decision layer, explainability, a digital twin, human-in-the-
loop approval, and MLflow tracking (see `docs/HANDOFF.md` for status).

**This repository currently implements Stage I through Stage IV: Data
Engineering & EDA, ML & Deep Learning, Computer Vision & NLP, and GenAI +
RAG.** Multi-agent orchestration, explainability, digital twin, HITL/MLflow,
and the Streamlit app are separate, later stages (see `docs/HANDOFF.md`).

## Stage I scope

- Synthetic data generation for a 6-machine fleet (sensors, production, maintenance)
- Schema validation, cleaning, and outlier analysis
- Data integration (sensor → shift-level → merged with production + maintenance)
- Leakage-safe feature engineering (lag + rolling features, past-only)
- Time-based train/validation/test split
- EDA (summary statistics + a small set of purposeful plots)

No ML, DL, CV, NLP, RAG, agents, XAI, digital twin, HITL, MLflow, or
Streamlit code exists yet. Those are separate, later stages.

## Data sources

All data in this repository is **synthetic** — see
[`data/synthetic/SOURCE.md`](data/synthetic/SOURCE.md) for the full
generation methodology, assumptions, and limitations. In short:

- Sensor generation follows the *documented methodology* of the AI4I 2020
  Predictive Maintenance Dataset (UCI, Matzka 2020) — itself a synthetic
  dataset — rather than claiming access to real collected sensor data.
- Vibration/pressure channels and the production/maintenance data are
  original synthetic additions with documented, plausible assumptions.
- Every random process uses `random_state = 42` (see `config/config.py`) for full reproducibility.

## Data modalities

| Modality | File | Grain |
|---|---|---|
| Sensor time-series | `data/processed/cleaned_sensor_data.csv` | 15-minute readings per machine |
| Production / tabular | `data/processed/cleaned_production.csv` | Per machine, per 8-hour shift |
| Maintenance / incident text | `data/processed/cleaned_maintenance.csv` | Per maintenance event |
| Integrated | `data/processed/integrated_dataset.csv` | Per machine, per shift (sensor + production + maintenance-derived features) |

## Cleaning strategy

Implemented in `src/preprocessing/clean_data.py`. Highlights:

- **Timestamps:** malformed values (`"unknown"`) are dropped — they cannot be safely imputed for a time-ordering key.
- **Duplicates:** exact duplicate rows removed.
- **Invalid values:** physically impossible readings (negative Kelvin, negative vibration/pressure, absurd rpm) are treated as sensor glitches and nulled, never silently kept.
- **Missing sensor data:** short gaps while a machine is `RUNNING` are forward/backward filled (dropout); missingness while a machine is `DOWN` is preserved as-is, because it reflects a real state (sensors offline), not a data error.
- **Outliers:** IQR-flagged and counted, but not removed — a vibration spike can be a genuine early-failure signal, which is exactly what later models need to learn from.
- **Categorical inconsistency:** machine type / operating status labels normalized (case/whitespace).

Every cleaning decision's row counts are recorded in `reports/stage1_eda_summary.json`, nothing is hard-coded.

## Feature engineering & leakage prevention

Implemented in `src/features/build_features.py`. All predictor features
(lag values, rolling means over 3/9 shifts) are computed with
`.shift(1)` applied **before** any rolling window, so a feature at shift *t*
is built strictly from shifts before *t* — never including *t* itself, and
never using future shifts. This is verified by
`tests/test_stage1.py::test_lag_features_do_not_leak_future_information`.

The prediction target, `failure_next_shift`, intentionally looks one shift
into the future — that is what a supervised label is — and is kept
separate from the predictor columns for that reason. Rows with no future
shift to look at (each machine's final shift) are dropped since they have
no valid label.

## Train / validation / test strategy

Time-based split by simulation day (`src/features/split_data.py`), never
randomly shuffled:

| Split | Days | Rows |
|---|---|---|
| Train | 1–50 | 900 |
| Validation | 51–65 | 270 |
| Test | 66–90 | 444 |

A machine-agnostic day boundary is used (not per-machine), so no split
contains information from a later calendar day than the next. A test
(`test_time_based_split_has_no_day_overlap`) checks no `sim_day` value
appears in more than one split. Scaling statistics (`train_scaler_stats.json`)
are computed from the training split only.

## How to run

```bash
pip install -r requirements.txt
python -m src.run_stage1
```

This regenerates everything in `data/raw/`, `data/processed/`, and `reports/`
from scratch (seeded, so results are identical run to run).

Run tests:

```bash
pytest tests/test_stage1.py -v
```

## Generated outputs

```
data/raw/                    Raw synthetic data with injected quality issues
data/processed/
  cleaned_sensor_data.csv
  cleaned_production.csv
  cleaned_maintenance.csv
  integrated_dataset.csv
  train.csv / validation.csv / test.csv
  train_scaler_stats.json    Train-only mean/std, for Stage II to reuse
data/synthetic/SOURCE.md     Full data provenance and assumptions
reports/
  stage1_eda_summary.json    All EDA numbers, cleaning counts, split stats
  figures/                   4 purposeful EDA plots
```

## Known limitations

- Sensor channels beyond temperature/torque/rotational speed are assumption-based, not empirically grounded (see `SOURCE.md`).
- Overall failure rate is low (~1% of shifts), which is realistic for predictive maintenance but means Stage II will need to treat this as an imbalanced classification problem.
- Maintenance note text is templated, capping realism for later NLP work.
- This is a hackathon-scale synthetic dataset; results should not be assumed to transfer to a real factory.

## Project structure

```
ai_factory_2/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── config.py            Fleet definition, paths, seeds, split boundaries
├── data/
│   ├── raw/                 Generated raw data (messy, by design)
│   ├── processed/           Cleaned/integrated/split outputs
│   ├── synthetic/SOURCE.md  Data provenance documentation
│   ├── synthetic/images/    Stage III synthetic equipment images
│   └── sample/
├── src/
│   ├── data/                 generation, loading, schema validation
│   ├── preprocessing/        cleaning, integration
│   ├── features/              feature engineering, time-based split
│   ├── utils/                 EDA
│   ├── ml/                    Stage II baseline models + metrics
│   ├── deep_learning/         Stage II GRU (NumPy, from scratch)
│   ├── computer_vision/       Stage III CNN (NumPy, from scratch) + synthetic images
│   ├── nlp/                   Stage III TF-IDF classifier + structured output
│   └── run_stage1.py          pipeline entrypoint
├── tests/test_stage1.py, test_stage2.py, test_stage3.py
├── reports/                   EDA summary + figures + Stage II/III metrics
├── models/                    Stage II/III saved models + feature contracts
└── docs/HANDOFF.md
```

## Stage II - Machine Learning & Deep Learning

Predicts `failure_next_shift` (binary, ~1% positive rate) from Stage I's
processed splits.

**Baseline ML** (`src/ml/train_baseline.py`): Logistic Regression, Random
Forest, Gradient Boosting on Stage I's engineered lag/rollmean features.
Categorical columns one-hot encoded (column set frozen from train); missing
values get a `_missing` indicator flag plus train-median imputation
(missingness is informative - e.g. sensors during `DOWN` state). Decision
threshold is tuned on validation only (never test) via F1, since the
positive class is too rare for a 0.5 cutoff to fire reliably. Metrics:
accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix -
accuracy is reported but never used to pick a model.

**Deep learning** (`src/deep_learning/`): a GRU implemented from scratch in
NumPy (`gru_numpy.py` - forward pass, full backprop-through-time, Adam),
because this environment has no network access and could not install
TensorFlow/PyTorch. Trained on sliding 5-shift windows built per machine
from the *raw* per-shift sensor values (not Stage I's pre-computed lag
features), so it learns temporal dynamics itself rather than re-consuming
the baseline's engineered columns. Windows never span the Stage I
train/validation/test day-range boundaries.

Run:
```bash
python -m src.ml.train_baseline
python -m src.deep_learning.train_gru
python -m src.ml.compare_models
pytest tests/test_stage2.py -v
```

Outputs: `models/baseline_*.pkl`, `models/gru_model.npz`, feature contracts
(`models/*_feature_contract.json`), and metrics in
`reports/stage2_baseline_metrics.json`, `reports/stage2_gru_metrics.json`,
`reports/stage2_model_comparison.json`.

**Limitation to read results with:** train/val/test contain only 7/3/6
positive examples respectively. Several models (notably the GRU) reach
near-perfect validation/test scores - this reflects a small number of
positives separated by a strong synthetic precursor signal (near-total
`downtime_ratio` in the shift immediately before failure), not validated
generalization. Precision/recall/F1 on such small positive counts are high
variance; treat as a hackathon-scale proof of concept, not a production
metric.

## Stage III - Computer Vision & NLP

**No real equipment images existed anywhere in the project** (confirmed by
inspection before implementation). A minimal 100-image synthetic dataset
(clean vs. defect components, 32x32 grayscale, seed 42) was generated for
this reason and is documented in `data/synthetic/SOURCE.md` -
`src/computer_vision/generate_synthetic_images.py`.

**Computer Vision** (`src/computer_vision/`): a from-scratch NumPy CNN
(`cnn_numpy.py` - conv, ReLU, max-pool, dense, sigmoid; backprop verified
against numerical gradients) trained on the synthetic images
(`train_cv.py`), for the same no-network reason as Stage II's GRU.

**NLP** (`src/nlp/`): TF-IDF + Logistic Regression (`train_nlp.py`)
classifying maintenance-note `incident_type`, reusing Stage I's existing
`cleaned_maintenance.csv` and Stage I's train/validation/test day-range
config (no new split logic invented).

Run:
```bash
python -m src.computer_vision.generate_synthetic_images
python -m src.computer_vision.train_cv
python -m src.nlp.train_nlp
python -m src.nlp.build_stage3_structured_output
```

Outputs: `models/cv_cnn_model.npz`, `models/nlp_tfidf_vectorizer.pkl`,
`models/nlp_incident_classifier.pkl`; `reports/stage3_cv_metrics.json`,
`reports/stage3_nlp_metrics.json`, `reports/stage3_cv_predictions.json`,
`reports/stage3_nlp_predictions.json`, `reports/stage3_structured_output.json`
(combined per-machine CV+NLP view for later stages).

**Limitations:** the maintenance-note text is templated (Stage I's own
documented limitation), so NLP scores near-perfectly and should be read as
a proof of concept, not a generalization result; the CV dataset is
synthetic and small (100 images), so its metrics reflect separability of
the generation rule, not real defect-detection performance.

## Stage IV - GenAI + RAG

**No real machine manuals, SOPs, or safety documents existed anywhere in
the project** (confirmed by inspection before implementation). Three
synthetic PDFs were generated for this reason - `knowledge_base/*.pdf` -
with operating thresholds computed directly from
`config.MACHINE_TYPE_PROFILES` (not hand-typed), documented in
`knowledge_base/SOURCE.md`.

**Pipeline** (`src/rag/`): PDF text extraction (`document_loader.py`,
`pdfplumber`) -> word-count-bounded chunking with document/page/section
metadata (`chunking.py`) -> TF-IDF embeddings (`embeddings.py`) -> a
from-scratch NumPy cosine-similarity vector store (`vector_store.py`) ->
semantic retrieval (`retriever.py`) -> grounded answer generation
(`generator.py`).

**Why TF-IDF instead of a sentence-transformer, and a NumPy store instead
of FAISS/Chroma:** this sandbox has no general network egress (PyPI,
Hugging Face, and LLM APIs are not on the allowlist - verified before
implementation), the same constraint Stage II/III hit with
TensorFlow/PyTorch. TF-IDF + brute-force cosine similarity is a legitimate,
fully local, lightweight retrieval baseline for a knowledge base this size,
not a placeholder.

**Generation layer:** `generate_grounded_answer()` checks for
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` and would call a live LLM if one
were reachable; with neither a key nor network access available here, it
falls back to a deterministic, evidence-only synthesis that only ever
states what the retrieved chunks actually say. `generate_unsupported_answer()`
is a separate, explicitly-labeled generic answer with **no** knowledge-base
access, used only for the RAG-vs-no-RAG demonstration.

**Stage II/III integration (read-only):** `context_builder.py` reads
`reports/stage3_structured_output.json` to attach a machine's current
CV/NLP findings to a machine-specific question. It does not modify any
Stage II or Stage III file.

Run:
```bash
python -m src.rag.build_knowledge_base_pdfs   # generate the 3 knowledge-base PDFs
python -m src.rag.build_knowledge_base        # ingest -> chunk -> embed -> persist index
python -m src.rag.demo_rag_vs_no_rag          # required RAG-vs-no-RAG demonstration
python -m src.run_stage4                      # full Stage IV pipeline + sample query
```

Outputs: `knowledge_base/*.pdf`, `knowledge_base/index/` (persisted vector
store + TF-IDF vectorizer); `reports/stage4_ingestion_summary.json`,
`reports/stage4_structured_output.json`, `reports/stage4_rag_vs_no_rag_demo.json`.

**Limitations:** the knowledge base is 3 short synthetic documents (the
minimum needed to demonstrate multi-document RAG with real, distinct
sources), not a real factory's full documentation set; retrieval quality
depends on TF-IDF term overlap rather than semantic embeddings, so
paraphrased queries with little vocabulary overlap with the source text may
retrieve weaker matches than a neural embedding model would.

---

## Stage V — Agentic AI / Multi-Agent System

Four cooperating agents coordinate the existing Stage II-IV components -
no earlier stage was retrained, rebuilt, or duplicated (verified by a
byte-for-byte checksum comparison of every Stage I-IV file before/after
Stage V; see `docs/HANDOFF.md`).

**Vision Agent** (`src/agents/vision_agent.py::get_vision_result`) reads
Stage III's existing `reports/stage3_structured_output.json` CV result for
a machine. `defect_type`/`severity` are reported as `null` (honestly
unavailable) rather than invented, since Stage III's CNN is a binary
defect/normal classifier only.

**Predictive Maintenance Agent**
(`src/agents/predictive_maintenance_agent.py::get_prediction`) loads Stage
II's saved, already-trained GRU model (`models/gru_model.npz` - the model
Stage II itself selected as best by validation F1) and scores a machine's
most recent 5-shift window. This is inference only: no retraining, same
feature contract/scaling stats Stage II produced. No per-machine live
prediction existed anywhere in Stage II's output (only aggregate metrics),
so this scoring function is new Stage V integration work, not a rebuild.

**Knowledge/RAG Agent** (`src/agents/knowledge_agent.py`) is a thin wrapper
around Stage IV's existing `RAGPipeline` - same vector store, retriever,
generator; no second index. `formulate_query()` builds the question from
the *actual* upstream findings (e.g. the predictive maintenance agent's
top signal, or a detected defect), never a fixed string.

**Planning/Decision Agent** (`src/agents/planning_agent.py::decide`)
combines all four results with explicit, documented rules over the real
field values (risk level, defect flag, incident type, RAG groundedness) to
set `priority` (`ROUTINE`/`ELEVATED`/`URGENT`) and `recommendation` - never
a hard-coded string, verified by a test that two different risk inputs
produce two different recommendations. An LLM is optionally used only to
rephrase the reasoning as prose (`ANTHROPIC_API_KEY` env var, same
optional-live-LLM/local-fallback pattern as Stage IV's generator) - the
recommendation/priority values themselves are decided before any LLM call,
keeping the LLM out of the predictive path per the Stage V requirement.

**Orchestration:** `src/agents/orchestrator.py::run_workflow(machine_id)`
runs the four agents in sequence into one shared state dict, matching the
brief's schema (`factory_context`, `vision_result`,
`predictive_maintenance_result`, `nlp_result`, `rag_result`, `decision`,
`agent_trace`). No LangGraph: the project had no orchestration framework
installed and a 4-node fan-in workflow doesn't justify a new dependency
(consistent with Stages II-IV's own "avoid unnecessary dependencies"
choices under this project's sandbox constraints). Every agent call is
wrapped so a missing/failed upstream result is marked `"status":
"unavailable"` or `"error"` with a reason - never silently replaced with a
fake value - and the Planning Agent still produces a (conservative)
decision even when some inputs are missing.

Run:
```bash
python -m src.agents.orchestrator --machine-id M-01
```

Output: `reports/stage5_multi_agent_decision.json` (also printed to stdout).

**Limitations:** the GRU currently reports `LOW` risk for all 6 machines in
the demonstration run, because the test split has only 6/444 positive
shifts and the model's validation-tuned decision threshold is high -
documented, not adjusted to force a more dramatic-looking demo. `M-05`'s
run has no detected defect (the others do), showing the recommendation
genuinely varies by machine rather than following one fixed path.

---

## Stage VI — Explainable AI (XAI)

Explains *why* the existing Stage II/III models produced their outputs.
No model was retrained or replaced; every explanation is computed from a
real forward pass through the actual trained artifact.

**Predictive maintenance (`src/xai/gru_explainer.py`)** explains the GRU -
the model Stage V's live Predictive Maintenance Agent actually scores with
- using **permutation importance**: each feature in the machine's 5-shift
window is neutralized (set to its standardized training mean) one at a
time, and the resulting probability shift is the feature's contribution.
SHAP's `DeepExplainer` needs a TensorFlow/PyTorch graph this hand-rolled
NumPy GRU doesn't have; `KernelExplainer` is model-agnostic but too slow
for per-agent-call use - permutation importance is the documented,
technically-appropriate alternative the Stage VI brief itself allows.
Verified against a **real historical failure** (M-01, day 23, Afternoon
shift - an actual positive-labeled row in `data/processed/train.csv`): the
GRU correctly predicts 99.99% failure probability with meaningful (not
near-zero) attributions led by `operating_status_DOWN` and
`downtime_minutes`.

**Baseline tabular models (`src/xai/shap_tabular_explainer.py`)** use real
**SHAP** - `TreeExplainer` for Random Forest/Gradient Boosting,
`LinearExplainer` (with a proper 50-row training-sample background, not a
degenerate single-point background) for Logistic Regression. These are
Stage II's other saved artifacts and are fully SHAP-compatible, satisfying
the brief's preferred method directly.

**Computer vision (`src/xai/gradcam_explainer.py`)** implements **Grad-CAM**
by hand for Stage III's custom NumPy CNN (again, no autodiff framework
exists for it): the gradient of the predicted logit is manually
back-propagated to the network's one convolutional layer, mirroring the
exact chain-rule steps the model's own `backward()` method already uses.
Verified two ways: the forward pass reproduces Stage III's originally
recorded confidence exactly (0.9402 for M-01), and the resulting heatmap
visibly highlights the image's real defect region (an edge/scratch), not
noise - see `reports/xai/`.

**Structured output (`src/xai/explanation_formatter.py`,
`src/xai/xai_pipeline.py`)** assembles both into one schema per machine:
`prediction`, `feature_explanations`, `top_features`, `vision_explanation`,
`human_summary` (dynamically generated prose, never hard-coded per
machine), and `method`. Any unavailable/failed component is marked as such
- never silently replaced with a fabricated value.

**Multi-agent integration:** Stage V's `orchestrator.py` gained one new,
additive step (`xai_layer`) that runs after the Predictive Maintenance
Agent and before the Planning Agent; `planning_agent.decide()` gained an
optional `xai=None` parameter (fully backward compatible - Stage V's
original 4-argument calls still work unchanged, reverified by test) that,
when provided, adds one more reasoning line citing the actual top
attributed features. Priority/recommendation logic itself is untouched by
this - XAI only adds an extra evidentiary reasoning line.

Run:
```bash
python -m src.xai.xai_pipeline --machine-id M-01                       # current window
python -m src.xai.xai_pipeline --machine-id M-01 --sim-day 23 --shift Afternoon  # explain a real past failure
python -m src.agents.orchestrator --machine-id M-01                    # full workflow with XAI included
pytest tests/test_stage6.py -v
```

**Limitation:** like Stage V, the *current* (most recent) window for all 6
machines predicts LOW risk (near-zero probability), so live permutation
importances at that point are tiny by construction - a saturated sigmoid
has near-flat gradient. This is why the GRU explainer accepts an optional
historical `sim_day`/`shift` to also explain real past HIGH-risk
predictions for demonstration/validation, rather than only ever showing a
quiet state.

---

## Stage VII — Digital Twin / What-If Simulation

Quantifies three operational choices for a machine over a configurable
horizon (default 6 shifts / 2 days), so a supervisor can compare them
side-by-side rather than reason about failure risk in the abstract.

**`src/digital_twin/simulator.py::run_scenarios(machine_id)`** builds the
machine's real current state (production rate + a real forward pass
through Stage II's GRU for its current failure probability - the same
model Stage V's Predictive Maintenance Agent uses) and simulates:

1. **`continue_operating`** - current failure probability compounds over
   the horizon (`1 - (1-p)^n`, standard probability compounding, not an
   arbitrary number); expected downtime = cumulative risk × the fleet's
   *empirical* mean unplanned-failure downtime (8.0h, computed from real
   `DOWN`-status shifts in `data/processed/cleaned_production.csv`).
2. **`stop_for_maintenance`** - incurs the fleet's empirical mean planned-
   maintenance downtime (4.5h, computed from real `MAINTENANCE`-status
   shifts), then the GRU is re-scored on a documented counterfactual input
   (fleet-wide healthy-state training averages + reset maintenance-history
   fields), never a guessed post-maintenance probability.
3. **`reduce_load`** - load/torque/rpm/production_rate scaled down by a
   documented 20% assumption, and the GRU is re-scored on that
   counterfactual input to see whether reduced stress actually lowers
   failure risk for *this* machine's real current state.

Each scenario reports expected production units, expected downtime,
cumulative failure risk, and a financial comparison
(`expected_revenue_usd − downtime_cost_usd − expected_failure_repair_cost_usd`)
using documented constants in `config/config.py` (unit margin, hourly
downtime cost, failure repair cost - explicit assumptions, since no
pricing data exists anywhere in this project; downtime *durations* are
empirical, not assumed). The scenario with the highest net value is
reported as `recommended_scenario`, with the ranking rule stated
explicitly (`recommendation_basis`) rather than left implicit.

**Validated against a real historical failure**, not just the (currently
uneventful) live state: simulating from M-01's actual day-23 failure
window shows `stop_for_maintenance` clearly winning on both risk and net
value over `continue_operating` - proving the three scenarios genuinely
diverge when risk is real, not just producing three near-identical numbers.

**Multi-agent integration:** `orchestrator.py` gained one more additive
step (`digital_twin`, state key `digital_twin_result`) between the XAI
layer and the Knowledge/RAG agent; `planning_agent.decide()` gained a
second optional parameter (`twin=None`, alongside Stage VI's `xai=None`) -
still fully backward compatible with Stage V's original 4-argument calls -
that adds one more reasoning line citing the actual simulated
recommendation when provided.

Run:
```bash
python -m src.digital_twin.digital_twin_pipeline --machine-id M-01                              # current state
python -m src.digital_twin.digital_twin_pipeline --machine-id M-01 --sim-day 23 --shift Afternoon # a real past failure
python -m src.agents.orchestrator --machine-id M-01                                               # full workflow incl. digital twin
pytest tests/test_stage7.py -v
```

**Limitation:** like Stages V/VI, all 6 machines' *live* state currently
shows near-zero failure probability (Stage II's documented class
imbalance), so `continue_operating` wins by default for all of them today
- not a simulation bug, the correct output given genuinely low current
risk. The `--sim-day`/`--shift` override exists specifically so the
scenario differentiation can be demonstrated and tested against a real
known failure instead of only a quiet state.

---

## Stage VIII — Human-in-the-Loop + MLOps/MLflow

### Human-in-the-Loop (`src/hitl/`)

Wraps a real AI recommendation (the exact output of Stage V's
`orchestrator.run_workflow`, which already carries Stage VI's XAI and Stage
VII's digital-twin evidence) in a supervisor decision step with exactly
three choices:

- **APPROVE** - `final_action` = the original recommendation, unchanged.
- **REJECT** - requires a `comment`; `final_action` is `None` (no action taken).
- **MODIFY** - requires both a `modified_action` and a `comment`;
  `final_action` = the modified action.

`src/hitl/hitl_workflow.py::submit_decision()` never generates or guesses a
recommendation itself - it only accepts a real orchestrator output and a
real human decision, and raises `HumanDecisionError` for an invalid
decision value or a missing required reason/action. The **original AI
recommendation is stored as an immutable field** in the audit record
alongside the human decision - verified by a test that the stored original
is byte-identical regardless of which of the three decisions was made.

`src/hitl/audit_store.py` persists every decision as one JSON line in
`reports/hitl_audit_log.jsonl` (append-only - matches how every other
stage in this project stores data: plain, inspectable files, no database
beyond MLflow's own). Each record has a unique `decision_id`, an ISO
`timestamp`, the untouched `original_ai_recommendation`, the full
supporting `ai_evidence` (vision/predictive-maintenance/XAI/digital-twin/
RAG results, so the record is self-contained), and the human's
`human_decision`/`human_comment`/`modified_action`/`supervisor`/`final_action`.

Run:
```bash
python -m src.hitl.hitl_pipeline --machine-id M-01 --decision APPROVE
python -m src.hitl.hitl_pipeline --machine-id M-01 --decision REJECT --comment "reason"
python -m src.hitl.hitl_pipeline --machine-id M-01 --decision MODIFY --modified-action "..." --comment "reason"
```

### MLOps / MLflow (`src/mlops/mlflow_logging.py`)

Logs Stage II's **already-trained, already-evaluated** models into MLflow
- nothing is retrained. Every parameter/metric logged is read from the real
`reports/stage2_baseline_metrics.json` / `stage2_gru_metrics.json` (or
copied verbatim from the known training hyperparameters in
`src/ml/train_baseline.py`/`src/deep_learning/train_gru.py`), never invented.

- **4 runs** (exceeds the required 3): `logistic_regression`,
  `random_forest`, `gradient_boosting` (all real `mlflow.sklearn.log_model`
  calls on the actual saved `.pkl` files), and `gru` (the hand-rolled NumPy
  model, wrapped as an `mlflow.pyfunc.PythonModel` - see `GRUPyfuncWrapper`
  - so it can be logged as a proper model, not just a bare file artifact).
- **Model Registry**: `register_best_model()` registers Stage II's own
  already-selected best model (`reports/stage2_model_comparison.json::
  best_model_by_validation_f1` - not re-decided here) under
  `ai_factory_failure_predictor`. Verified that predictions made through
  the registered model are numerically identical to a direct forward pass
  through the original trained model.
- Tracking store: local SQLite (`sqlite:///mlflow.db`) rather than the
  plain file store, because the Model Registry requires a database-backed
  URI.

**A real bug was found and fixed while building this**: the original
`log_gru_run()` logged the GRU's `.npz` weights with a plain
`mlflow.log_artifact(..., artifact_path="model")` call, which produces a
bare file with no MLflow model manifest - `mlflow.register_model()` then
fails because there is no actual "logged model" at that path for it to
point at. Fixed by wrapping the GRU in `GRUPyfuncWrapper` and using
`mlflow.pyfunc.log_model()` instead, which was then verified end-to-end
(registration succeeds, and the registered model's predictions match the
original exactly).

Run:
```bash
python -c "from src.mlops.mlflow_logging import log_all_stage2_runs, register_best_model; \
    run_ids = log_all_stage2_runs(); print(register_best_model(run_ids))"
mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs in the browser
pytest tests/test_stage8.py -v
```

**Limitation:** MLflow's own model-serialization warns that a
`cloudpickle`-serialized `python_model` "requires exercising caution" for
untrusted sources - acceptable here since it's this project's own code,
but worth knowing before deploying the registry to a shared environment.

---

## Stage IX — Web Application + Automated Report

`app.py` (repo root, required by Hugging Face Spaces) is a Streamlit app
that puts the full Stage I-VIII pipeline behind one interface — it contains
no modeling/prediction/business logic of its own, only calls to the real
functions already built in earlier stages:

```
Machine select / image upload
  → Stage V orchestrator.run_workflow() [Stage II prediction, Stage III
    vision/NLP, Stage VI XAI, Stage VII digital twin, Stage IV RAG, Stage V
    multi-agent recommendation - one call, all real]
  → 7 tabs: Prediction · XAI · RAG Evidence · Recommendation · Digital Twin
    · Human Decision · Report
  → Stage VIII submit_decision() [APPROVE/REJECT/MODIFY]
  → Stage IX report_generator.generate_report() → downloadable PDF
```

**Data/file upload:** besides selecting a machine, the sidebar accepts an
inspection image upload, which runs **live inference through the actual
trained Stage III CNN** (`src/xai/gradcam_explainer.py::explain_uploaded_image`,
verified to produce the same confidence as Stage III's stored result when
given the same image) plus a live Grad-CAM - not a stub.

**Report** (`src/reporting/report_generator.py`, reportlab, already a
project dependency): assembles prediction, XAI, RAG evidence, the
multi-agent recommendation, digital-twin scenarios, and (if submitted) the
human decision into one downloadable PDF - every field is read from the
real dicts already produced upstream, nothing computed or invented in the
report itself.

Tested with Streamlit's `AppTest` harness (`tests/test_stage9.py`): app
load, running the real pipeline, switching machines, all three human
decisions (including validation errors surfacing in the UI, not crashing),
PDF generation, and the live image-upload path against a real known image.

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Hugging Face Spaces deployment

The YAML frontmatter at the top of this README configures the Space
(`sdk: streamlit`, `app_file: app.py`). **Important:** unlike earlier
stages, `.gitignore` here deliberately keeps `models/`, `data/processed/`,
`reports/*.json`, and `knowledge_base/` **committed** - the deployed app
reads these directly at runtime (trained models, engineered features,
Stage II-IV reports, the RAG index) and there is no training step at
deploy time. Only `data/raw/` (regenerable, unused by the app) and
session-specific runtime files (`mlflow.db`, `mlruns/`, the HITL audit
log) are excluded. To deploy: push this repository to a new Streamlit-SDK
Space; no additional configuration is required.
