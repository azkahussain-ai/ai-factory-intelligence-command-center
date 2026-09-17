# Project Handoff

## Current Stage
Stage I — Data Engineering & EDA

## Status
COMPLETE

## Implemented
- Synthetic data generator for a 6-machine fleet (sensors, production, maintenance), seeded (`random_state=42`), grounded in the documented AI4I 2020 generation methodology
- Deliberately injected, documented data-quality issues in raw data (missing values, duplicates, invalid readings, inconsistent labels, malformed timestamps)
- Schema validation, cleaning (with per-decision counts logged, nothing hard-coded)
- Sensor→shift aggregation and integration with production + maintenance (past-only maintenance features)
- Leakage-safe lag/rolling feature engineering + shift-level failure target
- Time-based train/validation/test split (day 1–50 / 51–65 / 66–90), train-only scaler stats
- EDA summary (`reports/stage1_eda_summary.json`) + 4 plots

## Files Created/Updated
- `config/config.py`
- `src/data/generate_synthetic.py`, `load_data.py`, `validate_schema.py`
- `src/preprocessing/clean_data.py`, `integrate_data.py`
- `src/features/build_features.py`, `split_data.py`
- `src/utils/eda.py`
- `src/run_stage1.py` (pipeline entrypoint)
- `tests/test_stage1.py`
- `data/synthetic/SOURCE.md`, `README.md`, `requirements.txt`, `.env.example`, `.gitignore`

## Data / Outputs
- `data/raw/*.csv` — raw synthetic data (messy by design)
- `data/processed/cleaned_*.csv`, `integrated_dataset.csv`, `train.csv`, `validation.csv`, `test.csv`, `train_scaler_stats.json`
- `reports/stage1_eda_summary.json`, `reports/figures/*.png`
- Sizes (last run): sensor 51,834 rows / production 1,620 / maintenance 49 / integrated 1,620 / feature-with-target 1,614 / train 900 / val 270 / test 444

## Important Decisions
- Real public sensor data could not be downloaded in this environment (network allowlist has no UCI/Kaggle access); AI4I's *documented generation formulas* were replicated locally instead of fabricating an ungrounded distribution — AI4I itself is stated by UCI to be synthetic, not collected data.
- Missingness while a machine is `DOWN` is preserved (meaningful), not imputed; missingness while `RUNNING` is treated as dropout and short-gap filled.
- Outliers are counted (IQR) but never auto-removed — a vibration spike can be a genuine early-failure signal.
- Prediction target is `failure_next_shift` (next-shift binary failure), intentionally forward-looking as a label while all predictor features are strictly past-only.
- Failure-episode probability was tuned (`generate_synthetic.py::_plan_failure_episodes`) so every split (train/val/test) contains at least some positive examples — this changes data realism/coverage, not any model score.

## Tests
`pytest tests/test_stage1.py -v` — 13/13 passed. Covers: reproducibility, injected raw quality issues, schema validation failure, cleaning correctness (duplicates/timestamps/physical bounds/DOWN-state missingness preserved), integration row-count integrity, no-leakage lag/rolling features, correct forward-shifted target, and split day-range/no-overlap integrity.

## Known Issues
- Overall failure rate is low (~1% of shifts) — realistic for predictive maintenance, but Stage II must treat this as class-imbalanced.
- Maintenance note text is templated (not organic free text), which will limit later NLP realism.

## Stage II — Machine Learning & Deep Learning

## Status
COMPLETE

## Implemented
- Baseline ML (`src/ml/train_baseline.py`): Logistic Regression, Random
  Forest, Gradient Boosting on Stage I's `train/validation/test.csv`.
  One-hot categoricals (frozen train column contract), `_missing` indicator
  flags + train-median imputation, validation-tuned F1 threshold, sample-
  weighted Gradient Boosting (no native `class_weight` support).
- Deep learning (`src/deep_learning/gru_numpy.py`, `train_gru.py`): GRU
  implemented from scratch in NumPy (forward + BPTT + Adam) - TensorFlow/
  PyTorch could not be installed in this sandbox (no network access, see
  Known Issues). Trained on 5-shift sliding windows of *raw* per-shift
  sensor values per machine, built within-split only (never spans the
  Stage I day-range boundaries).
- Shared metric utility (`src/ml/metrics_utils.py`): accuracy, precision,
  recall, F1, ROC-AUC, PR-AUC, confusion matrix; guards ROC-AUC/PR-AUC when
  a split has a single class present.
- Comparison report (`src/ml/compare_models.py`) selecting the best model
  by validation F1 (not accuracy, given ~1% positive rate).
- `tests/test_stage2.py` - 10/10 passed (run manually; `pytest` itself is
  not installed in this sandbox, see Known Issues). Covers: sequence-window
  construction (no cross-machine spans, correct target alignment), GRU
  forward reproducibility, valid probability outputs, one-step loss
  decrease, threshold selection on validation only (not refit per split),
  single-class metric guarding.

## Files Created
- `src/ml/train_baseline.py`, `src/ml/metrics_utils.py`, `src/ml/compare_models.py`
- `src/deep_learning/gru_numpy.py`, `src/deep_learning/train_gru.py`
- `tests/test_stage2.py`
- `models/baseline_{logistic_regression,random_forest,gradient_boosting}.pkl`,
  `models/baseline_feature_contract.json`
- `models/gru_model.npz`, `models/gru_feature_contract.json`
- `reports/stage2_baseline_metrics.json`, `reports/stage2_gru_metrics.json`,
  `reports/stage2_model_comparison.json`

## Results (validation / test F1, selected model = highest validation F1)
- Logistic Regression: val F1 0.800, test F1 0.857
- Random Forest: val F1 0.800, test F1 0.923
- Gradient Boosting: val F1 0.022, test F1 0.027 (sample-weighting alone
  was insufficient here; not selected)
- GRU (NumPy, from scratch): val F1 1.000, test F1 1.000
- Selected: GRU, by validation F1 - see the "Limitation" note below before
  treating this as a strong result.

## Important Decisions
- Threshold selection: best-F1 threshold is fit on validation predictions
  only, then reused unchanged on test - never refit per split (tested
  explicitly).
- GRU consumes raw per-shift sensor values (not Stage I's pre-computed
  lag/rollmean features) so the recurrence has to learn temporal structure
  itself, rather than trivially re-deriving the tabular baseline's inputs.
- Missingness (`DOWN`-state sensor gaps, no-maintenance-history rows) is
  encoded as an explicit `_missing` indicator + train-median fill for the
  baselines, and as train-median fill for the GRU sequence features.

## Known Issues
- Positive class is extremely rare (train=7, val=3, test=6 examples) -
  reported metrics, especially the GRU's perfect scores, are high-variance
  and should be read as a hackathon proof of concept, not a validated
  production result (documented in README + `stage2_model_comparison.json`).
- This sandbox has no network access: `xgboost`, `tensorflow`, `torch`,
  `mlflow`, `shap`, and `pytest` could not be installed. XGBoost was
  substituted with sklearn's `GradientBoostingClassifier`; the GRU is a
  from-scratch NumPy implementation instead of a TensorFlow/PyTorch one;
  MLflow/SHAP are deferred to whichever later stage introduces them, in an
  environment with network access. `tests/test_stage2.py` was verified by
  direct function execution (10/10 passed), not via the `pytest` CLI.

## Next Stage
Stage III - see below (complete).

---

## Stage III — Computer Vision & NLP

## Status
COMPLETE

## Implemented
- **Computer Vision**: no real equipment images existed anywhere in the
  project (confirmed by inspection first). Generated a minimal 100-image
  synthetic dataset (`src/computer_vision/generate_synthetic_images.py`,
  documented in `data/synthetic/SOURCE.md`), then trained a from-scratch
  NumPy CNN (`src/computer_vision/cnn_numpy.py`: conv -> ReLU -> max-pool ->
  dense -> ReLU -> dense -> sigmoid; backward pass verified against
  numerical gradients, max abs diff ~1e-11) via `train_cv.py`. Same
  no-network rationale as Stage II's GRU (TensorFlow/PyTorch unavailable).
- **NLP**: TF-IDF + Logistic Regression (`src/nlp/train_nlp.py`)
  classifying maintenance-note `incident_type` from the existing
  `data/processed/cleaned_maintenance.csv` (Stage I output, not
  regenerated). Reuses Stage I's exact train/validation/test day-range
  config (`config.TRAIN_DAY_RANGE` etc.) via `add_sim_day`, applied to the
  maintenance timestamp's date - no new split logic invented.
- **Structured multimodal output**: `src/nlp/build_stage3_structured_output.py`
  joins the CV and NLP per-record predictions into one per-`machine_id`
  record (`reports/stage3_structured_output.json`) for later stages
  (multi-agent, digital twin) to consume without reading both source files.
- `tests/test_stage3.py` - 8/8 passed (run manually; `pytest` CLI itself is
  not installed in this sandbox, same as Stage II). Covers: CV/NLP metrics
  report validity, structured-output schema, CV/NLP model loadability for
  inference, CNN gradient-descent sanity, text-cleaning normalization, and
  that the maintenance split reuses Stage I's day boundaries with no
  overlap.
- Confirmed Stage I/II remain intact: Stage I processed CSVs unchanged
  (checksum comparison before/after), Stage II's 10/10 tests still pass.

## Files Created
- `src/computer_vision/generate_synthetic_images.py`, `cnn_numpy.py`, `train_cv.py`, `__init__.py`
- `src/nlp/train_nlp.py`, `build_stage3_structured_output.py`, `__init__.py`
- `tests/test_stage3.py`
- `data/synthetic/images/*.png` (100 images), `data/synthetic/images_metadata.csv`
- `models/cv_cnn_model.npz`, `models/nlp_tfidf_vectorizer.pkl`, `models/nlp_incident_classifier.pkl`
- `reports/stage3_cv_metrics.json`, `reports/stage3_cv_predictions.json`
- `reports/stage3_nlp_metrics.json`, `reports/stage3_nlp_predictions.json`
- `reports/stage3_structured_output.json`

## Results
- CV (test, n=20): accuracy 0.900, precision 1.000, recall 0.800, F1 0.889.
- NLP (test, n=16): accuracy 1.000, F1_macro 1.000 - expect this given
  Stage I's own documented limitation that maintenance note text is
  templated, not organic free text (near-perfect separability by keyword).

## Known Issues
- CV dataset is synthetic and small (100 images); metrics reflect
  separability of the generation rule, not real-world defect-detection
  performance.
- NLP dataset is tiny (49 total records; train=24/val=9/test=16) with some
  `incident_type` classes having very few examples per split (e.g. `Quality`
  has 1 in train, 0 in test) - per-class metrics for rare classes are
  high-variance; macro averaging is used but should still be read
  cautiously.
- Same sandbox constraints as Stage II: no network access, so
  TensorFlow/PyTorch could not be installed for the CV model either (used
  a gradient-checked from-scratch NumPy CNN instead); `pytest` CLI not
  installed (tests run via direct function execution).

## Next Stage
Stage IV - see below (complete).

---

## Stage IV — GenAI + RAG

## Status
COMPLETE

## Implemented
- **Inspection first**: confirmed Stage I/II/III complete and untouched,
  and that no knowledge-base/PDF documents existed anywhere in the project,
  before writing any Stage IV code (checksums of every Stage I-III file
  recorded before and re-verified after implementation — zero diffs).
- **Knowledge base** (`knowledge_base/`): 3 synthetic PDFs generated by
  `src/rag/build_knowledge_base_pdfs.py` — Machine Operation Manual,
  Preventive Maintenance SOP, Safety & Emergency Guidelines. Every numeric
  threshold is computed from `config.MACHINE_TYPE_PROFILES` (mean + 2σ /
  3.5σ), not hand-typed, so the knowledge base stays consistent with the
  Stage I sensor-generation parameters. Documented in
  `knowledge_base/SOURCE.md`.
- **RAG pipeline** (`src/rag/`): `document_loader.py` (pdfplumber
  extraction with per-page text + document/page metadata) →
  `chunking.py` (~90-word overlapping chunks, each tagged with the nearest
  preceding "Section N: ..." heading) → `embeddings.py` (TF-IDF, since
  sentence-transformer weights cannot be downloaded — no network egress to
  Hugging Face/PyPI/any LLM API confirmed before implementation) →
  `vector_store.py` (from-scratch NumPy cosine-similarity store, persisted
  as `.npy` + `chunks.json`, since faiss-cpu/chromadb are not installable
  here — same no-network pattern as Stage II's GRU / Stage III's CNN) →
  `retriever.py` (top-k search with a minimum-score cutoff so unrelated
  queries honestly return "no evidence" instead of a forced match) →
  `generator.py` (grounded, evidence-only answer synthesis; checks for a
  live LLM key first and falls back locally, matching the Stage IV brief's
  required local/fallback mode).
- **Stage II/III integration** (`context_builder.py`): reads (never writes)
  `reports/stage3_structured_output.json` to attach a machine's current
  CV/NLP findings to a machine-specific RAG question.
- **RAG vs. unsupported-LLM demonstration** (`src/rag/demo_rag_vs_no_rag.py`):
  same question answered with zero knowledge-base access
  (`generate_unsupported_answer`, generic, cites no document or number) vs.
  through the full RAG pipeline (cites specific documents, pages, sections,
  and threshold values). Saved to
  `reports/stage4_rag_vs_no_rag_demo.json`.
- `src/run_stage4.py`: end-to-end entrypoint (builds PDFs/index if missing,
  answers a sample machine-specific question, runs the demonstration).
- `tests/test_stage4.py` — 12/12 passed (run manually; `pytest` CLI is not
  installed in this sandbox, same constraint as Stage II/III). Covers:
  PDF extraction correctness, chunk metadata traceability, vector-store
  save/load round-trip determinism, relevant retrieval for a real factory
  question, honest "no evidence" behavior for an unrelated query, grounded
  vs. unsupported generation, the RAG-vs-no-RAG demo's structural
  guarantees, Stage IV's structured-output schema, and a checksum
  comparison proving every Stage I-III file is byte-identical to before
  Stage IV started.

## Files Created
- `knowledge_base/SOURCE.md`, `knowledge_base/*.pdf` (3 documents),
  `knowledge_base/index/` (persisted vector store + vectorizer, created on
  first run — not committed as static files, regenerable via
  `build_knowledge_base.py`)
- `src/rag/__init__.py`, `document_loader.py`, `chunking.py`,
  `embeddings.py`, `vector_store.py`, `retriever.py`, `generator.py`,
  `context_builder.py`, `rag_pipeline.py`, `build_knowledge_base_pdfs.py`,
  `build_knowledge_base.py`, `demo_rag_vs_no_rag.py`
- `src/run_stage4.py`
- `tests/test_stage4.py`
- `reports/stage4_ingestion_summary.json`, `reports/stage4_structured_output.json`,
  `reports/stage4_rag_vs_no_rag_demo.json`

## Files Modified
- `README.md` — added Stage IV section, corrected the stale "Stage I only"
  scope line (no Stage I-III content removed).
- `.env.example` — added optional `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`
  placeholders for the optional live-LLM path (unset here; local fallback
  is what actually runs in this sandbox).
- `requirements.txt` — added a Stage IV section listing already-installed
  packages this stage uses (`pdfplumber`, `reportlab`) — no new
  installation was needed or attempted.

## Results
- Knowledge base: 3 documents, 5 pages total, 21 chunks (see
  `reports/stage4_ingestion_summary.json` for the exact run).
- Retrieval: the demo question ("What should the operator do if machine
  vibration becomes dangerously high?") retrieves its top match from
  `preventive_maintenance_sop.pdf`'s DANGER-response section, with
  supporting matches from the WARNING-response section and both the CNC
  and Press sections of the operation manual — all genuinely relevant, not
  hard-coded.
- RAG vs. no-RAG: without RAG, the answer is generic and cites no document
  or number; with RAG, the answer cites `preventive_maintenance_sop.pdf`
  and `machine_operation_manual.pdf` by name, page, and section, including
  the exact mm/s thresholds.

## Known Issues / Honest Limitations
- Same no-network sandbox constraint as Stage II/III: no
  sentence-transformer embedding model, no FAISS/Chroma, no live LLM API
  could be used. TF-IDF embeddings + a from-scratch NumPy vector store +
  a local extractive/templated generator are used instead, with an
  optional live-LLM code path (`generator.py::_try_live_llm`) that a
  network-enabled environment could activate by setting
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — this path is untested in this
  sandbox since it cannot be reached, and is documented as such rather
  than claimed as working.
- Knowledge base is 3 short documents (5 pages) — enough to demonstrate a
  genuine multi-document RAG pipeline with distinct, real sources, not a
  full factory documentation set.
- TF-IDF retrieval depends on vocabulary overlap; a heavily paraphrased
  query with little lexical overlap with the source PDFs may retrieve a
  weaker match than a neural embedding model would.
- `tests/test_stage4.py` was verified by direct function execution
  (12/12 passed), not via the `pytest` CLI, matching Stage II/III's
  documented workaround.

## Next Stage
Stage V — Multi-Agent System (per the 9-stage project plan) — not started;
out of scope for this handoff.

---

## Stage V — Agentic AI / Multi-Agent System

## Status
COMPLETE

## Completed
- Vision Agent: reads Stage III's existing `stage3_structured_output.json` CV result (no re-inference); `defect_type`/`severity` honestly reported as unavailable (Stage III's CNN is binary defect/normal only)
- Predictive Maintenance Agent: loads Stage II's saved GRU model (`models/gru_model.npz`, Stage II's own selected-best model) and scores a machine's latest 5-shift window - inference only, no retraining
- Knowledge/RAG Agent: thin wrapper around Stage IV's existing `RAGPipeline`; query formulated from real upstream findings, not fixed
- Planning/Decision Agent: rule-based priority/recommendation from actual field values (never hard-coded - tested that different inputs produce different outputs); optional LLM narrative-only enhancement (`ANTHROPIC_API_KEY`), recommendation/priority decided before any LLM call
- Orchestrator: lightweight custom 4-agent sequential workflow (no LangGraph added - none existed in the project, and a 4-node fan-in doesn't justify a new dependency), shared state dict, per-agent trace with status/input/output/error
- Structured final output matches the brief's schema exactly: `factory_context`, `vision_result`, `predictive_maintenance_result`, `nlp_result`, `rag_result`, `decision`, `agent_trace`
- Graceful degradation verified for an unknown machine (`M-99`): all upstream agents report `"status": "unavailable"` with a reason, Planning Agent still runs and returns `ROUTINE`/generic guidance rather than crashing or fabricating

## Important Files
- `src/agents/schemas.py` - trace/unavailable/error helpers
- `src/agents/vision_agent.py` - Vision Agent + NLP info reader (both read Stage III's existing output)
- `src/agents/predictive_maintenance_agent.py` - GRU inference wrapper
- `src/agents/knowledge_agent.py` - Stage IV RAG wrapper + query formulation
- `src/agents/planning_agent.py` - decision rules + optional LLM narrative
- `src/agents/orchestrator.py` - runs all 4 agents, builds final structured output + trace
- `tests/test_stage5.py`

## Outputs
- `reports/stage5_multi_agent_decision.json` (from `python -m src.agents.orchestrator --machine-id M-01`)

## Important Decisions
- **No LangGraph added** - project had no orchestration framework; a plain sequential function pipeline with a shared dict is sufficient for a 4-node fan-in and avoids an unjustified new dependency, matching the project's established pattern.
- Vision/NLP results are **read**, not re-inferred - Stage III already produced them, re-running the models would risk producing a different confidence than what's in the structured output.
- Predictive Maintenance Agent **does perform inference** (not just reading a file) because no per-machine live-prediction artifact existed anywhere in Stage II's output - this was necessary new integration work, done by loading Stage II's exact saved weights/contract, not by retraining.
- Risk-level banding uses the GRU's own validation-tuned `selected_threshold` (from `stage2_gru_metrics.json`) as the HIGH cutoff, never a newly-fit threshold.
- Planning Agent's recommendation logic is deterministic and rule-based over real fields; the optional LLM call only rephrases the reasoning as prose and cannot alter the recommendation/priority.

## Tests
`pytest tests/test_stage5.py -v` → 13/13 passed. Covers: Vision/NLP agents reading real Stage III data (and reporting `null` for unavailable fields, not inventing them), unavailable-machine handling, predictive maintenance agent using the real saved GRU model and being deterministic, RAG agent reusing Stage IV's pipeline with real sources, query formulation reflecting actual upstream findings, planning agent producing different outputs for different real inputs (not hard-coded), reasoning citing actual numeric values, full orchestrator schema/trace correctness, graceful handling of an unknown machine, and a checksum-style existence check that Stage I-IV artifacts are untouched.

`pytest tests/` (all stages) → 56/56 passed. A byte-for-byte `cmp` comparison of every file that existed before Stage V (200 files) confirmed **zero** were modified - only 8 new files were added.

## Known Issues
- All 6 machines' GRU predictions come back `LOW` risk in the demo run, because the test split has only 6/444 positive shifts (Stage II's own documented class-imbalance limitation) and the validation-tuned threshold is correspondingly high - this is an honest consequence of Stage II's tiny positive class, not a Stage V bug, and is stated in README rather than adjusted to look more dramatic.
- The optional live-LLM narrative path (`ANTHROPIC_API_KEY`) is implemented but untested in this environment (no key was configured for this run) - same documented-but-unverified status as Stage IV's own live-LLM path.

## Next Stage
Stage VI - Explainable AI (not started; out of scope for this handoff)

---

## Stage VI — Explainable AI (XAI)

## Status
COMPLETE

## Completed
- GRU predictive-maintenance explainer: permutation importance (documented reason SHAP DeepExplainer/KernelExplainer don't fit this hand-rolled NumPy model), verified on both the live current window and a real historical failure (M-01 day 23 Afternoon - 99.99% probability, non-trivial attributions)
- SHAP explainer for Stage II's baseline tree/linear models: real `TreeExplainer`/`LinearExplainer`, fixed a real degenerate-background bug for the linear case (single-row background gave all-zero contributions; switched to a proper 50-row train sample)
- Grad-CAM for Stage III's custom NumPy CNN: manually derived (no autodiff framework available), forward pass reproduces Stage III's exact recorded confidence, heatmap visually verified to highlight the real defect region
- Structured schema + dynamic human-readable summary (`src/xai/explanation_formatter.py`, `xai_pipeline.py`)
- Additive integration into Stage V's orchestrator (`xai_layer` step) and `planning_agent.decide()` (new optional `xai=None` param) - verified fully backward compatible

## Files Created/Modified
Created:
- `src/xai/__init__.py`, `gru_explainer.py`, `shap_tabular_explainer.py`, `gradcam_explainer.py`, `explanation_formatter.py`, `xai_pipeline.py`
- `tests/test_stage6.py`

Modified (additive only, verified via test suite + before/after checksum diff):
- `src/agents/predictive_maintenance_agent.py` - extracted `build_gru_input()`/`_load_gru_artifacts()` as reusable helpers (same external behavior, reverified via Stage V's own tests)
- `src/agents/orchestrator.py` - added the `xai_layer` step and `state["xai_result"]`
- `src/agents/planning_agent.py` - added optional `xai=None` parameter to `decide()`

## Outputs
- `reports/xai/{machine_id}_gradcam_heatmap.png`, `{machine_id}_gradcam_overlay.png`
- `reports/stage6_xai_{machine_id}.json` (from `python -m src.xai.xai_pipeline`)

## Important Decisions
- GRU explained via permutation importance, not SHAP - SHAP's DeepExplainer needs a TF/PyTorch graph, KernelExplainer would be model-agnostic but too slow for per-call agent use; permutation importance is explicitly allowed by the brief and needs only forward passes through the exact deployed model.
- SHAP used directly (unmodified) for the baseline RF/GB/LR models, since those are real scikit-learn estimators and fully SHAP-native - this is the project's literal SHAP demonstration.
- Grad-CAM implemented by hand against the model's own cached forward-pass intermediates, reusing (not duplicating) the chain-rule logic already present in `cnn_numpy.py::backward()`.
- Added an optional historical `sim_day`/`shift` override to the GRU explainer, since every machine's *current* window predicts near-zero probability (same class-imbalance consequence already documented in Stage II/V) - without it, XAI could only ever demonstrate a trivial, unattributed LOW-risk case; the override lets both the live case and a real known failure be demonstrated and tested.
- All Stage V integration changes are additive/backward-compatible by construction (new optional parameters, new dict keys) - never a required signature change - and were reverified against Stage V's original test suite after each edit.

## Tests
`pytest tests/test_stage6.py -v` → 18/18 passed. Covers all 7 required categories: tabular XAI (GRU + SHAP), feature attribution (including a real historical-failure case with non-trivial attributions), human-readable explanation (reflects real inputs, doesn't fabricate when nothing is available), vision XAI (Grad-CAM matches Stage III's exact confidence, heatmap has genuine spatial variation not noise), structured output schema, error handling (unknown machine/model never produces fake values), and Stage preservation (orchestrator still runs all 5 steps, planning agent backward-compatible without the new `xai` argument, Stage I-V test modules still import cleanly).

`pytest tests/` (all stages) → **74/74 passed** (56 existing + 18 new).

Before/after file-checksum diff against the Stage V zip confirms exactly 3 pre-existing files were touched (all listed above, all additive) and 0 Stage I-IV files were touched at all.

## Known Issues
- Same root cause as Stage V's documented limitation: all 6 machines' *current* window predicts near-zero failure probability (severe class imbalance from Stage II), so live permutation-importance contributions are tiny by construction at that point - not a bug, a consequence of a saturated sigmoid's near-flat gradient. Demonstrated instead against a real historical failure via the optional `sim_day`/`shift` parameter.
- Grad-CAM's upsampling from the conv layer's small feature-map resolution (~30x30) back to the 32x32 image uses nearest-neighbor (simplest correct approach here, since the two resolutions are already very close) rather than bilinear interpolation - documented, not a source of incorrect attribution, just a coarser visual boundary.

## Next Stage
Stage VII - Digital Twin / What-If Simulation (not started; out of scope for this handoff)

---

## Stage VII — Digital Twin / What-If Simulation

## Status
COMPLETE

## Completed
- 3 required what-if scenarios (`continue_operating`, `stop_for_maintenance`, `reduce_load`), each driven by a real forward pass through Stage II's GRU (current state, or a documented counterfactual for the maintenance/load scenarios) - never a random or hand-picked probability
- Downtime durations are empirical (computed from real `data/processed/cleaned_production.csv` DOWN/MAINTENANCE shift durations); financial constants are explicit, clearly-labelled assumptions (no pricing data exists anywhere in this project)
- Financial comparison + `recommended_scenario` (highest net value) + explicit `recommendation_basis`
- Validated against a real historical failure (M-01, day 23 Afternoon) via an optional `sim_day`/`shift` override - proves scenarios genuinely diverge under real risk, not just near-identical numbers
- Additive integration into Stage V's orchestrator (`digital_twin` step) and `planning_agent.decide()` (second optional `twin=None` parameter, alongside Stage VI's `xai`) - reverified fully backward compatible

## Files Created/Modified
Created:
- `src/digital_twin/__init__.py`, `simulator.py`, `digital_twin_pipeline.py`
- `tests/test_stage7.py`

Modified (additive only, verified via test suite + checksum diff):
- `config/config.py` - added Stage VII constants (horizon, empirical downtime hours, documented financial assumptions)
- `src/agents/orchestrator.py` - added the `digital_twin` step and `state["digital_twin_result"]`
- `src/agents/planning_agent.py` - added optional `twin=None` parameter to `decide()`

## Outputs
- `reports/stage7_digital_twin_{machine_id}.json` (from `python -m src.digital_twin.digital_twin_pipeline`)

## Important Decisions
- GRU re-scored on documented counterfactual inputs for the maintenance/load scenarios (fleet-wide healthy-state training averages for post-maintenance; real current readings scaled 20% down for load-reduction) rather than a separate rule-based risk formula - keeps every scenario's risk number traceable to the same real trained model Stage V actually uses.
- Downtime *durations* are empirical (measured from real generated data); only the *financial* constants (unit margin, hourly cost, repair cost) are documented assumptions, since this project never generated any pricing data - kept explicit in `config.py` rather than blended together.
- Cumulative risk uses standard probability compounding (`1-(1-p)^n`), not linear scaling or a random walk - documented, reproducible, and not "random numbers" per the brief's explicit prohibition.
- Added an optional historical `sim_day`/`shift` override (mirroring Stage VI's same pattern) since the live state for all 6 machines is currently low-risk - without it, the three scenarios would only ever look nearly identical, undermining the demonstration.

## Tests
`pytest tests/test_stage7.py -v` → 16/16 passed. Covers: exactly the 3 required scenarios with all required fields, correct max-net-value scenario selection, determinism (no randomness), the documented probability-compounding formula, empirical downtime duration enforcement, real historical-failure scenario differentiation, real (not fabricated) financial-constant arithmetic, configurable horizon scaling, error handling for an unknown machine, real output-file persistence, orchestrator integration, and Stage V backward compatibility (both with and without the new `twin` argument).

`pytest tests/` (all stages) → **90/90 passed** (74 existing + 16 new).

Before/after file-checksum diff against the Stage VI zip confirms exactly 3 pre-existing files were touched (all listed above, all additive) and 0 Stage I-IV files were touched at all.

## Known Issues
- Same root cause as Stages V/VI: all 6 machines' *live* state is currently low-risk (Stage II's documented class imbalance), so `continue_operating` is the default winner for all of them today - addressed the same way, via the optional historical-window override for demonstration.
- Financial constants (unit margin $12/unit, $150/downtime-hour, $2000/failure) have no real pricing data behind them anywhere in this project and should be treated as illustrative, not calibrated - stated plainly here and in `config.py`'s comments rather than presented as measured.

## Next Stage
Stage VIII - Human-in-the-Loop + MLOps/MLflow (not started; out of scope for this handoff)

---

## Stage VIII — Human-in-the-Loop + MLOps/MLflow

## Status
COMPLETE

## Completed
- HITL: APPROVE/REJECT/MODIFY workflow over real Stage V orchestrator output, with REJECT/MODIFY requiring a comment (and MODIFY requiring a modified_action), original AI recommendation stored immutably and verified identical across all 3 decision types on the same input
- Append-only JSONL audit log (`reports/hitl_audit_log.jsonl`) with unique decision IDs, timestamps, full supporting evidence (vision/PM/XAI/digital-twin/RAG), and retrieval helpers
- MLflow: 4 real runs (3 baseline + GRU) logging actual Stage II params/metrics from `reports/stage2_*_metrics.json` - no retraining, no invented values
- Model Registry: registers Stage II's own already-selected best model; fixed a real bug where the GRU's plain file artifact couldn't be registered (wrapped it as a proper `mlflow.pyfunc.PythonModel` instead) - verified registered-model predictions match the original exactly

## Files Created/Modified
Created only (verified 0 pre-existing files touched):
- `src/hitl/audit_store.py`, `hitl_workflow.py`, `hitl_pipeline.py` (`src/hitl/__init__.py` already existed, empty)
- `src/mlops/mlflow_logging.py` (already existed from earlier in this session, `GRUPyfuncWrapper`/pyfunc logging added to fix the registry bug; `src/mlops/__init__.py` already existed, empty)
- `tests/test_stage8.py`
- `requirements.txt`, `README.md`, `.gitignore` updated

## Outputs
- `reports/hitl_audit_log.jsonl` - append-only human decision audit trail
- `mlflow.db` (SQLite tracking store), `mlruns/` (artifact store) - both gitignored, regenerable via `src/mlops/mlflow_logging.py`
- MLflow experiment `ai_factory_stage2_failure_prediction`, registered model `ai_factory_failure_predictor`

## Important Decisions
- Found `src/mlops/mlflow_logging.py` and `config.py`'s HITL-adjacent scaffolding already on disk at the start of this task (from earlier in this session) - inspected and ran it first rather than rebuilding, per the task's own "inspect before modifying" instruction. Found and fixed one real bug (GRU registry logging) rather than assuming it worked.
- JSONL append-only file chosen for the audit log over a database, consistent with every other stage's plain-file storage choice in this project - MLflow's SQLite remains the only database, used only for MLflow itself.
- REJECT requires a comment; MODIFY requires both a modified_action and a comment; APPROVE requires neither - matches the brief's description of what each decision type needs to be meaningful.
- `ai_evidence` (vision/PM/XAI/digital-twin/RAG) is stored in full inside each audit record so it's self-contained and auditable without needing to re-run the pipeline later.
- GRU wrapped as `mlflow.pyfunc.PythonModel` specifically so the Model Registry works for it - this is packaging for logging purposes only; the underlying trained weights and predictions are untouched (verified byte-identical).

## Tests
`pytest tests/test_stage8.py -v` → 19/19 passed. Covers all 11 required categories: approve/reject/modify workflows (including required-field validation), preservation of the original AI recommendation (byte-identical across all 3 decision types), human comment storage, audit record uniqueness/timestamp/persistence/retrieval, MLflow run creation (4 runs, exceeds the required 3), real parameter/metric logging verified against actual Stage II reports, Model Registry registration of Stage II's own selected best model, registered-model prediction correctness, and Stage I-VII test-module preservation.

`pytest tests/` (all stages) → **109/109 passed** (90 existing + 19 new).

Before/after file-checksum diff against the Stage VII zip confirms 7 new files and **0** pre-existing files modified.

## Known Issues
- MLflow's cloudpickle-based serialization of the GRU pyfunc wrapper logs a caution warning (safe here, since it's only this project's own code, but relevant if the registry were ever shared across untrusted environments).
- The registered "best" model is whichever Stage II selected by validation F1 at the time `log_all_stage2_runs()` is run (currently the GRU) - if Stage II is ever rerun with different results, the registry will register a new version pointing at the new best model, which is the intended behavior, not a limitation, but worth noting.

## Next Stage
Stage IX - Web Application + Automated Report (not started; out of scope for this handoff)

---

## Stage IX — Web Application + Automated Report

## Status
COMPLETE

## Completed
- `app.py` (Streamlit, repo root): 7-tab UI over the real Stage I-VIII pipeline - Prediction, XAI, RAG Evidence, Multi-Agent Recommendation, Digital Twin, Human Decision (APPROVE/REJECT/MODIFY), Report - no modeling/business logic in the app itself, only calls into existing modules
- Live image-upload inference through the actual trained Stage III CNN (`gradcam_explainer.explain_uploaded_image`, already existed from earlier in this session - verified it reproduces Stage III's stored confidence on the same image)
- PDF report generator (`src/reporting/report_generator.py`, already existed from earlier in this session - verified it produces valid, non-trivial PDFs and correctly includes/excludes the HITL record)
- Hugging Face Spaces deployment: YAML frontmatter added to `README.md`, `streamlit` added to `requirements.txt`
- **Fixed a real deployment-breaking bug**: `.gitignore` was excluding `models/*.pkl`, `data/processed/*`, and `reports/*.json` - all of which `app.py` reads directly at runtime. A `git`-based HF Spaces deploy would have shipped an app with no trained models or data. Rewrote `.gitignore` to keep everything the app actually needs committed, excluding only regenerable raw data and session-runtime files.

## Files Created/Modified
Created:
- `app.py`
- `tests/test_stage9.py`

Modified (all additive/config, verified via test suite):
- `requirements.txt` (added `streamlit`)
- `.gitignore` (fixed to stop excluding runtime-required artifacts - see above)
- `README.md` (HF Spaces frontmatter + Stage IX section)

Found already on disk from earlier in this session (inspected and verified, not rebuilt):
- `src/reporting/report_generator.py`, `src/reporting/__init__.py`
- `src/xai/gradcam_explainer.py::explain_uploaded_image` (+ its `_explain_array` refactor)

## Human-in-the-Loop-in-the-UI
The Human Decision tab requires a REJECT/MODIFY reason exactly like
Stage VIII's underlying `submit_decision()` - tested that an invalid
submission shows a UI error rather than crashing or silently accepting it.

## MLflow/MLOps
Unchanged from Stage VIII - the web app does not call MLflow directly (out
of Stage IX's scope: "web application + automated report", not MLOps UI).

## Tests/Validation
`pytest tests/test_stage9.py -v` → 12/12 passed, using Streamlit's real
`AppTest` harness (not a mock): app load, full pipeline execution, machine
switching, APPROVE/REJECT(validation-error)/MODIFY flows in the actual UI,
PDF generation + download button, report-generator correctness (valid PDF
bytes, HITL inclusion, graceful handling of missing upstream results), and
live-upload inference matching the real stored Stage III result.

`pytest tests/` (all stages) → **121/121 passed** (109 existing + 12 new).

## Known Issues
- No automated check that the HF Spaces YAML frontmatter itself is valid outside of Hugging Face's own parser - only manually reviewed against their documented schema.
- The report is PDF only (the brief allows "PDF and/or DOCX") - no DOCX variant was built, since one format already satisfies the requirement and a second would duplicate the same assembly logic.

## Next Stage
None - Stage IX was the final planned stage.
