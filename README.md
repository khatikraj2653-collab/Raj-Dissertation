# Raj-Dissertation

MSc Applied AI dissertation project by Raj Khatik, *"Understanding the Role of Artificial Intelligence in Life Cycle Assessment"*, WMG, University of Warwick, supervised by Dr. You Wu, completed September 2026. This repository holds the technical evaluation component of the dissertation: an LLM benchmarking pipeline that tests how accurately large language models (Claude, GPT, Gemini, Llama, GPT-OSS) can extract structured Life Cycle Inventory (LCI) fields from natural-language descriptions of ecoinvent activities, alongside a complementary unsupervised-learning pipeline that clusters those same activities by environmental impact profile and flags statistical anomalies.

## Stack

- **openpyxl** — reading the raw ecoinvent LCIA Excel workbook (`RAJ_DISS.xlsx`)
- **pandas** — data loading, sampling, feature/CSV handling
- **openai** — GPT-4o-mini / GPT-5.6 Terra API calls
- **anthropic** — Claude Haiku 4.5 API calls
- **python-dotenv** — loading API keys from `.env`
- (used by individual scripts, not pinned in `requirements.txt`) **google-genai** for Gemini 3.5 Flash, **groq** for Llama 3.3 70B and GPT-OSS 120B, **scikit-learn** (KMeans, IsolationForest, PCA, StandardScaler/PowerTransformer) for clustering/anomaly detection, **scipy** (Cochran's Q, McNemar's, binomial tests) for significance testing

## Architecture

```mermaid
flowchart TD
    subgraph Source["Data source"]
        XLSX["data/RAJ_DISS.xlsx (ecoinvent LCIA sheet)"]
    end

    subgraph ExtractionTask["Objective 3 - Structured field extraction"]
        SAMPLE["sample_data.py / expand_sample.py / build_hard_extraction_task.py"]
        TESTSETS["data/test_set.json, test_set_200.json, test_set_hard.json"]
        PROMPTS["prompts.py (EXTRACTION_SYSTEM_PROMPT)"]
        RUNEVAL["run_evaluation.py (Claude+GPT, 100 cases)"]
        RUNMULTI["run_multi_model_evaluation.py (6 models, 200 cases)"]
        RUNHARD["run_hard_task_evaluation.py (6 models, paraphrased 50 cases)"]
        RAW["results/raw_responses.json, multi_model_raw_responses.json, hard_task_raw_responses.json"]
        SCORE["score_results.py / score_multi_model.py / score_hard_task.py"]
        SUMMARY["results/summary.json, multi_model_summary.json, hard_task_summary.json (accuracy, Cochran's Q, McNemar's pairwise)"]
    end

    subgraph DraftTask["Objective 3 - Goal & Scope drafting (rubric task)"]
        DRAFTGEN["generate_drafts.py (Claude+GPT, 10 cases)"]
        DRAFTMULTI["generate_drafts_multi_model.py (5 models)"]
        DRAFTOUT["results/goal_scope_drafts.json, goal_scope_drafts_multi_model.json"]
        DRAFTSCORE["score_drafts_multi_model.py"]
        DRAFTSUMMARY["results/task2_rubric_summary.json"]
    end

    subgraph ClusterTask["Environmental-profile clustering & anomaly detection"]
        PREPFEAT["prepare_features.py"]
        CLEANFEAT["data/clean_features.csv (68 climate-change indicators)"]
        REDUNDANCY["check_redundancy.py"]
        REDUCED["data/reduced_features.csv (7 representative indicators)"]
        CLUSTER["cluster_and_anomaly.py (Yeo-Johnson + StandardScaler -> KMeans k=8, IsolationForest, PCA)"]
        CLUSTEROUT["results/clustering_anomaly_results.csv"]
        MULTICLUSTER["multi_clustering_comparison.py / multi_anomaly_comparison.py"]
        CLUSTERSUMMARY["results/clustering_comparison_summary.json, anomaly_comparison_summary.json, redundancy_summary.json"]
    end

    XLSX --> SAMPLE --> TESTSETS
    PROMPTS --> RUNEVAL
    PROMPTS --> RUNMULTI
    TESTSETS --> RUNEVAL --> RAW
    TESTSETS --> RUNMULTI --> RAW
    TESTSETS --> RUNHARD --> RAW
    RAW --> SCORE --> SUMMARY

    XLSX --> DRAFTGEN --> DRAFTOUT
    DRAFTOUT --> DRAFTMULTI --> DRAFTOUT
    DRAFTOUT --> DRAFTSCORE --> DRAFTSUMMARY

    XLSX --> PREPFEAT --> CLEANFEAT
    CLEANFEAT --> REDUNDANCY --> REDUCED
    REDUNDANCY --> CLUSTERSUMMARY
    REDUCED --> CLUSTER --> CLUSTEROUT
    CLUSTEROUT --> MULTICLUSTER --> CLUSTERSUMMARY
```

## Repository structure

- **`data/`** — the raw ecoinvent LCIA workbook (`RAJ_DISS.xlsx`, gitignored), derived feature CSVs (`clean_features.csv`, `reduced_features.csv`), and generated LLM test sets (`test_set.json`, `test_set_200.json`, `test_set_hard.json`) built by the sampling scripts.
- **`scripts/`** — the pipeline code: test-set builders, prompt templates, per-model API call wrappers (Claude/GPT/Gemini/Llama/GPT-OSS), scoring/statistical-testing scripts for the extraction and drafting tasks, and the feature-preparation/clustering/anomaly-detection scripts for the environmental-profile analysis.
- **`results/`** — all pipeline outputs: raw and parsed model responses, accuracy/statistical-test summaries (Cochran's Q, Bonferroni-corrected McNemar's), goal-and-scope draft text and rubric scores, clustering/anomaly CSVs, and the indicator correlation/redundancy analysis.
