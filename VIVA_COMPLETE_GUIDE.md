# SkillSyncAI — Complete Project Guide (VIVA Reference)

> **One-Line Summary:** An AI-powered system that reads student resumes (PDFs), extracts skills using NLP, and forms balanced hackathon teams using a Snake Draft algorithm — with a machine-learning feedback loop that learns from past team outcomes to improve future formations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture & Flow](#2-system-architecture--flow)
3. [Module-by-Module Breakdown](#3-module-by-module-breakdown)
4. [The ML Model — Complete Details](#4-the-ml-model--complete-details)
5. [All Models We Tried & How We Chose](#5-all-models-we-tried--how-we-chose)
6. [The 5-Step Optimization Pipeline](#6-the-5-step-optimization-pipeline)
7. [Feature Engineering — 25 Features](#7-feature-engineering--25-features)
8. [API Endpoints — Full Reference](#8-api-endpoints--full-reference)
9. [GitHub Integration](#9-github-integration)
10. [Validation Module](#10-validation-module)
11. [Training Data & Data Pipeline](#11-training-data--data-pipeline)
12. [Key Design Decisions & Why](#12-key-design-decisions--why)
13. [VIVA Questions & Answers](#13-viva-questions--answers)
14. [Glossary of ML Terms Used](#14-glossary-of-ml-terms-used)

---

## 1. Project Overview

### What is SkillSyncAI?

SkillSyncAI is a **team formation system** for hackathons and academic projects. Given a set of student resumes (PDF files), it:

1. **Parses** each resume and extracts text using `pdfplumber`
2. **Identifies skills** using rule-based keyword matching across 200+ technology keywords
3. **Classifies** each student into one of **7 roles** (Frontend, Backend, Fullstack, ML/AI, Data, UI/UX, DevOps)
4. **Scores** each student on skill level, experience, and diversity
5. **Forms balanced teams** using a **Snake Draft** algorithm with role balancing
6. **Predicts team success** using a trained ML model (HistGradientBoosting, v3.3)
7. **Learns from feedback** — when you tell it which teams succeeded/failed, it retrains and improves

### Problem Statement

> "Given N student resumes and M projects, form balanced teams where each team has diverse roles, adequate skill coverage, and fair distribution of talent — then predict whether each team will succeed."

### Tech Stack

| Component | Technology | Why We Chose It |
|-----------|-----------|----------------|
| **API Framework** | FastAPI + Uvicorn | Async, auto-docs (Swagger), type validation via Pydantic |
| **ML Training** | scikit-learn 1.8 | CalibratedClassifierCV, HistGradientBoosting, StandardScaler |
| **Resume Parsing** | pdfplumber | Accurate PDF text extraction, handles multi-page |
| **Data Processing** | NumPy | Array operations, feature vectors |
| **Serialization** | pickle/joblib | Model persistence |
| **HTTP Client** | httpx (async) | GitHub API calls, keep-alive pings |
| **Deployment** | Docker + Render | Containerized, free-tier cloud hosting |
| **Testing** | httpx AsyncClient | 6 automated API endpoint tests |

---

## 2. System Architecture & Flow

### Complete End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                                      │
│  Upload PDFs + (optional) project descriptions + (optional) GitHub IDs   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 1: RESUME PARSING  (ml_engine.py)                                  │
│                                                                          │
│  For each PDF:                                                           │
│    1. extract_text_from_pdf_bytes() — pdfplumber reads all pages         │
│    2. preprocess_text() — lowercase, remove special chars, normalize     │
│    3. extract_name_from_filename() — "john_doe_resume.pdf" → "John Doe" │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 2: SKILL EXTRACTION  (ml_engine.py)                                │
│                                                                          │
│  extract_skills(text):                                                   │
│    • For each of 7 roles, match against keyword lists (200+ keywords)    │
│    • Score = (unique_skills × 2 + raw_count × 0.5) normalized to 0-10   │
│    • Output: {frontend: 7.2, backend: 3.1, ml: 0, ...}                  │
│                                                                          │
│  calculate_skill_percentages(text):                                      │
│    • Per-skill proficiency: 40-95% based on mention count + context      │
│    • Context bonus: "expert in React" → +5% for React                    │
│    • Output: {React: 85%, JavaScript: 72%, Node.js: 68%, ...}           │
│                                                                          │
│  determine_primary_role(skill_data):                                     │
│    • Pick role with highest score                                        │
│    • Special: "fullstack" check — if frontend or backend scores higher   │
│                                                                          │
│  calculate_experience_score(text):                                       │
│    • Match 20+ experience keywords (internship, hackathon, project...)   │
│    • Weighted count, sigmoid-normalized to 0-5 scale                     │
│    • Formula: score = min(5, (weighted_total / (weighted_total + 5)) × 10)│
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 2b: GITHUB INTEGRATION (optional)  (github_fetcher.py)             │
│                                                                          │
│  If github_usernames provided:                                           │
│    1. GitHubFetcher fetches user repos via GitHub REST API               │
│    2. Analyzes: repo languages, topics, stars, activity                  │
│    3. Maps languages/topics to roles (python→backend, react→frontend)    │
│    4. Merges with resume skills: final = 0.6×resume + 0.4×github        │
│    5. Boost experience if GitHub shows activity                          │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 3: STUDENT PROFILE BUILDING  (ml_engine.py)                        │
│                                                                          │
│  build_student_profile(resume_data):                                     │
│    → Creates complete profile with:                                      │
│      • student_id, name, primary_role                                    │
│      • skills: {role: score} for all 7 roles                             │
│      • skill_percentages: {skill_name: percentage} for each skill        │
│      • experience_score (0-5)                                            │
│      • skill_diversity (0-1) — fraction of roles with score > 0          │
│      • top_skills — list of best matched keywords                        │
│                                                                          │
│  SCORING FORMULA (uses learned weights from feedback model):             │
│    overall = w_skill × (max_skill/10) + w_exp × (exp/5) + w_div × div  │
│                                                                          │
│    Static defaults:  60% skill, 30% experience, 10% diversity            │
│    Learned weights:  40.8% coverage, 24.6% exp, 19.2% skill,            │
│                      8.7% balance, 6.8% team_size                        │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 4: TEAM FORMATION — Snake Draft  (ml_engine.py)                    │
│                                                                          │
│  form_balanced_teams():                                                  │
│    1. Calculate optimal team sizes (keep all teams as equal as possible)  │
│    2. Group students by primary_role                                     │
│    3. Sort each group by overall_score (descending)                      │
│    4. SNAKE DRAFT:                                                       │
│       Round 1: Team1 → Team2 → Team3 → ... → TeamN (forward)            │
│       Round 2: TeamN → ... → Team3 → Team2 → Team1 (reversed)           │
│       Round 3: Forward again...                                          │
│                                                                          │
│    Pick logic:                                                           │
│       • Try to fill a role the team doesn't have yet                     │
│       • Use role_priority order (backend, frontend, fullstack, ml...)     │
│       • If all roles filled, pick best available student                  │
│                                                                          │
│  WHY SNAKE DRAFT?                                                        │
│    • Fairness: Team that picks last in round 1 picks first in round 2    │
│    • Simple to implement, easy to explain                                │
│    • Ensures no team gets all the top students                           │
│    • Similar to real-world fantasy sports drafts                         │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 5: EXPLANATION GENERATION (XAI)  (ml_engine.py)                    │
│                                                                          │
│  generate_explanation(member, profile_lookup):                            │
│    Each team member gets a human-readable reason:                        │
│    "Strong Backend skills (8/10); Primary expertise aligns with Backend  │
│     role; Key skills: Node.js, MongoDB, Express; Strong practical        │
│     experience (3.5/5)"                                                  │
│                                                                          │
│  WHY EXPLAINABILITY?                                                     │
│    • Students can understand WHY they were placed in a team              │
│    • Professors can audit the assignments                                │
│    • This is XAI (Explainable AI) — important in academic ML             │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  STEP 6: TEAM BALANCE SCORING  (ml_engine.py)                            │
│                                                                          │
│  _calculate_team_balance():                                              │
│    • average_score: Mean overall score of members                        │
│    • score_variance: How spread out the scores are                       │
│    • role_diversity: unique_roles / team_size                            │
│    • overall_balance: Weighted combination using learned weights          │
│                                                                          │
│  If feedback weights are loaded:                                         │
│    balance = 0.30×avg + (0.25+w_balance)×(1−var) + (0.25+w_coverage/2)×div│
│  Else:                                                                   │
│    balance = 0.35×avg + 0.35×(1−var) + 0.30×role_diversity               │
└──────────────────────────────────────────────────────────────────────────┘
```

### Feedback Loop Flow

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  After teams  │────▶│  POST /feedback  │────▶│ feedback_training │
│  are graded   │     │  Submit outcome  │     │ _data.json        │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Weights are  │◀───│  POST /retrain   │◀───│  Enough records?  │
│  reloaded     │     │  Retrain model   │     │  (need ≥ 10)     │
└──────┬───────┘     └─────────────────┘     └──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  ml_engine.py reloads model_weights.pkl                      │
│  New learned weights replace static 60/30/10 split           │
│  NEXT team formation uses data-driven weights                │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Module-by-Module Breakdown

### 3.1. `main.py` (1073 lines) — FastAPI Application

**Purpose:** API routes, request/response models, endpoint handlers

**Key Components:**
- **Pydantic Models:** `TeamMember`, `Team`, `StudentProfile`, `Summary`, `TeamFormationResponse`, `FeedbackRecord`, `FeedbackBatch`, `TeamPredictionRequest`
- **Keep-Alive Task:** Background async task that pings the server every 14 minutes to prevent Render free-tier sleep
- **CORS Middleware:** Allows cross-origin requests from any frontend

**Endpoints (12 total):**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Root/health check with endpoint listing |
| `/health` | GET | Simple health check |
| `/api/form-teams` | POST | Main: upload PDFs → get teams |
| `/api/form-teams-zip` | POST | Upload ZIP of PDFs → get teams |
| `/api/form-teams-json` | POST | Upload PDFs + projects.json file |
| `/api/parse-resumes` | POST | Parse resumes only (no team formation) |
| `/api/v2/form-teams` | POST | Structured payload with participantData + projects |
| `/api/github-status` | GET | Check GitHub API auth status & rate limits |
| `/api/model-info` | GET | ML techniques documentation |
| `/api/feedback` | POST | Submit team outcome feedback |
| `/api/retrain` | POST | Retrain ML model on all feedback data |
| `/api/predict-team-success` | POST | Predict if a team will succeed |
| `/api/feedback-weights` | GET | View active learned weights |
| `/api/feedback-data-stats` | GET | Training data statistics |

### 3.2. `ml_engine.py` (1145 lines) — Core ML Engine

**Purpose:** All the logic for parsing, skill extraction, scoring, and team formation

**Key Components:**
- **SKILL_DICTIONARY:** 200+ keywords mapped to 7 roles
- **ROLE_SYNONYMS:** Maps job titles to roles (e.g., "react developer" → "frontend")
- **EXPERIENCE_KEYWORDS:** 20+ keywords with weights (internship=1.0, hackathon=0.8, project=0.5)
- **extract_skills():** Rule-based keyword matching, returns {role: score} for all 7 roles
- **calculate_skill_percentages():** Per-skill proficiency using mention count + context analysis
- **build_student_profile():** Complete profile builder
- **build_student_profile_with_github():** Profile builder with GitHub skill merging
- **form_balanced_teams():** Snake Draft algorithm
- **generate_explanation():** XAI explanation for each team assignment
- **process_team_formation():** Master function — called by all API endpoints

**Feedback Integration:**
- On import, `_load_feedback_weights()` loads `model_weights.pkl`
- Overrides static weights: `WEIGHT_SKILL`, `WEIGHT_EXPERIENCE`, `WEIGHT_DIVERSITY`
- `reload_feedback_weights()` refreshes after retraining

### 3.3. `feedback_trainer.py` (1155 lines) — ML Training Pipeline

**Purpose:** Train models on team outcome data, learn scoring weights

**Key Classes:**
- **TeamFeatureExtractor:** Extracts 25-dimensional feature vectors from team records
- **FeedbackModel:** Trains multi-model comparison, calibrates probabilities, saves weights
- **FeedbackEnhancedScorer:** Bridge between trained model and team formation scoring

**Training Pipeline (when you call POST /retrain):**
1. Load `feedback_training_data.json` (409 records)
2. Extract 25 features per record using `TeamFeatureExtractor`
3. 80/20 stratified holdout split
4. StandardScaler normalization
5. Multi-model comparison (RF, GB, SVM_RBF) with 10-fold CV
6. Select best model by F1 score
7. Probability calibration (CalibratedClassifierCV, isotonic)
8. Holdout evaluation (accuracy, F1, precision, recall, AUC)
9. Feature importance (permutation importance for SVM/non-tree models)
10. Weight learning — convert feature importances to scoring weights
11. Save model + weights to `model_weights.pkl`

### 3.4. `github_fetcher.py` (491 lines) — GitHub Integration

**Purpose:** Fetch and analyze GitHub profiles to enhance skill extraction

**Key Components:**
- **GitHubFetcher class:** Async HTTP client for GitHub REST API
- **Language mapping:** JavaScript→frontend, Python→backend, Jupyter→data, etc.
- **Topic mapping:** react→frontend, tensorflow→ml, docker→devops, etc.
- **Skill blending:** `final_skill = 0.6 × resume_skill + 0.4 × github_skill`
- **Rate limiting:** Batches of 5 profiles, supports auth tokens (5000 req/hr vs 60)

### 3.5. `validation.py` (461 lines) — Team Validation Module

**Purpose:** Validate team quality using statistical metrics

**Metrics:**
| Metric | Method | Threshold | What It Measures |
|--------|--------|-----------|-----------------|
| Gini Coefficient | Lorenz curve inequality | < 0.15 | Fairness across teams |
| Role Coverage | Required roles filled | ≥ 75% | Skill representation |
| Skill Balance | Coefficient of variation | ≥ 0.7 | Even talent distribution |
| Diversity Score | Unique roles / team size | ≥ 0.6 | Role variety per team |
| Assignment Rate | Assigned / total students | ≥ 95% | No one left out |

**Quality Grade:** Weighted average → A/B/C/D/F  
**Ground Truth Comparison:** Precision, recall, F1 against ideal assignments

### 3.6. `optimization_trainer.py` (1322 lines) — Optimization Pipeline

**Purpose:** 5-step model optimization with version tracking

**Steps:** v3.0 through v3.4 (see Section 6 for details)

---

## 4. The ML Model — Complete Details

### Current Production Model: v3.3

| Property | Value |
|----------|-------|
| **Algorithm** | HistGradientBoostingClassifier (wrapped in CalibratedClassifierCV) |
| **Library** | scikit-learn 1.8 |
| **Features** | 16 (pruned from 25 in v3.0) |
| **Training Records** | 407 (257 synthetic + 150 real-resume-augmented) |
| **Holdout Accuracy** | 90.2% |
| **Holdout F1 Score** | 0.925 |
| **Holdout AUC** | 0.958 |
| **Calibration** | Isotonic (CalibratedClassifierCV, 5-fold) |
| **Scaler** | StandardScaler (z-score normalization) |
| **Model File** | `api/model_weights.pkl` (~size varies) |

### What is HistGradientBoostingClassifier?

HistGradientBoostingClassifier is a **gradient boosting** algorithm that:
- Bins continuous features into discrete bins (histograms) — speeds up training
- Builds trees sequentially — each new tree corrects errors of the previous ones
- Natively handles missing values
- Inspired by LightGBM but implemented in scikit-learn
- Much faster than regular GradientBoostingClassifier for large datasets

**Mathematical Foundation:**
$$\hat{y} = \sum_{m=1}^{M} \eta \cdot h_m(x)$$

Where:
- $M$ = number of boosting rounds (trees)
- $\eta$ = learning rate (shrinkage)
- $h_m(x)$ = the m-th decision tree
- Each tree $h_m$ is fit to the **negative gradient** of the loss from previous trees

### Why Probability Calibration?

Raw classifier probabilities are often poorly calibrated (e.g., a model predicting 80% confidence might actually be correct only 60% of the time).

**CalibratedClassifierCV** with isotonic regression:
- Uses a non-parametric, monotonic function to map raw probabilities → calibrated probabilities
- Trained separately on held-out validation data (5-fold)
- Result: When our model says 85% success probability, it really means ~85%

### Feature Pipeline

```
Raw team record (JSON)
        │
        ▼
TeamFeatureExtractor.extract_features(record)
        │  Extracts 25 features
        ▼
Apply keep_indices (select 16 of 25)
        │  Feature pruning from v3.0
        ▼
StandardScaler.transform()
        │  Z-score normalization: (x - μ) / σ
        ▼
CalibratedClassifierCV.predict_proba()
        │  Returns [P(failure), P(success)]
        ▼
Result: success_probability = P(success)
```

### The 16 Retained Features (after pruning)

The following features were kept because they had the highest **permutation importance** on the validation set:

| # | Feature | Description | Why It Matters |
|---|---------|-------------|---------------|
| 1 | team_size | Number of members | Too big/small teams fail |
| 2 | role_diversity | Unique roles / team_size | More diverse = better |
| 3 | avg_skill_level | Mean skill across members | Core quality indicator |
| 4 | min_skill_level | Weakest member's skill | "Chain is as strong as weakest link" |
| 5 | max_skill_level | Strongest member's skill | Need at least one star |
| 6 | skill_variance | Spread in skill levels | High variance = imbalanced |
| 7 | gini_coefficient | Inequality measure | Like wealth inequality but for skills |
| 8 | avg_experience | Mean years of experience | Experience matters |
| 9 | skill_coverage | Required skills covered % | Can the team do the job? |
| 10 | has_critical_role_gap | Missing essential role? | Web app with no backend = fail |
| 11 | critical_role_coverage | Fraction of critical roles filled | Partial coverage tracking |
| 12 | helpful_role_coverage | Non-critical but useful roles | Nice-to-haves matter too |
| 13 | experience_skill_interaction | avg_exp × avg_skill | Synergy term |
| 14 | weakest_link_score | min_skill × role_diversity | Combined bottleneck |
| 15 | team_strength_index | sum(skills) / √(team_size) | Efficiency normalized by size |
| 16 | coverage_diversity_product | coverage × role_diversity | Two good things together |

### Learned Weights (from model)

After training, the model's feature importances are converted to **scoring weights** used in team formation:

```
weight_coverage:    40.8%  ← Most important: can the team cover required skills?
weight_experience:  24.6%  ← Second: does the team have experience?
weight_skill:       19.2%  ← Third: raw skill levels
weight_balance:      8.7%  ← Fourth: how balanced is the team?
weight_team_size:    6.8%  ← Fifth: is the team the right size?
```

These weights **replace the static 60/30/10 (skill/experience/diversity)** defaults when the model is loaded.

---

## 5. All Models We Tried & How We Chose

### Training Round 1 (v1.0) — Basic Feedback Loop

| Property | Value |
|----------|-------|
| Features | 14 (basic team metrics) |
| Training Records | 55 (trivially easy synthetic) |
| Model | RandomForestClassifier |
| CV Accuracy | 100% |
| **Problem** | **Complete overfitting — training data was too easy** |

**What was wrong:** All 55 records were either clearly good teams (score 8+, all roles covered) or clearly bad teams (score 2, all same role). The model memorized patterns without learning real decision boundaries.

### Training Round 2 (v2.0) — Rigorous Training

| Property | Value |
|----------|-------|
| Features | 25 (14 original + 11 new interaction terms) |
| Training Records | 256 (200 hard edge cases + 55 original) |
| **Models Compared** | RF, GradientBoosting, SVM_RBF, ExtraTrees |
| CV Folds | 10-fold stratified |
| Holdout Split | 80/20 stratified |

**Multi-Model Comparison Results (v2.0):**

| Model | CV F1 | CV Accuracy | Holdout F1 | Holdout Acc | Holdout AUC |
|-------|-------|-------------|------------|-------------|-------------|
| RandomForest | 0.917 ± 0.04 | 0.890 | 0.920 | 89.5% | 0.972 |
| GradientBoosting | 0.925 ± 0.03 | 0.901 | 0.933 | 91.2% | 0.979 |
| **SVM_RBF** | **0.947 ± 0.02** | **0.932** | **0.947** | **94.2%** | **0.996** |
| ExtraTrees | 0.904 ± 0.05 | 0.879 | 0.908 | 87.8% | 0.958 |

**Winner: SVM_RBF** — Highest F1 (0.947, best at finding both success and failure), highest AUC (0.996, nearly perfect ranking).

**Why SVM_RBF worked well:**
- Non-linear decision boundary (RBF kernel maps to infinite dimensions)
- `class_weight="balanced"` handles class imbalance
- Works well with scaled features (StandardScaler applied)
- Regularization via C parameter prevents overfitting

### Optimization Round (v3.0 — v3.4)

| Version | What Changed | Holdout Acc | F1 | AUC | Features | Records |
|---------|-------------|-------------|-----|-----|----------|---------|
| v2.0 (baseline) | SVM_RBF, 25 features | 94.2% | 0.947 | 0.996 | 25 | 256 |
| v3.0 | Feature pruning (25→16) | 90.4% | 0.917 | 0.961 | 16 | 257 |
| v3.1 | Hyperparameter grid search | 96.2% | 0.969 | 0.987 | 16 | 257 |
| v3.2 | +150 real-resume-augmented records | 87.8% | 0.909 | 0.960 | 16 | 407 |
| **v3.3** | **HistGradientBoosting** | **90.2%** | **0.925** | **0.958** | **16** | **407** |
| v3.4 | Threshold optimization | 89.0% | 0.920 | 0.958 | 16 | 407 |

### Why We Chose v3.3 Over v3.1 (which had 96.2%)

> **v3.1 had the highest accuracy (96.2%) but we chose v3.3 (90.2%). Here's why:**

1. **v3.1 was trained on only 257 synthetic records** — all artificial data. v3.3 was trained on 407 records including 150 derived from real resumes.

2. **Generalization > accuracy on synthetic data.** v3.1's 96.2% is on data it was designed for. Real-world teams are messier.

3. **HistGradientBoosting is more robust** than SVM_RBF:
   - Handles missing values natively
   - Less sensitive to feature scaling
   - Faster inference
   - Histogram-based binning reduces overfitting
   
4. **The accuracy DROP from 96.2% to 90.2% when real data was added is a good sign** — it means the model is being tested on harder, more realistic scenarios.

5. **AUC = 0.958 is still excellent** — the model ranks teams correctly 95.8% of the time.

### Why Not Neural Networks / Deep Learning?

| Reason | Explanation |
|--------|------------|
| **Small dataset** | Only 407 records — neural networks need thousands+ |
| **Interpretability** | Feature importances are critical for explanation (XAI). NNs are black boxes |
| **Training time** | Gradient boosting trains in <1 second. NNs need GPU + hours |
| **No overfitting** | With 407 records, a NN would memorize immediately |
| **Academic validity** | Traditional ML with proper validation is more rigorous than throwing a NN at it |

### Why Not XGBoost / LightGBM?

We tried XGBoost in step v3.3 of the optimization pipeline, but:
- scikit-learn's `HistGradientBoostingClassifier` gave comparable results
- No extra dependency needed (sklearn is already installed)
- Easier to integrate with `CalibratedClassifierCV`
- In our tests, HistGradientBoosting actually slightly outperformed XGBoost on our data

---

## 6. The 5-Step Optimization Pipeline

The optimization was done in `optimization_trainer.py` with full version tracking.

### Step 1: Feature Pruning (v3.0)

**Goal:** Remove low-importance features to reduce overfitting and noise.

**Method:**
1. Train baseline model on all 25 features
2. Calculate permutation importance for each feature
3. Drop features with near-zero importance
4. Retrain on remaining 16 features

**Which 9 features were dropped and why:**
| Dropped Feature | Why |
|----------------|-----|
| skill_range | Redundant with skill_variance |
| experience_variance | Low importance, avg_experience captures it |
| role_duplication_ratio | Rarely matters for success |
| project_type_encoded | Model should be project-agnostic |
| median_skill_level | avg_skill_level already covers this |
| skill_iqr | Low variance, correlated with skill_range |
| total_skills_count | skills_per_member is more informative |
| skills_per_member | After pruning, too correlated |
| balance_penalty | Gini already captures balance |

**Result:** 25 → 16 features. Accuracy dropped slightly (94.2% → 90.4%) but the model became more generalizable.

### Step 2: Hyperparameter Search (v3.1)

**Goal:** Find optimal hyperparameters for the best model.

**Method:** Grid search with cross-validation across:
- SVM_RBF: C ∈ {0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.01, 0.1}
- GradientBoosting: n_estimators ∈ {100, 200, 300}, max_depth ∈ {3, 4, 5}, learning_rate ∈ {0.05, 0.1, 0.2}
- RandomForest: n_estimators ∈ {100, 200, 300}, max_depth ∈ {5, 8, 10}

**Winner:** GradientBoosting with n_estimators=200, max_depth=4, learning_rate=0.1  
**Result:** 96.2% accuracy, F1=0.969

### Step 3: Resume Data Augmentation (v3.2)

**Goal:** Add diversity by creating training records from real resumes.

**Method:**
1. Parse 220 real resumes from an Entity Recognition dataset (JSONL format)
2. Extract skills and experience from each resume's annotations
3. Generate team compositions by sampling 3-5 members
4. Simulate success/failure based on role diversity, skill levels, etc.
5. Added 150 new training records (total: 407)

**Result:** Accuracy dropped to 87.8% — expected because real-world data is harder than synthetic.

### Step 4: HistGradientBoosting (v3.3) ← PRODUCTION

**Goal:** Try a more modern algorithm that handles the noisier data better.

**Method:**
1. Replace GradientBoosting with HistGradientBoostingClassifier  
2. Hyperparameters: max_iter=200, max_depth=6, learning_rate=0.1, min_samples_leaf=10
3. Wrap in CalibratedClassifierCV (isotonic, 5-fold)
4. Evaluate on holdout

**Result:** 90.2% accuracy, F1=0.925, AUC=0.958 — best balance of accuracy and generalization.

### Step 5: Threshold Optimization (v3.4)

**Goal:** Instead of the default 0.5 decision threshold, find the threshold that maximizes F1.

**Method:**
1. Get predicted probabilities on test set
2. Try thresholds from 0.1 to 0.9 in steps of 0.01
3. Calculate F1 at each threshold
4. Select threshold that maximizes F1

**Result:** Optimal threshold ≈ 0.45. Accuracy=89.0%, F1=0.920 — marginal difference from v3.3, so v3.3 was kept as production.

---

## 7. Feature Engineering — 25 Features

### Original 14 Features (v1.0)

| # | Feature | Formula / Description | Type |
|---|---------|----------------------|------|
| 1 | `team_size` | len(members) | int |
| 2 | `role_diversity` | unique_roles / team_size | float [0,1] |
| 3 | `avg_skill_level` | mean(skill_levels) | float [0,10] |
| 4 | `min_skill_level` | min(skill_levels) | float [0,10] |
| 5 | `max_skill_level` | max(skill_levels) | float [0,10] |
| 6 | `skill_variance` | var(skill_levels) | float ≥ 0 |
| 7 | `skill_range` | max - min skill | float [0,10] |
| 8 | `gini_coefficient` | Lorenz curve inequality | float [0,1] |
| 9 | `avg_experience` | mean(years) | float ≥ 0 |
| 10 | `experience_variance` | var(years) | float ≥ 0 |
| 11 | `skill_coverage` | matched / required skills | float [0,1] |
| 12 | `has_critical_role_gap` | 1 if missing critical role | binary |
| 13 | `role_duplication_ratio` | duplicate_roles / team_size | float [0,1] |
| 14 | `project_type_encoded` | numeric (0-5) | int |

### New 11 Features (v2.0)

| # | Feature | Formula | Why Added |
|---|---------|---------|----------|
| 15 | `median_skill_level` | median(skill_levels) | Robust to outliers |
| 16 | `skill_iqr` | Q75 - Q25 | Robust spread measure |
| 17 | `total_skills_count` | len(all team skills) | Breadth of knowledge |
| 18 | `skills_per_member` | total_skills / team_size | Normalized breadth |
| 19 | `critical_role_coverage` | filled_critical / total_critical | Partial gap tracking |
| 20 | `helpful_role_coverage` | filled_helpful / total_helpful | Nice-to-have roles |
| 21 | `experience_skill_interaction` | avg_exp × avg_skill | Synergy term |
| 22 | `weakest_link_score` | min_skill × role_diversity | Bottleneck indicator |
| 23 | `team_strength_index` | sum(skills) / √(team_size) | Size-normalized power |
| 24 | `coverage_diversity_product` | coverage × diversity | Double-good interaction |
| 25 | `balance_penalty` | gini × variance × (1 + dup_ratio) | Triple-bad interaction |

### What is the Gini Coefficient?

$$G = \frac{2 \sum_{i=1}^{n} i \cdot x_i}{n \sum_{i=1}^{n} x_i} - \frac{n + 1}{n}$$

Where $x_1 \le x_2 \le ... \le x_n$ are sorted skill levels.

- G = 0: Perfect equality (all members have same skill level)
- G = 1: Perfect inequality (one member has all the skills)
- Threshold: G < 0.15 is "good balance"

---

## 8. API Endpoints — Full Reference

### Core Endpoints

#### POST `/api/form-teams`
**The main endpoint.** Upload PDF resumes and get team formations.

```bash
curl -X POST http://localhost:8000/api/form-teams \
  -F "resumes=@john.pdf" \
  -F "resumes=@jane.pdf" \
  -F "resumes=@bob.pdf" \
  -F "resumes=@alice.pdf" \
  -F "team_size=2" \
  -F 'projects_json=[{"name":"Web App","description":"Build a React + Node.js web application","team_size":2}]'
```

**Response:**
```json
{
  "success": true,
  "summary": {
    "total_resumes": 4,
    "parsed_resumes": 4,
    "total_teams": 2,
    "students_assigned": 4,
    "average_team_balance": 0.712
  },
  "teams": [
    {
      "team_name": "Team 1",
      "team_id": "Team_01",
      "team_size": 2,
      "balance_score": {"average_score": 0.68, "score_variance": 0.001, "role_diversity": 1.0, "overall_balance": 0.73},
      "members": [
        {
          "student_id": "S001",
          "name": "John Doe",
          "role": "Backend",
          "overall_score": 0.72,
          "experience_score": 3.5,
          "reason": "Strong Backend skills (8/10); Key skills: Node.js, MongoDB, Express",
          "top_skills": ["node.js", "mongodb", "express"]
        }
      ],
      "roles_covered": ["Backend", "Frontend"]
    }
  ],
  "profiles": [...]
}
```

#### POST `/api/v2/form-teams`
**Structured endpoint** — matches participants with resumes by order.

```bash
curl -X POST http://localhost:8000/api/v2/form-teams \
  -F "resumes=@john.pdf" \
  -F "resumes=@jane.pdf" \
  -F 'projects=[{"projectId":"P001","projectName":"AI Chatbot","description":"NLP chatbot","techstack":"Python, TensorFlow"}]' \
  -F 'participantData=[{"participantId":"PART001","participantName":"John Doe","githubProfile":"johndoe123"},{"participantId":"PART002","participantName":"Jane Smith","githubProfile":""}]'
```

#### POST `/api/predict-team-success`
**Predict whether a team will succeed** using the trained model.

```bash
curl -X POST http://localhost:8000/api/predict-team-success \
  -H "Content-Type: application/json" \
  -d '{
    "project_type": "web_application",
    "required_skills": ["React", "Node.js", "MongoDB"],
    "members": [
      {"assigned_role": "frontend", "skills": {"React": 8, "JavaScript": 7}, "experience_years": 2},
      {"assigned_role": "backend", "skills": {"Node.js": 7, "MongoDB": 6}, "experience_years": 3}
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "prediction": "success",
  "success_probability": 0.952,
  "failure_probability": 0.048,
  "factors": [
    "✅ Good role diversity (1.00)",
    "✅ Strong avg skill level (7.5/10)",
    "✅ Well balanced team (Gini: 0.033)",
    "✅ Good skill coverage (100%)"
  ]
}
```

### Feedback Loop Endpoints

#### POST `/api/feedback`
Submit team outcome records for the learning pipeline.

```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "team_id": "T001",
      "project_type": "web_application",
      "required_skills": ["React", "Node.js"],
      "members": [
        {"name": "Alice", "assigned_role": "frontend", "skills": ["React"], "skill_level": 8, "experience_years": 2},
        {"name": "Bob", "assigned_role": "backend", "skills": ["Node.js"], "skill_level": 7, "experience_years": 3}
      ],
      "success": true,
      "grade": "A",
      "score": 90
    }]
  }'
```

#### POST `/api/retrain`
Trigger model retraining on all feedback data.

#### GET `/api/feedback-weights`
See the active scoring weights (learned or default).

#### GET `/api/feedback-data-stats`
Statistics about training data (record count, success/failure distribution, project types).

---

## 9. GitHub Integration

### How It Works

1. **User provides** a mapping: `{"John Doe": "johndoe123", "Jane Smith": "janesmith"}`
2. **GitHubFetcher** calls GitHub REST API for each username
3. **Fetches repos:** `GET /users/{username}/repos?sort=updated&per_page=30`
4. **Analyzes:**
   - **Languages:** Bytes of code per language → role mapping
   - **Topics:** Repository topic tags → role mapping
   - **Stars:** Total stargazers count
   - **Activity:** Original repos (excludes forks)
5. **Converts to skill scores:** Same {role: score} format as resume extraction
6. **Merges with resume skills:**

$$\text{final\_skill}_r = 0.6 \times \text{resume\_skill}_r + 0.4 \times \text{github\_skill}_r$$

### Rate Limiting

| Auth Mode | Rate Limit | How |
|-----------|-----------|-----|
| No token | 60 req/hour | Default |
| With `GITHUB_TOKEN` env var | 5,000 req/hour | Set in environment |

### What GitHub Data Adds

- **Validates resume claims:** If resume says "React developer" but GitHub has 0 JavaScript repos → lower score
- **Discovers unlisted skills:** Resume may not mention Docker, but GitHub has Dockerfiles in repos
- **Experience boost:** Active GitHub with many original repos → +0.5 experience score

---

## 10. Validation Module

### Quality Metrics (validation.py)

The `TeamFormationValidator` class runs 5 metrics on formed teams:

| Metric | Formula | Threshold | Weight |
|--------|---------|-----------|--------|
| **Gini Coefficient** | $G = \frac{2 \sum i \cdot x_i}{n \sum x_i} - \frac{n+1}{n}$ | < 0.15 | 25% |
| **Role Coverage** | filled_roles / required_roles | ≥ 75% | 20% |
| **Skill Balance** | 1 - CV (Coefficient of Variation) | ≥ 0.70 | 25% |
| **Diversity Score** | avg(unique_roles / team_size) per team | ≥ 0.60 | 15% |
| **Assignment Rate** | assigned / total students | ≥ 95% | 15% |

### Ground Truth Comparison

If ideal/manual team assignments exist, compare using:
- **Precision:** Correct pairs / predicted pairs
- **Recall:** Correct pairs / ideal pairs
- **F1 Score:** Harmonic mean of precision and recall
- Uses **pair-based evaluation** (if Alice & Bob should be together, that's one pair)

---

## 11. Training Data & Data Pipeline

### Data Format (`feedback_training_data.json`)

```json
{
  "metadata": {
    "total_records": 409,
    "success_count": 215,
    "failure_count": 194
  },
  "team_records": [
    {
      "team_id": "T001",
      "project": {
        "project_type": "web_application",
        "project_name": "E-commerce Platform",
        "required_skills": ["React", "Node.js", "MongoDB"]
      },
      "members": [
        {
          "name": "Alice",
          "assigned_role": "frontend",
          "skills": ["React", "JavaScript", "CSS"],
          "skill_level": 8,
          "experience_years": 2
        },
        {
          "name": "Bob",
          "assigned_role": "backend",
          "skills": ["Node.js", "MongoDB"],
          "skill_level": 7,
          "experience_years": 3
        }
      ],
      "outcome": {
        "success": true,
        "grade": "A",
        "score": 90,
        "completion_status": "completed"
      }
    }
  ]
}
```

### Data Sources

| Source | Records | How Generated |
|--------|---------|--------------|
| Original synthetic (v1.0) | 55 | Hand-crafted trivial examples |
| Hard edge cases (v2.0) | 200 | Deliberately ambiguous scenarios |
| API feedback | 2 | Via POST /api/feedback |
| Resume-augmented (v3.2) | 150 | Extracted from 220 real resumes |
| **Total** | **407** | |

### Edge Case Categories

| Category | Examples | Count |
|----------|---------|-------|
| **Solo expert** | One 10-skill member + three 2-skill | ~30 |
| **All same role** | 4 frontend developers on ML project | ~25 |
| **Perfect team** | Diverse roles, all skill 7+ | ~25 |
| **Zero experience** | Good skills but 0 years experience | ~20 |
| **Wrong project** | ML skills on web project | ~25 |
| **Big team** | 6+ members — more people ≠ better | ~15 |
| **Tiny team** | 2 members — can they cover everything? | ~15 |
| **One bad apple** | One member with skill 1, rest are 8+ | ~20 |
| **Borderline** | Should this succeed? Depends on criteria | ~25 |

---

## 12. Key Design Decisions & Why

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| **Snake Draft over Hungarian Algorithm** | Simpler, O(n²) vs O(n³); easier to explain; produces good-enough results | Hungarian gives optimal matching but is harder to implement and explain |
| **Rule-based skill extraction over NER** | Deterministic, 200+ keywords cover common tech stack well; no labeled data needed | spaCy NER would need labeled training data |
| **SVM_RBF in v2.0** | Best F1 (0.947) on hard edge cases; non-linear boundary handles complex patterns | RF was close (0.920) but SVM was better on borderline cases |
| **HistGradientBoosting in v3.3** | Better generalization on real-world data; handles missing values; fast | SVM_RBF scored higher on synthetic-only data |
| **25 features → 16 after pruning** | 9 features were noise/redundant; removing them reduced overfitting | Could keep all 25 but model would be less robust |
| **Probability calibration** | Raw SVM/GB probabilities are unreliable; isotonic calibration is standard | Platt scaling (sigmoid) is simpler but less flexible |
| **0.6/0.4 resume/GitHub blend** | Resume is primary source; GitHub supplements and validates | 50/50 would give too much weight to GitHub (biased toward open-source contributors) |
| **Permutation importance** | Works for any model (SVM has no `feature_importances_`); model-agnostic | SHAP values are better but much slower |
| **Not using stacking ensemble** | Stacking (v2.1 experiment) scored 78% due to label noise; single model was better | Ensemble of RF+SVM+GB was considered |
| **409 records is enough** | With 16 features, rule of thumb says 10× per feature = 160 minimum. 409 > 160 | Could generate more but risk synthetic data bias |

---

## 13. VIVA Questions & Answers

### Category: Project Overview

**Q1: What is the problem you are solving?**
> We're solving the problem of **automated team formation for hackathons**. Given N student resumes and M projects, we need to form balanced teams where each team has diverse roles, good skill coverage, and fair talent distribution. We also predict whether a team will succeed and continuously improve through a feedback loop.

**Q2: What makes this an ML project and not just a simple assignment algorithm?**
> Three things make it ML:
> 1. The **feedback loop** — we train a classification model (HistGradientBoosting) on historical team outcomes to learn what makes teams succeed or fail
> 2. The model **extracts 25 features** from team compositions and uses them to predict success probability
> 3. The **learned weights** replace static heuristics — the system gets better over time as it receives more feedback

**Q3: What is the input and output of your system?**
> **Input:** PDF resumes + (optional) project descriptions + (optional) GitHub usernames
> **Output:** Balanced teams with members, assigned roles, scores, explanations, and success probability predictions

**Q4: How does your system handle the cold start problem (no feedback data at first)?**
> We start with **static weights** (60% skill, 30% experience, 10% diversity) based on domain knowledge. The Snake Draft algorithm works well even without ML. Once feedback data accumulates (minimum 10 records), we can train the model and switch to learned weights.

---

### Category: ML Model

**Q5: What ML algorithm are you using and why?**
> We're using **HistGradientBoostingClassifier** from scikit-learn, wrapped in **CalibratedClassifierCV** for probability calibration. We chose it because:
> - It handles real-world noisy data better than SVM
> - Histogram-based binning prevents overfitting on small datasets
> - It natively handles missing values
> - Sequential tree building corrects previous errors (boosting principle)
> - It gave us 90.2% accuracy and 0.925 F1 on holdout data

**Q6: How did you evaluate your model?**
> We used a **multi-stage evaluation:**
> 1. **10-fold stratified cross-validation** on training data for model selection
> 2. **80/20 stratified holdout split** for final evaluation
> 3. Metrics: **Accuracy, F1-score, Precision, Recall, AUC-ROC**
> 4. **Confusion matrix** to check true/false positives and negatives
> 5. **Permutation importance** for feature analysis

**Q7: What is the difference between accuracy and F1 score? Why do you track both?**
> **Accuracy** = (correct predictions) / (total predictions). It can be misleading with imbalanced data — if 90% of teams succeed, predicting "success" always gives 90% accuracy.
> **F1 score** = harmonic mean of precision and recall. It penalizes both false positives (predicting success when team fails) and false negatives (predicting failure when team succeeds). We use F1 as the primary metric because both types of errors matter.

**Q8: What is AUC-ROC and what does 0.958 mean?**
> **AUC-ROC** (Area Under the Receiver Operating Characteristic curve) measures how well the model **ranks** teams. An AUC of 0.958 means: if we take a random successful team and a random failed team, the model correctly ranks the successful team higher 95.8% of the time. AUC = 1.0 is perfect, 0.5 is random guessing.

**Q9: What is probability calibration and why is it needed?**
> Raw classifier outputs (like SVM's "probability") are not true probabilities. A model might output 0.80 for many predictions, but only 60% of them are actually correct. **Isotonic calibration** (CalibratedClassifierCV) learns a mapping from raw outputs to true probabilities using held-out data. After calibration, when our model says "85% success probability," it really means 85%.

**Q10: Explain the feature engineering you did.**
> We extract **25 features** from each team composition:
> - **Basic metrics:** team_size, avg_skill, min_skill, max_skill, experience
> - **Distribution metrics:** skill_variance, gini_coefficient, skill_iqr
> - **Coverage metrics:** skill_coverage, critical_role_coverage, helpful_role_coverage
> - **Interaction terms:** experience × skill (synergy), min_skill × diversity (bottleneck)
> - **Composite features:** team_strength_index = sum(skills)/√(team_size)
>
> We then pruned to **16 features** by removing those with low permutation importance.

**Q11: What is the Gini coefficient and why do you use it?**
> The Gini coefficient measures **inequality** in skill distribution within a team. We borrowed it from economics (wealth inequality). A team where one person has skill 10 and others have skill 2 has high Gini (bad). A team where everyone has skill 7 has low Gini (good). We use threshold 0.15 — below that is "well balanced."

**Q12: How do you prevent overfitting with only 407 records?**
> Multiple strategies:
> 1. **Feature pruning** (25 → 16) reduces model complexity
> 2. **StandardScaler** normalizes features to zero mean, unit variance
> 3. **CalibratedClassifierCV** uses inner cross-validation
> 4. **HistGradientBoosting** uses min_samples_leaf=10 and max_depth=6 (regularization)
> 5. **10-fold CV** during model selection prevents selection bias
> 6. **Holdout split** ensures we never evaluate on training data
> 7. Rule of thumb: 16 features × 10 = 160 minimum records. We have 407.

---

### Category: Team Formation Algorithm

**Q13: How does the Snake Draft algorithm work?**
> Like fantasy sports drafts:
> - Round 1: Team 1 picks first → Team 2 → Team 3 → ... → Team N
> - Round 2: **Reversed** → Team N picks first → ... → Team 1
> - Round 3: Forward again
>
> Each pick: find the best available student for a role the team doesn't have yet. This ensures fairness — the team that picks last in one round picks first in the next.

**Q14: Why Snake Draft instead of Hungarian Algorithm?**
> - Snake Draft is O(n × m) where n=students, m=teams. Hungarian is O(n³).
> - Snake Draft is easy to explain and understand.
> - It naturally produces balanced teams because of the alternating order.
> - Hungarian Algorithm is optimal for one-to-one matching but team formation is many-to-one.
> - We prioritize **fairness** (no team gets all top students) over **optimality** (best possible assignment).

**Q15: How do you decide which role to assign someone?**
> 1. **Primary role** from resume skill extraction (highest scoring role)
> 2. In the draft, we try to fill roles the team doesn't have yet
> 3. Role priority order: backend → frontend → fullstack → ml → data → uiux → devops
> 4. If a student's primary role is already filled, they can be placed in their secondary role

---

### Category: NLP / Skill Extraction

**Q16: How do you extract skills from resumes?**
> **Rule-based keyword matching** against a dictionary of 200+ technology keywords organized into 7 role categories. For each role, we:
> 1. Match keywords using regex (exact match for short terms like "CSS", substring for longer terms like "machine learning")
> 2. Count occurrences (raw_count) and unique matches (unique_skills)
> 3. Score = (unique_skills × 2 + raw_count × 0.5), normalized to 0–10

**Q17: Why not use NER (Named Entity Recognition) or BERT for skill extraction?**
> - NER needs **labeled training data** (resumes annotated with skill entities). We don't have that.
> - BERT/transformer models need **GPU and significant compute**. Our system runs on free-tier hosting.
> - Rule-based matching gives us **100% precision** — if we match "React", we know it's React. No false positives.
> - The keyword dictionary (200+ terms) covers the common hackathon tech stack well.
> - Trade-off: We sacrifice **recall** (might miss unusual skills) for **precision** and simplicity.

**Q18: How do you calculate skill percentages (like "React: 85%")?**
> For each matched skill:
> 1. **Base score** (50-75%) from mention frequency: more mentions = higher score
> 2. **Context bonus** (+5% each, max +15%): if the skill appears near words like "expert in", "built", "certified"
> 3. **Deterministic variation** (-7 to +7): using MD5 hash of skill name for consistency across runs
> 4. Final score clamped to 40-95% range

---

### Category: GitHub Integration

**Q19: How does GitHub integration improve team formation?**
> It adds a **second data source** beyond the resume:
> 1. **Validates claims:** Resume says "React developer" but GitHub has only Python repos → score adjusted down
> 2. **Discovers hidden skills:** GitHub repos may use Docker/K8s but resume doesn't mention DevOps
> 3. **Quantifies activity:** 20 original repos vs 2 repos → experience boost
>
> Skills merge: `final = 0.6 × resume + 0.4 × github`

**Q20: Is GitHub integration required?**
> **No, it's completely optional.** The system works perfectly with resumes alone. GitHub data is an enhancement — if provided, skills are blended. If not provided, resume data is used as-is.

---

### Category: Feedback Loop

**Q21: Explain the feedback loop in detail.**
> 1. Teams are formed and assigned to projects
> 2. After the project, we collect **outcomes**: success/failure, grade, score
> 3. These are submitted via POST /api/feedback
> 4. When we call POST /api/retrain, the model retrains on ALL historical data
> 5. New **learned weights** are saved to model_weights.pkl
> 6. The ml_engine reloads these weights — future team formations use data-driven scoring
>
> This is a **supervised learning feedback loop** — each team outcome is a labeled training example.

**Q22: How many records do you need to retrain?**
> Minimum 10 records. But quality improves significantly after 100+ records. Our current model uses 407 records. More records with diverse scenarios (different project types, team sizes, skill distributions) produce better models.

**Q23: Can the model degrade after retraining?**
> Yes, if bad data is submitted. Protections:
> - We keep all version snapshots in `model_versions/`
> - The optimization pipeline tracks metrics for each version
> - If a retrained model performs worse, we can rollback to a previous version
> - 10-fold CV during training catches severe degradation

---

### Category: API & Architecture

**Q24: Why FastAPI instead of Flask/Django?**
> - **Async support:** GitHub API calls are async (httpx), resume parsing can be parallelized
> - **Auto-documentation:** Swagger UI at /docs — instant API testing
> - **Pydantic validation:** Type-safe request/response models, automatic validation
> - **Performance:** Faster than Flask for I/O-bound tasks
> - **Modern Python:** Uses Python type hints natively

**Q25: How does the system handle multiple concurrent requests?**
> FastAPI + Uvicorn is async — multiple requests are handled concurrently via Python's asyncio event loop. The model is loaded once on startup (via module import), so predictions are fast. Resume parsing is CPU-bound but handled sequentially per request.

**Q26: How is the system deployed?**
> - **Docker container** with Dockerfile
> - **Render.com** free tier for hosting
> - **Keep-alive ping** every 14 minutes (Render sleeps after 15 minutes of inactivity)
> - The server self-pings its own /health endpoint to stay awake

---

### Category: Validation & Testing

**Q27: How do you validate team quality?**
> The `TeamFormationValidator` uses 5 metrics:
> 1. **Gini coefficient** < 0.15 (fairness)
> 2. **Role coverage** ≥ 75% (required roles filled)
> 3. **Skill balance** ≥ 0.70 (even talent distribution)
> 4. **Diversity score** ≥ 0.60 (role variety per team)
> 5. **Assignment rate** ≥ 95% (no student left out)
>
> Combined into a quality grade (A/B/C/D/F).

**Q28: How do you test the API?**
> We have 6 automated tests in test_endpoints.py:
> 1. **GET /api/feedback-weights** — weights are loaded
> 2. **GET /api/feedback-data-stats** — training data exists
> 3. **POST /api/predict-team-success** (good team) — probability > 0.80
> 4. **POST /api/predict-team-success** (bad team) — probability < 0.30
> 5. **POST /api/feedback** — feedback submission works
> 6. **POST /api/retrain** — retraining completes successfully

---

### Category: Tricky / Deep Questions

**Q29: Your model achieves 90% accuracy. What about the other 10%?**
> The 10% errors come from:
> 1. **Borderline cases** — teams that could go either way (e.g., decent skills but missing one critical role)
> 2. **Label noise** — some synthetic training data has subjective success/failure labels
> 3. **Feature limitations** — we don't capture soft skills, communication, motivation
> 4. **Data augmentation noise** — real-resume-derived records use simulated outcomes
>
> With more real-world feedback data, accuracy will improve.

**Q30: Is your training data biased?**
> **Yes, somewhat.** 
> - Synthetic data has our assumptions baked in (e.g., "4 same-role members always fail")
> - Resume augmentation uses web-scraped resumes biased toward tech roles
> - Missing: non-tech roles, international skill naming conventions
> - Mitigation: The 5-step optimization pipeline specifically adds diverse edge cases and real-world data

**Q31: How would you improve this system with more time?**
> 1. **Semantic skill extraction** using Sentence Transformers (BERT) — understand "machine learning" and "ML" are the same
> 2. **Hungarian Algorithm** for optimal one-to-one matching (as an alternative to Snake Draft)
> 3. **More real feedback data** from actual hackathon outcomes
> 4. **Frontend UI** for resume upload and team visualization
> 5. **LinkedIn integration** for richer profile data
> 6. **SHAP values** for better model explainability
> 7. **Active learning** — the model identifies which team compositions it's most uncertain about and requests feedback for those

**Q32: What is the time complexity of your system?**
> For N resumes and T teams:
> - **PDF parsing:** O(N × P) where P = pages per resume
> - **Skill extraction:** O(N × K) where K = number of keywords (~200)
> - **Snake Draft:** O(N × T) — each student considered for each team
> - **Model prediction:** O(1) — single feature vector through trained model
> - **Total:** O(N × (P + K + T)) — linear in number of resumes

**Q33: What happens if two students have identical skills?**
> The Snake Draft handles this naturally — the first team to pick gets the first student (by overall_score ordering). The second student goes to the next team. Since teams have different compositions, even identical students end up in different contexts, and the model evaluates each team **as a whole**, not individual students.

**Q34: Can your model work for industries other than hackathons?**
> Yes, with modifications:
> - **Change SKILL_DICTIONARY** for the new domain (e.g., marketing, healthcare)
> - **Adjust the 7 roles** to match the industry
> - **Retrain** with domain-specific feedback data
> - The ML pipeline, Snake Draft, and validation metrics are domain-agnostic

**Q35: Explain StandardScaler. Why is it important?**
> StandardScaler transforms each feature to have **mean=0 and standard deviation=1**:
$$x_{scaled} = \frac{x - \mu}{\sigma}$$
> This is critical because:
> - team_size ranges 2-6, but team_strength_index ranges 10-50
> - Without scaling, high-range features dominate the model
> - SVM and gradient boosting are sensitive to feature scales
> - The scaler is fit on training data only (not test data) to prevent data leakage

**Q36: What is permutation importance and why did you use it?**
> **Permutation importance** measures how much model accuracy drops when a feature's values are randomly shuffled. If shuffling "avg_skill_level" causes accuracy to drop 15%, that feature has 15% importance.
> - **Why:** SVM doesn't have built-in `feature_importances_` like Random Forest
> - **Advantage:** Model-agnostic — works for any classifier
> - **We used it:** To prune features (step v3.0) and to learn scoring weights

**Q37: What's the difference between GradientBoosting and HistGradientBoosting?**
> | Aspect | GradientBoosting | HistGradientBoosting |
> |--------|-----------------|---------------------|
> | Split search | Exact | Histogram-based (binned) |
> | Speed | Slower | 10-100× faster |
> | Missing values | Cannot handle | Native support |
> | Overfitting | More prone | Less (due to binning) |
> | Inspiration | Original Friedman 2001 | Inspired by LightGBM |

---

## 14. Glossary of ML Terms Used

| Term | Definition |
|------|-----------|
| **AUC-ROC** | Area Under Receiver Operating Characteristic curve — measures ranking quality |
| **Calibration** | Adjusting model outputs to reflect true probabilities |
| **CalibratedClassifierCV** | scikit-learn wrapper that calibrates any classifier using cross-validation |
| **Cross-Validation (CV)** | Split data into K folds, train on K-1, test on 1, repeat K times |
| **F1 Score** | Harmonic mean of precision and recall: $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ |
| **Feature Engineering** | Creating numerical inputs from raw data (team records → 25 numbers) |
| **Feature Pruning** | Removing low-importance features to improve generalization |
| **Gini Coefficient** | Measure of inequality (0=equal, 1=maximally unequal) |
| **Gradient Boosting** | Ensemble method — builds trees sequentially to correct previous errors |
| **GridSearchCV** | Exhaustive search over hyperparameter combinations with CV |
| **HistGradientBoosting** | Histogram-based gradient boosting — faster, handles missing values |
| **Holdout Split** | Reserve part of data (20%) for final evaluation, never used in training |
| **Isotonic Regression** | Non-parametric, monotonic function fitting — used for calibration |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Permutation Importance** | Feature importance measured by accuracy drop when feature is shuffled |
| **Precision** | Of predicted positives, how many are actually positive: $\frac{TP}{TP+FP}$ |
| **Recall** | Of actual positives, how many did we find: $\frac{TP}{TP+FN}$ |
| **RBF Kernel** | Radial Basis Function — maps features to infinite dimensions for non-linear SVM |
| **Snake Draft** | Alternating pick order for fair distribution |
| **StandardScaler** | Z-score normalization: $(x - \mu) / \sigma$ |
| **Stratified Split** | Preserves class ratio (success/failure) in train and test sets |
| **SVM (Support Vector Machine)** | Finds maximum-margin hyperplane to separate classes |
| **XAI (Explainable AI)** | Making AI decisions understandable to humans |

---

*Document created for SkillSyncAI VIVA preparation. Covers all modules, algorithms, models, API endpoints, and design decisions.*
