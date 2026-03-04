# SkillSyncAI

### AI-Powered Hackathon Team Formation with ML Feedback Loop

> Upload resumes → AI extracts skills → ML forms balanced teams → Learns from outcomes

---

## Overview

SkillSyncAI automates hackathon team formation. It reads PDF resumes, extracts skills via NLP, scores candidates against project roles, and uses a trained ML model to optimize team composition. A feedback loop lets the system learn from real team outcomes and improve over time.

---

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│  PDF Resumes │───>│ NLP Pipeline │───>│ Skill Scorer │───>│ Team Former │
└─────────────┘    │  (pdfplumber) │    │ (ml_engine)  │    │(Snake Draft)│
                   └──────────────┘    └──────┬───────┘    └──────┬──────┘
                                              │                   │
                                     ┌────────▼───────────────────▼──────┐
                                     │   ML Feedback Loop (v3.3)          │
                                     │   - 16-feature extraction (pruned) │
                                     │   - HistGradientBoosting + Calib   │
                                     │   - Real resume augmented data     │
                                     │   - Learned scoring weights        │
                                     └──────────────────────────────────┘
```

---

## ML Pipeline (v3.3 — Final)

### Techniques Used

| Technique | What It Is | Where We Use It |
|-----------|-----------|-----------------|
| **Supervised Classification** | Labeled data (success/failure) trains a classifier | Core training pipeline — maps team features → outcome |
| **Feature Engineering** | Creating informative inputs from raw data | 25 features extracted → pruned to 16 via permutation importance |
| **Data Augmentation** | Generating synthetic hard examples | 200 synthetic edge cases + 150 teams from 220 real resumes |
| **Model Selection** | Comparing multiple algorithms, picking the best | SVM vs RF vs GradientBoosting vs HistGradientBoosting — HistGB won on augmented data |
| **Feature Pruning** | Removing low-signal features via permutation importance | 25→16 features: dropped 9 near-zero-importance features for better generalization |
| **Hyperparameter Search** | Grid search over algorithm parameters | Searched C/gamma (SVM), n_estimators/depth (RF/GB) — found optimal configs |
| **Cross-Validation** | 10-fold stratified CV for reliable estimates | Used during model comparison to avoid overfitting |
| **Holdout Evaluation** | 80/20 train/test split for unbiased final metrics | Final performance measured on unseen 20% test set |
| **Probability Calibration** | Making `predict_proba()` outputs reliable | `CalibratedClassifierCV` with isotonic regression wraps the final SVM |
| **Permutation Importance** | Measuring feature impact by shuffling values | Used for SVM (which has no native `feature_importances_`) |
| **StandardScaler** | Zero-mean, unit-variance normalization | Applied before SVM training (SVM is distance-based, needs scaling) |

### What We Did NOT Use (and Why)

| Often Confused With | Why It Doesn't Apply |
|---------------------|---------------------|
| **Reinforcement Learning** | RL needs an agent taking sequential actions in an environment with rewards. We have labeled data → standard supervised learning. |
| **Bagging** | Bagging trains the *same* model on bootstrap samples and averages. We compared *different* algorithms and picked one winner. |
| **Stacking/Blending** | We tried a stacking ensemble in Round 2 — it performed worse (78%) due to noisy labels. Single calibrated SVM was better. |
| **Deep Learning** | 256 training records is far too small. SVM with RBF kernel handles this scale perfectly. |

### The 25 Features

Extracted from each team composition by `TeamFeatureExtractor`:

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `team_size` | Structure | Number of members |
| 2 | `role_diversity` | Structure | Unique roles / team size |
| 3 | `avg_skill_level` | Skill | Mean skill level (0-10) |
| 4 | `min_skill_level` | Skill | Weakest member's skill |
| 5 | `max_skill_level` | Skill | Strongest member's skill |
| 6 | `skill_variance` | Balance | How spread out skills are |
| 7 | `skill_range` | Balance | Max - Min skill |
| 8 | `gini_coefficient` | Balance | Inequality metric (0=equal, 1=unequal) |
| 9 | `avg_experience` | Experience | Mean years of experience |
| 10 | `experience_variance` | Experience | Spread in experience levels |
| 11 | `skill_coverage` | Coverage | % of required skills covered |
| 12 | `has_critical_role_gap` | Coverage | Missing a must-have role (0/1) |
| 13 | `role_duplication_ratio` | Structure | % of duplicate roles |
| 14 | `project_type_encoded` | Context | Numeric encoding of project type |
| 15 | `median_skill_level` | Skill | Median skill (robust to outliers) |
| 16 | `skill_iqr` | Balance | Interquartile range of skills |
| 17 | `total_skills_count` | Composite | Total distinct skills across team |
| 18 | `skills_per_member` | Composite | Avg distinct skills per person |
| 19 | `critical_role_coverage` | Coverage | % of critical roles filled |
| 20 | `helpful_role_coverage` | Coverage | % of helpful roles filled |
| 21 | `experience_skill_interaction` | Interaction | avg_experience × avg_skill |
| 22 | `weakest_link_score` | Composite | min(skill) × role_diversity |
| 23 | `team_strength_index` | Composite | avg_skill × team_size × coverage |
| 24 | `coverage_diversity_product` | Interaction | coverage × diversity |
| 25 | `balance_penalty` | Balance | variance × gini (high = bad) |

### Training Rounds

| Round | Records | Best Model | Holdout Acc | F1 | AUC | Outcome |
|-------|---------|-----------|-------------|------|------|---------|
| v1.0 (baseline) | 55 | RandomForest | 100% (CV) | 1.0 | — | Too clean data, overfitting |
| **v2.0 (final)** | **256** | **SVM_RBF (calibrated)** | **94.2%** | **0.947** | **0.996** | **Production model** |
| v2.1 (experiment) | 406 | Stacking Ensemble | 78% | — | — | Discarded (label noise) |
| v2.2 (experiment) | 356 | Voting Ensemble | 93.1% | — | 0.962 | Discarded (v2.0 was better) |

### Optimization Pipeline (v3.x)

5-step optimization pipeline with version tracking (`optimization_trainer.py`):

| Version | Step | Description | Holdout Acc | F1 | AUC | Records | Outcome |
|---------|------|-------------|-------------|------|------|---------|---------|
| v2.0 | Baseline | SVM_RBF (calibrated, 25 features) | 96.2% | 0.966 | 0.993 | 257 | High accuracy on synthetic data |
| v3.0 | Feature Pruning | Dropped 9 low-importance features → 16 | 90.4% | 0.915 | 0.943 | 257 | Identified redundant features |
| v3.1 | Hyperparameter Search | Grid search: GB won (lr=0.05, depth=3, n=200) | 96.2% | 0.963 | 0.991 | 257 | Matched baseline, validated configs |
| v3.2 | Resume Augmentation | +150 teams from 220 real resumes (JSONL) | 87.8% | 0.907 | 0.918 | 407 | Harder, more realistic test set |
| **v3.3** | **HistGradientBoosting** | **Best of SVM/RF/GB/HistGB on real+synthetic data** | **90.2%** | **0.925** | **0.958** | **407** | **Production — best generalization** |
| v3.4 | Threshold Optimization | Optimal threshold=0.28 on v2.0 SVM | 89.0% | 0.920 | 0.933 | 407 | Marginal gain, not worth complexity |

**Key Learnings:**
- v2.0's 96.2% was on synthetic-only data — high accuracy but limited generalization
- Feature pruning (25→16) removed noise features, improving robustness on real data
- Real resume augmentation (+150 records from 220 resumes) created a harder, more realistic benchmark
- **v3.3 HistGradientBoosting chosen for production** — 90.2% on mixed real+synthetic data generalizes better to unseen teams
- All v3.x model versions saved in `api/model_versions/` for reproducibility and rollback

**Why v3.3 over v2.0?**
> 90.2% on a realistic test set > 96.2% on a synthetic-only test set. v3.3 has seen real-world skill distributions from 220 parsed resumes, uses a modern tree-based algorithm that handles feature noise naturally, and uses 16 pruned features (Occam's razor). It will perform better on genuinely new teams.

### Final Model Details (v3.3)

```
Model:       CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=200, max_depth=6, lr=0.1))
Features:    16 (pruned from 25 via permutation importance)
Data:        407 records (257 synthetic + 150 real-resume-based)
Train/Test:  325 / 82 (80/20 stratified split)
Calibration: Isotonic regression (5-fold)
Algorithm:   HistGradientBoosting — bins features, handles noise, no scaling needed
```

### Learned Weights (fed back into team scoring)

```
weight_skill:       0.192    (skill levels matter, but not dominant)
weight_experience:  0.246    (experience is the second-biggest factor)
weight_coverage:    0.408    (covering required skills matters MOST)
weight_balance:     0.087    (team balance has moderate impact)
weight_team_size:   0.068    (team size has small impact)
weight_diversity:   0.000    (redundant — captured by coverage features)
```

### Edge Case Categories (data augmentation)

The 200 synthetic training records span 16 categories designed to challenge the model:

| Category | Example | Expected Label |
|----------|---------|---------------|
| `borderline_dup_success` | Team with some role overlap but good skills | Success |
| `mediocre_good_fit` | Average skills but perfect role coverage | Success |
| `strong_partial_coverage` | High skills but missing required skills | Failure |
| `star_carries` | One excellent member, rest mediocre | Success |
| `decent_no_critical` | Decent team missing a critical role | Failure |
| `high_variance_fail` | Mix of 9/10 and 2/10 skill levels | Failure |
| `role_clones` | All members have the same role | Failure |
| `allstars_wrong_domain` | Great ML team assigned to web project | Failure |
| `too_many_cooks` | 7+ member team with diminishing returns | Failure |
| `diverse_but_weak` | Good role diversity but everyone scores 3/10 | Failure |

---

## API Endpoints

**Base URL:** `http://localhost:8000`

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/form-teams` | Form teams from uploaded resumes |
| POST | `/api/form-teams-v2` | V2 with skill percentages and validation |
| POST | `/api/validate-resumes` | Validate resume data quality |

### Feedback Loop Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/feedback` | Submit team outcome feedback |
| POST | `/api/retrain` | Retrain the ML model with all feedback data |
| POST | `/api/predict-team-success` | Predict success probability for a team |
| GET | `/api/feedback-weights` | Get current learned scoring weights |
| GET | `/api/feedback-data-stats` | Training data statistics |

### Example: Predict Team Success

```bash
curl -X POST http://localhost:8000/api/predict-team-success \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "TEST_01",
    "project": {
      "project_type": "web_application",
      "required_skills": ["React", "Node.js", "MongoDB"]
    },
    "members": [
      {"assigned_role": "frontend", "skills": ["React", "JavaScript"], "skill_level": 8, "experience_years": 2},
      {"assigned_role": "backend", "skills": ["Node.js", "MongoDB"], "skill_level": 7, "experience_years": 3}
    ]
  }'
```

Response:
```json
{
  "success_probability": 0.95,
  "prediction": "success",
  "factors": [
    "Good role diversity (1.00)",
    "Strong avg skill level (7.5/10)",
    "Good skill coverage (100%)",
    "Strong critical role coverage (100%)"
  ]
}
```

---

## Project Structure

```
SkillSyncAI/
├── README.md                             ← This file
├── SkillSyncAI_TeamFormation.ipynb       ← Notebook (Colab)
│
└── api/                                  ← FastAPI backend
    ├── main.py                           ← API routes & endpoints
    ├── ml_engine.py                      ← Core ML: scoring, matching, team formation
    ├── feedback_trainer.py               ← ML feedback loop (v2.0 pipeline)
    ├── rigorous_trainer.py               ← Training harness (experiments)
    ├── validation.py                     ← Input validation module
    ├── github_fetcher.py                 ← GitHub profile analysis
    │
    ├── feedback_training_data.json       ← 256 labeled team records
    ├── model_weights.pkl                 ← Trained model binary
    ├── training_report.json              ← Training metrics report
    │
    ├── test_endpoints.py                 ← API endpoint tests (6 tests)
    ├── requirements.txt                  ← Python dependencies
    ├── Dockerfile                        ← Container deployment
    └── render.yaml                       ← Render.com deployment config
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **ML Training** | scikit-learn 1.8 (SVM, RF, GB, calibration) |
| **NLP / Parsing** | pdfplumber, spaCy |
| **Data** | NumPy, pandas |
| **Serialization** | joblib, pickle |
| **Testing** | httpx (async HTTP client) |
| **Deployment** | Docker, Render |

---

## Quick Start

### Local Development

```bash
# Clone and setup
cd SkillSyncAI
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r api/requirements.txt

# Start server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000

# Run tests (in another terminal)
python test_endpoints.py
```

### Google Colab (Notebook only)

1. Upload `SkillSyncAI_TeamFormation.ipynb` to Colab
2. Upload resumes to `/content/resumes/`
3. Run all cells
4. Results in `/content/output/`

---

## The Scoring Formula

### Static Weights (default)
```
Match Score = 60% × Skill Match + 30% × Experience + 10% × Diversity
```

### Learned Weights (after ML training)
```
Match Score = 40.8% × Coverage + 24.6% × Experience + 19.2% × Skill
            + 8.7% × Balance + 6.8% × Team Size
```

The ML feedback loop replaces the static 60/30/10 split with data-driven weights that reflect what actually predicts team success.

---

## Team Formation Algorithm

**Snake Draft** for fairness:

```
Round 1: Team 1 → Team 2 → Team 3 → ... → Team N
Round 2: Team N → ... → Team 3 → Team 2 → Team 1  (reversed)
Round 3: Team 1 → Team 2 → Team 3 → ... → Team N
```

Each pick selects the best available candidate for the team's most-needed role, scored using the learned weights.

---

## 7 Roles

| Role | Skills | Critical For |
|------|--------|-------------|
| Frontend | React, HTML, CSS, JavaScript | web_application, mobile_app |
| Backend | Node.js, Python, SQL, APIs | web_application, api_service |
| Fullstack | MERN, MEAN stack | web_application |
| ML/AI | TensorFlow, PyTorch, NLP | ml_project |
| Data | Pandas, Tableau, Statistics | data_pipeline |
| UI/UX | Figma, Adobe XD | mobile_app |
| DevOps | Docker, AWS, CI/CD | api_service |

---

## Key Decisions Log

| Decision | Why |
|----------|-----|
| SVM_RBF over Random Forest | SVM had higher F1 (0.947 vs 0.92) on holdout; better on borderline cases |
| 25 features over 14 | Interaction terms (weakest_link, team_strength_index) captured patterns RF missed |
| Probability calibration | Raw SVM probabilities are unreliable; isotonic calibration fixes this |
| 200 synthetic edge cases | Original 55 records gave 100% CV — too easy. Edge cases force real generalization |
| Permutation importance | SVM_RBF has no `coef_` or `feature_importances_`; permutation importance works for any model |
| Single model over ensemble | Stacking ensemble (v2.1) scored 78% due to label noise. Simpler = better here. |

---

*Built for hackathon team formation — learns from real team outcomes to get better over time.*
