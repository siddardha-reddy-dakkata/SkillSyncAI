# 🧠 SkillSyncAI

### AI-Powered Hackathon Team Formation System

> **One-liner:** Upload resumes → AI extracts skills → Forms balanced teams → Explains why

---

## 🚀 Quick Start (2 minutes)

```
1. Open SkillSyncAI_TeamFormation.ipynb in Google Colab
2. Upload PDFs to /content/resumes/ folder
3. Run all cells
4. Get teams in /output/ folder
```

That's it! 🎉

---

## 📖 What Does This Project Do?

Imagine you're organizing a hackathon with **70 students** and need to form **balanced teams**. Doing this manually would take hours and might not be fair.

**SkillSyncAI solves this by:**

| Step | What Happens |
|------|--------------|
| 📄 **Read Resumes** | AI reads all PDF resumes automatically |
| 🔍 **Find Skills** | Detects skills like React, Python, ML, etc. |
| 👤 **Build Profiles** | Creates a skill profile for each student |
| 🎯 **Match to Projects** | Calculates who fits which project best |
| ⚖️ **Form Teams** | Creates balanced teams (not all stars in one team) |
| 💬 **Explain Decisions** | Tells you WHY each person was selected |

---

## 🎬 For Presentations: Key Talking Points

### Slide 1: The Problem
> "Manually forming hackathon teams is time-consuming and often unfair. Strong students cluster together, leaving weaker teams behind."

### Slide 2: Our Solution
> "SkillSyncAI uses NLP to read resumes, extract skills, and automatically form balanced teams with explainable AI decisions."

### Slide 3: How It Works (Simple Version)
```
Resume PDFs → Text Extraction → Skill Detection → Scoring → Team Formation → Results
```

### Slide 4: The AI Magic
> "We use rule-based NLP with a dictionary of 170+ tech keywords mapped to 7 roles. Each student gets a score for each role."

### Slide 5: Fairness Algorithm
> "Snake draft ensures no team gets all the best students. Teams are balanced by design."

### Slide 6: Explainable AI
> "Every team assignment comes with a human-readable reason. No black box decisions."

---

## 🎭 The 7 Roles (Know These!)

Every student is classified into one of these roles:

| Role | What They Do | Example Skills |
|------|--------------|----------------|
| 🎨 **Frontend** | Build what users see | React, HTML, CSS, JavaScript |
| ⚙️ **Backend** | Build server & database | Node.js, Python, SQL, APIs |
| 🔄 **Fullstack** | Both frontend + backend | MERN, MEAN stack |
| 🤖 **ML/AI** | Machine learning models | TensorFlow, PyTorch, NLP |
| 📊 **Data** | Data analysis & visualization | Pandas, Tableau, Statistics |
| ✏️ **UI/UX** | Design user interfaces | Figma, Adobe XD, Wireframes |
| ☁️ **DevOps** | Deployment & infrastructure | Docker, AWS, CI/CD |

---

## 📊 The Matching Formula (Important for Viva!)

```
Match Score = 60% × Skill Match + 30% × Experience + 10% × Diversity
```

**In plain English:**
- **60%** — How good are they at the required role?
- **30%** — Do they have internships/projects/hackathon experience?
- **10%** — Can they help with other roles too?

**Example:**
> "John has React score 8/10, 2 internships, and knows some backend too. His Frontend match score = 0.85"

---

## 🔄 How Teams Are Formed (The Fair Way)

We use **Snake Draft** strategy for balanced team formation:

```
Round 1: Team 1 picks → Team 2 picks → Team 3 picks → ... → Team N
Round 2: Team N picks → ... → Team 3 picks → Team 2 picks → Team 1 (reversed!)
Round 3: Team 1 picks → Team 2 picks → Team 3 picks → ... → Team N
...
```

**Key Features:**
- **Multiple teams can work on the same project** — realistic hackathon scenario
- **Even team sizes** — if 73 students ÷ 5 target size, we make mix of 4s and 5s (not 14 teams of 5 + 1 team of 3)
- **Skill balancing** — no team gets all the best students
- **Role diversity** — each team gets a mix of Frontend, Backend, ML, etc.

**Why Snake Draft?** Alternating direction ensures late-picking teams in Round 1 get first picks in Round 2.

---

## 📁 What You Get (Outputs)

After running the notebook, you get these files:

| File | What's Inside | Use For |
|------|---------------|---------|
| `final_teams.json` | Complete teams with explanations | Frontend integration |
| `student_profiles.json` | All student skill profiles | Debugging/analysis |
| `team_assignments.csv` | Teams in spreadsheet format | Quick viewing |
| `analysis_charts.png` | Visual summary graphs | Presentations |

### Sample Output (What the JSON looks like)

```json
{
  "team_name": "Team 1",
  "team_id": "Team_01",
  "team_size": 5,
  "members": [
    {
      "name": "John Doe",
      "role": "Frontend",
      "overall_score": 0.85,
      "reason": "Strong React skills (8/10); 2 internship experiences; Primary frontend developer"
    },
    {
      "name": "Jane Smith", 
      "role": "Backend",
      "overall_score": 0.78,
      "reason": "Good Node.js and MongoDB experience; Led a hackathon project"
    }
  ],
  "roles_covered": ["Frontend", "Backend", "ML/AI", "Data", "DevOps"]
}
```

---

## 🛠️ Setup Instructions (For Running It)

### Option A: Google Colab (Recommended)

1. **Upload the notebook**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload `SkillSyncAI_TeamFormation.ipynb`

2. **Create resume folder**
   - Click folder icon (left sidebar)
   - Create new folder: `resumes`

3. **Upload PDFs**
   - Upload all resume PDFs to `/content/resumes/`
   - **Important:** Name files as `StudentName.pdf` (e.g., `John_Doe.pdf`)

4. **Run the notebook**
   - Click "Runtime" → "Run all"
   - Wait 2-3 minutes

5. **Download results**
   - Find outputs in `/content/output/` folder

### Option B: Local Setup

```bash
# Install dependencies
pip install pdfplumber spacy pandas matplotlib
python -m spacy download en_core_web_sm

# Run Jupyter
jupyter notebook SkillSyncAI_TeamFormation.ipynb
```

---

## 📝 How to Name Resume Files

The AI uses the **filename** as the student's name:

| Filename | Extracted Name |
|----------|----------------|
| `John_Doe.pdf` | John Doe |
| `jane-smith.pdf` | Jane Smith |
| `RAHUL KUMAR.pdf` | Rahul Kumar |
| `Alice_Resume.pdf` | Alice |
| `bob-cv-2024.pdf` | Bob |

**Tips:**
- Use underscores `_` or hyphens `-` between names
- Words like "resume" and "cv" are automatically removed
- Case doesn't matter (auto-converts to Title Case)

---

## 🎯 How to Define Projects

Edit this section in the notebook:

```python
PROJECTS = [
    {
        "name": "Smart Campus Assistant",
        "description": "We need a React frontend, Node.js backend, and ML model for predictions.",
        "team_size": 4,
    },
    {
        "name": "Health Dashboard",
        "description": "Data visualization with Python, pandas, Plotly. Flask backend needed.",
        "team_size": 4,
    },
]
```

**The AI reads the description and figures out:**
- Which roles are needed (Frontend, Backend, ML, etc.)
- Which specific skills to prioritize (React, Node.js, etc.)

---

## ❓ Viva Questions & Answers

### Q1: "Why not use Machine Learning for skill extraction?"

> "For a prototype with ~70 resumes, rule-based NLP is faster, doesn't need training data, and is fully explainable. ML would be overkill and harder to debug."

### Q2: "How do you ensure teams are balanced?"

> "Two ways: (1) Snake draft so no team gets all top candidates, (2) Even team sizes (73 students ÷ 5 = mix of 4s and 5s, not 14×5 + 1×3)."

### Q3: "What makes this 'Explainable AI'?"

> "Every assignment includes a human-readable reason citing specific skills, experience counts, and match scores. No black box."

### Q4: "What's the matching formula?"

> "60% role skill score, 30% experience score, 10% skill diversity. We also add bonuses for primary role match and priority skills."

### Q5: "How do you handle students with no matching skills?"

> "They get assigned to remaining slots after role-specific picks. The system logs unassigned students if there aren't enough project slots."

### Q6: "Can you customize the algorithm?"

> "Yes! Weights are configurable. You can change 60/30/10 split. You can also add keywords to the skill dictionary."

---

## 🐛 Common Problems & Fixes

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `!pip install pdfplumber spacy pandas` |
| "Folder not found" | Create the `resumes` folder manually |
| "No text extracted" | PDF might be scanned image, not selectable text |
| "All students same role" | Resumes might be too similar, or add more keywords |
| "Too many unassigned" | Add more projects or increase team sizes |

---

## 📈 What the Charts Show

The notebook generates 4 visualizations:

1. **Pie Chart** — Role distribution (how many Frontend vs Backend vs ML students)
2. **Bar Chart** — Team balance scores (which teams are well-balanced)
3. **Histogram** — Experience score distribution (are students experienced?)
4. **Box Plot** — Match score spread per team (consistency within teams)

---

## 🔮 Future Improvements (Mention in Presentation)

If we had more time, we could add:

1. **GitHub Analysis** — Check actual code quality
2. **LinkedIn Integration** — Extract endorsements
3. **Semantic Matching** — Use AI embeddings instead of keywords
4. **Feedback Loop** — Learn from past team performance
5. **Web Interface** — Real-time team formation

---

## 📂 Project Structure

```
SkillSyncAI/
│
├── 📓 SkillSyncAI_TeamFormation.ipynb   ← Main AI notebook
├── 📖 README.md                          ← This file
│
├── 📁 resumes/                           ← Put PDFs here
│   ├── John_Doe.pdf
│   ├── Jane_Smith.pdf
│   └── ...
│
└── 📁 output/                            ← Results appear here
    ├── final_teams.json
    ├── student_profiles.json
    ├── team_assignments.csv
    └── analysis_charts.png
```

---

## 👥 Team Responsibilities

| Member | Responsibility |
|--------|----------------|
| **[Your Name]** | AI Logic (this notebook) |
| **[Teammate 1]** | Frontend UI |
| **[Teammate 2]** | Backend APIs |
| **[Teammate 3]** | Modal Design |

---

## 📌 One-Page Summary (Print This!)

```
┌─────────────────────────────────────────────────────────┐
│                    SKILLSYNCAI                          │
│         AI-Powered Hackathon Team Formation             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUT:   PDF Resumes (~70) + Project Descriptions      │
│                                                         │
│  PROCESS: NLP Skill Extraction → Scoring → Matching     │
│                                                         │
│  OUTPUT:  Balanced Teams + Explanations (JSON/CSV)      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  KEY FORMULA:                                           │
│  Score = 60% Skill + 30% Experience + 10% Diversity     │
│                                                         │
│  FAIRNESS: Snake Draft + Even Team Sizes                │
│                                                         │
│  ROLES: Frontend, Backend, Fullstack, ML/AI,            │
│         Data, UI/UX, DevOps                             │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TECH STACK:                                            │
│  Python, pdfplumber, spaCy, pandas, matplotlib          │
│                                                         │
│  ENVIRONMENT: Google Colab                              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist Before Demo

- [ ] Resume folder has all PDFs
- [ ] Filenames are proper (StudentName.pdf)
- [ ] Projects are defined in the notebook
- [ ] All cells run without errors
- [ ] Output folder has JSON/CSV files
- [ ] Charts are generated
- [ ] Tested with sample data first

---

*Made with ❤️ for college project evaluation*
