# 🎓 PlacementPredict — ML Analytics Dashboard

A full-stack machine learning dashboard for analysing, visualising, and modelling **50,000 student placement records**. Built with Flask and Python — no external ML libraries used for modelling; all algorithms implemented from scratch using NumPy and Pandas.

---

## 📸 Preview

> Home dashboard with animated KPI counters, ML pipeline visualisation, and placement intelligence cards.

---

## ✨ Features

| Module | What it does |
|---|---|
| **Load Data** | Paginated preview of the raw 50K CSV dataset with schema inspection |
| **EDA (Stats)** | Descriptive statistics (count, mean, std, min, Q1, median, Q3, max), missing values, random 10-row sample |
| **Graphs** | 7 chart types — bar, pie, histogram, scatter, box plot, count plot, correlation matrix |
| **Feature Engineering** | Min-Max, Standard, and Robust scaling · Ordinal & One-Hot Encoding · Downloadable CSVs |
| **Regression** | Simple Linear Regression (Gradient Descent) + Multi-Linear Regression (Normal Equation) with live salary predictor |
| **Dark Mode** | System-wide ☀️/🌙 toggle, persisted in `localStorage` |

---

## 🛠 Tech Stack

- **Backend** — Python 3, Flask, Pandas, NumPy, Seaborn, Matplotlib
- **Frontend** — Jinja2 templates, Vanilla CSS, Vanilla JS, Chart.js, Lucide Icons
- **ML** — All models implemented manually (no scikit-learn for modelling)
- **Data** — `placement_predict_50k Dataset.csv` (50,000 rows × 32 columns)

---

## 📂 Project Structure

```
PlacementPredict/
│
├── app.py                  # Flask routes & all ML logic
├── config.py               # Column definitions & constants
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment (Render/Heroku)
│
├── Dashboard/              # Jinja2 HTML templates + CSS
│   ├── base.html           # Base layout (header, sidebar, dark mode)
│   ├── index.html          # Home dashboard
│   ├── load.html           # Data loader
│   ├── eda.html            # Exploratory data analysis
│   ├── graphs.html         # Visualisations
│   ├── feature_engg.html   # Feature engineering
│   ├── regression.html     # Regression models
│   └── style.css           # Global stylesheet
│
├── Data/                   # Dataset (CSV)
├── Dashboard/Plot/         # Pre-generated Seaborn/Matplotlib charts
└── Source/                 # Data preprocessing scripts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/shambhavi-mahi/Placement_Predict.git
cd Placement_Predict

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Rows | 50,000 |
| Columns | 32 |
| Target | `PlacementStatus` (0 / 1) |
| Anomaly Flag | `IsAnomaly` (IQR-based) |

**Key features:** CGPA, SGPA (8 semesters), Internships, Projects, AptitudeTestScore, CodingTestScore, MockInterviewScore, SoftSkillsRating, Certifications, Publications, Salary Package

---

## 🤖 ML Algorithms (from scratch)

### Simple Linear Regression (SLR)
- Feature: CGPA → Target: Salary Package
- Algorithm: **Gradient Descent** (α = 0.01, 1000 epochs)
- Outputs: learned θ₀, θ₁, cost history, R² score

### Multi-Linear Regression (MLR)
- Features: CGPA, AptitudeTestScore, CodingTestScore, MockInterviewScore
- Algorithm: **Normal Equation** — `θ = (XᵀX)⁻¹ Xᵀy`
- Outputs: feature coefficients, MSE, RMSE, MAE, R²

---

## 🎨 Design Highlights

- **Color palette** — Deep Navy `#0F172A` · Bright Blue `#2563EB` · Cyan `#06B6D4` · Emerald `#10B981`
- **Animated hero** — Placement probability ring counts up on page load
- **KPI counters** — All statistics animate from 0 with ease-out cubic
- **ML Pipeline** — Animated glowing dot travels across 6 pipeline stages
- **Intelligence cards** — Arc progress visualisations for placement probability & data quality

---

## 📦 Deployment

The project includes a `Procfile` and `requirements.txt` for deployment on [Render](https://render.com/) or Heroku.

```
web: gunicorn app:app
```

---

## 👤 Author

**Shambhavi Mahi**  
GitHub: [@shambhavi-mahi](https://github.com/shambhavi-mahi)

---

## 📄 License

This project is for educational and academic purposes.
