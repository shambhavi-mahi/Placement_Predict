import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw dataset (actual file on disk)
RAW_DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")

# Processed / cleaned outputs
BASE_DATA_PATH    = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# Train / Validation splits (used by Source scripts)
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

# Dashboard static assets
PLOTS_DIR    = os.path.join(BASE_DIR, "Dashboard", "Plot")
LOGISTIC_DIR = os.path.join(BASE_DIR, "Dashboard", "Plot", "Logistic")

# Reports
REPORTS_DIR    = os.path.join(BASE_DIR, "Output", "Report")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary_report.txt")

# ML constants
RANDOM_STATE = 42
TARGET_COLUMN = "PlacementStatus"

# Numeric feature columns used by logistic / linear regression
NUMERIC_COLUMNS = [
    "CGPA", "AptitudeTestScore", "CodingTestScore",
    "MockInterviewScore", "AttendancePercent",
    "Internships", "Projects", "Certifications",
    "WorkshopsAttended", "SoftSkillsRating",
]

# Column metadata
CATEGORICAL_COLS = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier"
]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
ID_COL  = "StudentID"
ID_COLS = [ID_COL]
