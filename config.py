import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw dataset (actual file on disk)
RAW_DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset (2).csv")

# Processed / cleaned outputs
BASE_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# Dashboard static assets
PLOTS_DIR = os.path.join(BASE_DIR, "Dashboard", "Plot")

# Reports
REPORTS_DIR = os.path.join(BASE_DIR, "Output", "Report")
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, "eda_summary_report.txt")

# Column metadata
CATEGORICAL_COLS = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier"
]
TARGET_COLS = ["PlacementStatus", "IsAnomaly"]
ID_COL = "StudentID"          # singular — used by data_utils
ID_COLS = [ID_COL]            # list alias kept for backwards compat
