"""
Power BI Export Utility
Exports processed disaster data into a clean CSV that Power BI Desktop can consume.
Includes schema documentation and refresh instructions.
"""

import os
import pandas as pd
from datetime import datetime

EXPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'exports')

# Columns Power BI will use
POWERBI_COLUMNS = [
    "record_id",
    "timestamp",
    "source",
    "disaster_type",
    "city",
    "country",
    "severity",
    "urgency_level",
    "people_affected",
    "resources_str",
    "organizations",
    "confidence_score",
    "summary",
    "recommended_action",
]

SEVERITY_ORDER  = {"low": 1, "medium": 2, "high": 3}
URGENCY_ORDER   = {"low": 1, "medium": 2, "high": 3}


def export_for_powerbi(df: pd.DataFrame, filename: str = None) -> str:
    """
    Clean and export a processed dataframe to CSV for Power BI.
    Returns the export file path.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)

    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"disaster_data_{ts}.csv"

    export_path = os.path.join(EXPORT_DIR, filename)

    # Keep only relevant columns (fill missing with "unknown")
    export_df = pd.DataFrame()
    for col in POWERBI_COLUMNS:
        if col in df.columns:
            export_df[col] = df[col]
        else:
            export_df[col] = "unknown"

    # Add numeric severity/urgency for Power BI sorting
    export_df["severity_score"] = export_df["severity"].map(SEVERITY_ORDER).fillna(0).astype(int)
    export_df["urgency_score"]  = export_df["urgency_level"].map(URGENCY_ORDER).fillna(0).astype(int)

    # People affected as numeric where possible
    export_df["people_count"] = pd.to_numeric(
        export_df["people_affected"].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0).astype(int)

    export_df.to_csv(export_path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat
    return export_path


def get_powerbi_instructions(export_path: str) -> str:
    """Return step-by-step Power BI Desktop connection instructions."""
    abs_path = os.path.abspath(export_path)
    return f"""
## 📊 Connecting to Power BI Desktop — Step by Step

### Step 1 — Open Power BI Desktop
Download free from: https://powerbi.microsoft.com/desktop

### Step 2 — Import the CSV
1. Click **Home → Get Data → Text/CSV**
2. Browse to: `{abs_path}`
3. Click **Load**

### Step 3 — Verify Data Types
In the **Power Query Editor**:
- `timestamp`        → Date/Time
- `severity_score`   → Whole Number
- `urgency_score`    → Whole Number
- `people_count`     → Whole Number
- `confidence_score` → Decimal Number

### Step 4 — Build Recommended Visuals
| Visual Type      | Fields                                        |
|------------------|-----------------------------------------------|
| Bar Chart        | disaster_type → Count                         |
| Map              | city, country → Bubble size = people_count    |
| Donut Chart      | severity (Low / Medium / High)                |
| Card             | Total records, Max urgency_score              |
| Table            | record_id, city, summary, recommended_action  |
| Stacked Bar      | disaster_type × urgency_level                 |

### Step 5 — Live Refresh (Auto-Update)
1. In Power BI Desktop → **Transform Data → Data Source Settings**
2. Point to the **same CSV path**
3. Every time you run the Streamlit app and export, click **Refresh** in Power BI
4. For scheduled refresh → upgrade to Power BI Pro or use Power BI Gateway

### Step 6 — Publish (Optional, needs Pro)
Home → **Publish → My Workspace**

---
💡 **Tip:** Keep the CSV filename as `disaster_latest.csv` in the Streamlit app
   so Power BI always refreshes the same file without re-linking.
"""
