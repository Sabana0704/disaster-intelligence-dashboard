# 🚨 Disaster Intelligence Dashboard
**NLP-Based Disaster Information Extraction & Automated Situation Summarization**

Academic Project · Python · Streamlit · Claude AI · Power BI

---

## Project Structure

```
disaster_nlp/
├── app.py                    ← Main Streamlit application
├── requirements.txt          ← Python dependencies
├── POWERBI_GUIDE.md          ← Step-by-step Power BI instructions
├── utils/
│   ├── extractor.py          ← NLP extraction pipeline
│   ├── summarizer.py         ← Claude AI summarization
│   └── powerbi_export.py     ← CSV export for Power BI
├── data/
│   └── sample_disasters.csv  ← Sample test data (10 records)
└── exports/
    └── disaster_latest.csv   ← Auto-generated Power BI export
```

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app.py

# 3. Open in browser (auto-opens)
# http://localhost:8501
```

---

## How to Use

### Option A — Upload Your Own Data
Your CSV needs a `text` column with disaster descriptions:
```csv
id,source,text
REC-001,Reddit,"A 6.8 earthquake struck Turkey..."
```

### Option B — Use Sample Data
Click **Load Sample Data** in the sidebar for 10 pre-loaded disaster records.

---

## AI Summaries (Optional)
Add your [Anthropic API key](https://console.anthropic.com/) in the sidebar for Claude-powered summaries.
Without a key, intelligent rule-based summaries are generated automatically.

---

## CSV Input Format

| Column   | Required | Description                        |
|----------|----------|------------------------------------|
| `text`   | ✅ Yes   | Unstructured disaster description  |
| `id`     | No       | Record identifier (auto-generated) |
| `source` | No       | Source name (Reddit, News, etc.)   |

---

## What Gets Extracted

| Field              | Description                                           |
|--------------------|-------------------------------------------------------|
| `disaster_type`    | earthquake / flood / fire / cyclone / landslide / etc |
| `severity`         | high / medium / low                                   |
| `urgency_level`    | high / medium / low                                   |
| `people_affected`  | Numeric estimate or "unknown"                         |
| `city / country`   | Extracted location entities                           |
| `resources_needed` | food, water, rescue, shelter, medical aid, etc.       |
| `summary`          | AI-generated 2-sentence situation summary             |
| `recommended_action`| AI-generated response recommendation                 |
| `confidence_score` | 0.0 – 1.0 extraction confidence                      |
