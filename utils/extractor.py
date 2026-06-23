"""
Disaster NLP Extractor
Handles entity extraction, severity classification, and urgency scoring.
"""

import re
import json
import pandas as pd
from datetime import datetime

# ─── Keyword Dictionaries ────────────────────────────────────────────────────

DISASTER_KEYWORDS = {
    "earthquake": ["earthquake", "quake", "tremor", "seismic", "magnitude", "richter", "aftershock"],
    "flood":      ["flood", "flooding", "inundation", "overflow", "submerged", "deluge", "waterlogged"],
    "fire":       ["fire", "wildfire", "blaze", "inferno", "burning", "arson", "forest fire", "bushfire"],
    "cyclone":    ["cyclone", "hurricane", "typhoon", "storm", "tornado", "windstorm", "gale"],
    "landslide":  ["landslide", "mudslide", "rockslide", "mudflow", "debris flow", "slope failure"],
    "drought":    ["drought", "dry spell", "water scarcity", "famine", "crop failure", "water shortage"],
    "accident":   ["accident", "collision", "crash", "explosion", "blast", "industrial accident", "chemical leak"],
}

SEVERITY_HIGH    = ["massive", "catastrophic", "devastating", "major", "severe", "critical", "extreme",
                    "collapsed", "destroyed", "killed", "dead", "deaths", "fatalities", "casualties"]
SEVERITY_MEDIUM  = ["significant", "moderate", "considerable", "damaged", "injured", "wounded", "affected",
                    "disrupted", "evacuated", "displaced"]
SEVERITY_LOW     = ["minor", "small", "limited", "slight", "contained", "manageable", "localized"]

URGENCY_HIGH     = ["rescue", "emergency", "trapped", "missing", "survivors", "immediate", "critical",
                    "urgent", "help needed", "sos", "life-threatening", "ongoing", "active"]
URGENCY_MEDIUM   = ["displaced", "evacuated", "shelter needed", "relief", "response", "teams deployed"]

RESOURCE_MAP = {
    "food":        ["food", "meals", "nutrition", "ration", "supplies", "hungry", "starving"],
    "water":       ["water", "drinking water", "clean water", "dehydration", "thirst"],
    "medical aid": ["medical", "hospital", "injured", "wounds", "treatment", "ambulance", "healthcare", "doctors"],
    "shelter":     ["shelter", "displaced", "homeless", "housing", "tents", "accommodation", "camp"],
    "rescue":      ["rescue", "trapped", "missing", "search", "survivors", "buried", "stranded"],
    "electricity": ["power outage", "electricity", "blackout", "generator", "grid failure"],
    "evacuation":  ["evacuation", "evacuate", "flee", "escape route", "relocation"],
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_disaster_type(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for dtype, keywords in DISASTER_KEYWORDS.items():
        scores[dtype] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def extract_severity(text: str) -> str:
    text_lower = text.lower()
    high   = sum(1 for w in SEVERITY_HIGH   if w in text_lower)
    medium = sum(1 for w in SEVERITY_MEDIUM if w in text_lower)
    low    = sum(1 for w in SEVERITY_LOW    if w in text_lower)
    if high > 0:   return "high"
    if medium > 0: return "medium"
    if low > 0:    return "low"
    return "medium"  # default


def extract_urgency(text: str) -> str:
    text_lower = text.lower()
    high   = sum(1 for w in URGENCY_HIGH   if w in text_lower)
    medium = sum(1 for w in URGENCY_MEDIUM if w in text_lower)
    if high > 0:   return "high"
    if medium > 0: return "medium"
    return "low"


def extract_resources(text: str) -> list:
    text_lower = text.lower()
    needed = []
    for resource, keywords in RESOURCE_MAP.items():
        if any(kw in text_lower for kw in keywords):
            needed.append(resource)
    return needed if needed else ["unknown"]


def extract_people_affected(text: str) -> str:
    patterns = [
        r'(\d[\d,]*)\s*(people|persons|individuals|residents|families|victims|casualties|displaced|injured|killed|dead)',
        r'(over|more than|at least|approximately|around|nearly)\s+(\d[\d,]*)\s*(people|families|victims)',
        r'(\d[\d,]*)\s*\+?\s*(affected|homeless|missing)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            nums = re.findall(r'\d[\d,]*', match.group())
            if nums:
                return nums[0].replace(",", "")
    return "unknown"


def extract_location(text: str) -> dict:
    """Simple heuristic location extraction (no spaCy required)."""
    # Look for capitalized proper nouns near disaster keywords
    sentences = text.split('.')
    location = {"city": "unknown", "region": "unknown", "country": "unknown"}

    country_hints = ["turkey", "india", "japan", "nepal", "indonesia", "philippines",
                     "bangladesh", "pakistan", "usa", "australia", "china", "brazil",
                     "mexico", "haiti", "chile", "myanmar", "afghanistan", "iran"]
    city_hints    = ["izmir", "istanbul", "mumbai", "tokyo", "kathmandu", "jakarta",
                     "manila", "dhaka", "karachi", "beijing", "rio", "sydney", "miami",
                     "new orleans", "port-au-prince", "kabul", "tehran"]

    text_lower = text.lower()
    for country in country_hints:
        if country in text_lower:
            location["country"] = country.title()
            break
    for city in city_hints:
        if city in text_lower:
            location["city"] = city.title()
            break

    # Also try capitalised word extraction
    caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    skip = {"The", "This", "That", "These", "Those", "A", "An", "At", "In", "On",
            "Rescue", "Teams", "Early", "Morning", "Several", "Buildings"}
    places = [w for w in caps if w not in skip]
    if places and location["city"] == "unknown":
        location["city"] = places[0]
    if len(places) > 1 and location["country"] == "unknown":
        location["country"] = places[-1]

    return location


def extract_entities(text: str) -> dict:
    orgs  = []
    org_patterns = ["rescue team", "red cross", "ndrf", "fema", "un", "unicef",
                    "army", "navy", "air force", "police", "fire department",
                    "national guard", "government", "ministry", "relief fund"]
    for org in org_patterns:
        if org in text.lower():
            orgs.append(org.title())

    location = extract_location(text)
    places = [v for v in location.values() if v != "unknown"]

    return {"places": places, "organizations": orgs if orgs else ["unknown"]}


def compute_confidence(text: str, extracted: dict) -> float:
    score = 0.4  # base
    if extracted["disaster_type"] != "unknown": score += 0.15
    if extracted["people_affected"] != "unknown": score += 0.15
    if extracted["location"]["city"] != "unknown": score += 0.10
    if extracted["location"]["country"] != "unknown": score += 0.10
    if len(extracted["resources_needed"]) > 1: score += 0.05
    if len(text.split()) > 30: score += 0.05
    return round(min(score, 1.0), 2)


# ─── Main Extraction Function ────────────────────────────────────────────────

def extract_disaster_info(text: str, source: str = "upload", record_id: str = None) -> dict:
    location  = extract_location(text)
    disaster  = extract_disaster_type(text)
    severity  = extract_severity(text)
    urgency   = extract_urgency(text)
    people    = extract_people_affected(text)
    resources = extract_resources(text)
    entities  = extract_entities(text)

    base = {
        "record_id":        record_id or f"REC-{datetime.now().strftime('%H%M%S')}",
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":           source,
        "disaster_type":    disaster,
        "location":         location,
        "severity":         severity,
        "urgency_level":    urgency,
        "people_affected":  people,
        "resources_needed": resources,
        "key_entities":     entities,
        "summary":          "",   # filled by summarizer
        "confidence_score": 0.0,
        "raw_text":         text[:500],
    }
    base["confidence_score"] = compute_confidence(text, base)
    return base


# ─── Batch Processing ────────────────────────────────────────────────────────

def process_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Process a dataframe of disaster texts and return enriched dataframe."""
    results = []
    for i, row in df.iterrows():
        text = str(row.get(text_col, ""))
        if not text.strip():
            continue
        info = extract_disaster_info(
            text,
            source=str(row.get("source", "upload")),
            record_id=str(row.get("id", f"REC-{i:04d}"))
        )
        # Flatten location for CSV
        info["city"]    = info["location"].get("city", "unknown")
        info["country"] = info["location"].get("country", "unknown")
        info["resources_str"]  = ", ".join(info["resources_needed"])
        info["organizations"]  = ", ".join(info["key_entities"].get("organizations", []))
        results.append(info)
    return pd.DataFrame(results)
