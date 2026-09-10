"""
Dallas Building Permits — Flask Prediction API
================================================
Two-Stage Hurdle Model REST API

Endpoints:
    POST /predict  — Predict permit value from raw permit details
    GET  /health   — Health check
    GET  /         — API info page

Usage:
    curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{
            "permit_type": "Building (BU) Commercial New Construction",
            "zip_code": "75201",
            "square_footage": 15000,
            "occupancy_type": "COMMERCIAL OFFICE BUILDING",
            "contractor_name": "ROGERS OBRIEN CONSTRUCTION",
            "contractor_city": "DALLAS",
            "issue_month": 6,
            "issue_year": 2024
        }'
"""

import os
import json
import re
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# ============================================================================
# LOAD MODELS & ENCODERS AT STARTUP
# ============================================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading models and encoders...")
stage1_clf = joblib.load(os.path.join(MODEL_DIR, "stage1_classifier.joblib"))
stage2_reg = joblib.load(os.path.join(MODEL_DIR, "stage2_final_model.joblib"))

with open(os.path.join(MODEL_DIR, "encoders.json"), "r") as f:
    ENC = json.load(f)

FEATURE_COLS = ENC["feature_cols"]
FEATURE_COLS_REG = ENC.get("feature_cols_reg", [c for c in FEATURE_COLS if c != "area_value_ratio"])
print(f"  Stage 1 classifier loaded")
print(f"  Stage 2 regressor loaded")
print(f"  Stage 1 feature vector size: {len(FEATURE_COLS)}")
print(f"  Stage 2 feature vector size: {len(FEATURE_COLS_REG)}")

# ============================================================================
# PERMIT TYPE PARSER (mirrors data_cleaning.py logic)
# ============================================================================

def parse_permit_type(permit_type_str):
    """Parse composite permit type string into 4 components."""
    standard_pattern = re.compile(
        r'^(.*?)\s*\((\w+)\)\s*(Single Family|Commercial|Multi Family)\s+'
        r'(New Construction|Alteration|Renovation|Addition|Reconstruction|Finish Out)$'
    )
    match = standard_pattern.match(permit_type_str.strip())
    if match:
        return {
            "permit_category": match.group(1).strip(),
            "permit_code": match.group(2).strip(),
            "sub_category": match.group(3).strip(),
            "work_type": match.group(4).strip(),
        }
    # Fallback: use full string as category
    return {
        "permit_category": permit_type_str.strip(),
        "permit_code": "NA",
        "sub_category": "General",
        "work_type": "Other",
    }


# ============================================================================
# FEATURE ENGINEERING (single-record version)
# ============================================================================

# Build a case-insensitive lookup for land use + substring matching
_LAND_USE_LOWER = {k.upper(): k for k in ENC["land_use_freq_map"]}

def _fuzzy_land_use(occupancy_type):
    """Find the best-matching land use category from the encoder map."""
    occ_upper = occupancy_type.upper().strip()
    # Exact match (case-insensitive)
    if occ_upper in _LAND_USE_LOWER:
        return _LAND_USE_LOWER[occ_upper]
    # Substring match: find the longest key contained in the input
    best_match = None
    best_len = 0
    for key_upper, key_orig in _LAND_USE_LOWER.items():
        if key_upper in occ_upper and len(key_upper) > best_len:
            best_match = key_orig
            best_len = len(key_upper)
    # Reverse substring: find keys that contain the input
    if best_match is None:
        for key_upper, key_orig in _LAND_USE_LOWER.items():
            if occ_upper in key_upper and len(key_upper) > best_len:
                best_match = key_orig
                best_len = len(key_upper)
    return best_match


def build_feature_vector(data):
    """
    Transform raw API input into a 63-element feature vector.
    
    Expected input fields:
        permit_type (str): e.g., "Building (BU) Commercial New Construction"
        zip_code (str): e.g., "75201"
        square_footage (float): e.g., 15000
        occupancy_type (str): e.g., "OFFICE BUILDING" (Land Use)
        contractor_name (str, optional): contractor company name
        contractor_city (str, optional): contractor city
        issue_month (int): 1-12
        issue_year (int): e.g., 2024
    """
    # Parse permit type
    parsed = parse_permit_type(data.get("permit_type", ""))
    
    # Temporal features — clamp to training range to avoid extrapolation
    issue_year = int(data.get("issue_year", 2020))
    issue_month = int(data.get("issue_month", 6))
    issue_quarter = (issue_month - 1) // 3 + 1
    epoch = datetime(2000, 1, 1)
    approx_date = datetime(min(issue_year, 2020), issue_month, 15)
    days_since_epoch = (approx_date - epoch).days
    # Clamp to training range [6576, 7546]
    days_since_epoch = max(6576, min(7546, days_since_epoch))
    issue_year = max(2018, min(2020, issue_year))
    issue_day_of_week = 2  # Default Wednesday
    is_q4 = 1 if issue_quarter == 4 else 0
    
    # Area
    area = float(data.get("square_footage", 0))
    log_area = float(np.log1p(area)) if area > 0 else 0.0
    
    # Contractor features
    contractor_name = data.get("contractor_name", "UNKNOWN")
    contractor_city = data.get("contractor_city", "UNKNOWN").upper()
    
    # Look up contractor permit count (default to median if unknown)
    contractor_permit_count = ENC["median_contractor_count"]
    is_repeat_contractor = 1 if contractor_permit_count >= 10 else 0
    
    # Contractor city frequency
    contractor_city_freq = ENC["city_freq_map"].get(contractor_city, 1)
    
    # Zip code target encoding
    zip_code = str(data.get("zip_code", "75229"))
    zip_target_encoded = ENC["zip_target_map"].get(zip_code, ENC["zip_overall_mean"])
    
    # Land use frequency encoding — with fuzzy matching
    occupancy_type = str(data.get("occupancy_type", "SINGLE FAMILY DWELLING"))
    matched_land_use = _fuzzy_land_use(occupancy_type)
    if matched_land_use:
        land_use_freq = ENC["land_use_freq_map"][matched_land_use]
    else:
        # Default to median frequency
        all_freqs = sorted(ENC["land_use_freq_map"].values())
        land_use_freq = all_freqs[len(all_freqs) // 2]
    
    # Permit code frequency
    permit_code = parsed["permit_code"]
    permit_code_freq = ENC["permit_code_freq_map"].get(permit_code, 1)
    
    # area_value_ratio: can't compute without Value (that's what we're predicting)
    # Use training median when area > 0 (typical ratio for construction permits)
    # Training data median area_value_ratio for positive-value permits ~ 0.03
    if area > 0:
        area_value_ratio = 0.03  # Approximate median from training data
    else:
        area_value_ratio = 0.0

    # Build the feature vector in the EXACT order the model expects
    feature_dict = {
        "Area": area,
        "issue_year": issue_year,
        "issue_month": issue_month,
        "issue_quarter": issue_quarter,
        "days_since_epoch": days_since_epoch,
        "issue_day_of_week": issue_day_of_week,
        "is_q4": is_q4,
        "contractor_permit_count": contractor_permit_count,
        "is_repeat_contractor": is_repeat_contractor,
        "contractor_city_freq": contractor_city_freq,
        "land_use_freq": land_use_freq,
        "zip_target_encoded": zip_target_encoded,
        "permit_code_freq": permit_code_freq,
        "area_value_ratio": area_value_ratio,
        "log_area": log_area,
    }
    
    # One-hot encode permit_category
    for col in ENC["ohe_permit_cats"]:
        cat_name = col.replace("permit_category_", "")
        feature_dict[col] = 1 if parsed["permit_category"] == cat_name else 0
    
    # One-hot encode sub_category
    for col in ENC["ohe_sub_cats"]:
        cat_name = col.replace("sub_category_", "")
        feature_dict[col] = 1 if parsed["sub_category"] == cat_name else 0
    
    # One-hot encode work_type
    for col in ENC["ohe_work_types"]:
        cat_name = col.replace("work_type_", "")
        feature_dict[col] = 1 if parsed["work_type"] == cat_name else 0
    
    # Build vector in exact column order
    vector = np.array([feature_dict.get(col, 0) for col in FEATURE_COLS], dtype=np.float64)
    
    return vector, parsed


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)

# Landing page HTML with interactive prediction form
LANDING_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dallas Permit Value Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', system-ui, sans-serif; background: #0a0e1a; color: #e2e8f0; min-height: 100vh; }

        /* Animated gradient background */
        .bg-glow {
            position: fixed; top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle at 30% 40%, rgba(56,189,248,0.06) 0%, transparent 50%),
                        radial-gradient(circle at 70% 60%, rgba(129,140,248,0.05) 0%, transparent 50%);
            animation: drift 20s ease-in-out infinite alternate;
            z-index: 0;
        }
        @keyframes drift { 0% { transform: translate(0,0); } 100% { transform: translate(-3%,-2%); } }

        .page { position: relative; z-index: 1; max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

        /* Header */
        .header { text-align: center; margin-bottom: 2.5rem; padding-top: 1.5rem; }
        h1 { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.4rem; }
        .subtitle { color: #64748b; font-size: 1rem; }
        .badge { display: inline-block; background: #065f46; color: #6ee7b7; padding: 0.2rem 0.7rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; margin-left: 0.5rem; animation: pulse 2s ease-in-out infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }

        /* Stats bar */
        .stats-bar { display: flex; justify-content: center; gap: 2.5rem; margin: 1.5rem 0 2rem; }
        .stat { text-align: center; }
        .stat .val { font-size: 1.6rem; font-weight: 800; color: #38bdf8; }
        .stat .label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.15rem; }

        /* Cards */
        .card { background: rgba(30,41,59,0.7); backdrop-filter: blur(12px); border: 1px solid #1e293b; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; transition: border-color 0.3s; }
        .card:hover { border-color: #334155; }
        .card h2 { color: #38bdf8; font-size: 1.05rem; font-weight: 700; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem; }
        .card h2 .icon { font-size: 1.2rem; }

        /* Form grid */
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .form-group { display: flex; flex-direction: column; }
        .form-group.full { grid-column: 1 / -1; }
        label { font-size: 0.78rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.06em; }
        input, select { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 0.7rem 0.9rem; color: #e2e8f0; font-family: inherit; font-size: 0.9rem; transition: border-color 0.2s, box-shadow 0.2s; outline: none; }
        input:focus, select:focus { border-color: #38bdf8; box-shadow: 0 0 0 3px rgba(56,189,248,0.15); }
        select { cursor: pointer; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='%2394a3b8' viewBox='0 0 20 20'%3E%3Cpath d='M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 0.7rem center; background-size: 1.1rem; padding-right: 2.2rem; }

        /* Button */
        .btn { background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; border: none; border-radius: 12px; padding: 0.85rem 2rem; font-family: inherit; font-size: 1rem; font-weight: 700; cursor: pointer; transition: transform 0.15s, box-shadow 0.2s; margin-top: 0.5rem; width: 100%; letter-spacing: 0.02em; }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(37,99,235,0.35); }
        .btn:active { transform: translateY(0); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn .spinner { display: inline-block; width: 1rem; height: 1rem; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 0.5rem; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Result */
        .result { display: none; margin-top: 1.5rem; }
        .result.show { display: block; animation: slideUp 0.4s ease-out; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

        .result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
        .result-value { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #34d399, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .result-class { font-size: 0.8rem; font-weight: 600; padding: 0.3rem 0.8rem; border-radius: 8px; }
        .result-class.construction { background: rgba(52,211,153,0.15); color: #34d399; }
        .result-class.administrative { background: rgba(251,191,36,0.15); color: #fbbf24; }

        .result-details { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-top: 1rem; }
        .detail-item { background: #0f172a; border-radius: 10px; padding: 0.8rem 1rem; }
        .detail-item .dl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
        .detail-item .dv { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-top: 0.2rem; }

        .ci-bar { margin-top: 1rem; background: #0f172a; border-radius: 10px; padding: 1rem; }
        .ci-bar .ci-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
        .ci-track { height: 8px; background: #1e293b; border-radius: 4px; position: relative; overflow: hidden; }
        .ci-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #7c3aed); border-radius: 4px; transition: width 0.6s ease-out; }
        .ci-labels { display: flex; justify-content: space-between; margin-top: 0.4rem; font-size: 0.8rem; color: #94a3b8; }

        /* Footer */
        .footer { text-align: center; margin-top: 2rem; color: #334155; font-size: 0.75rem; }
        .footer a { color: #38bdf8; text-decoration: none; }
    </style>
</head>
<body>
    <div class="bg-glow"></div>
    <div class="page">
        <div class="header">
            <h1>Dallas Permit Value Predictor</h1>
            <p class="subtitle">Predict building permit construction costs</p>
        </div>

        <!-- Prediction Form -->
        <div class="card">
            <h2><span class="icon">&#9881;</span> Predict Permit Value</h2>
            <form id="predictForm" onsubmit="return submitPrediction(event)">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="permit_category">Permit Category</label>
                        <select id="permit_category">
                            <option value="Building">Building</option>
                            <option value="Electrical">Electrical</option>
                            <option value="Plumbing">Plumbing</option>
                            <option value="Mechanical">Mechanical</option>
                            <option value="Fence">Fence</option>
                            <option value="Swimming Pool">Swimming Pool</option>
                            <option value="Fire Alarm">Fire Alarm</option>
                            <option value="Fire Sprinkler (Major Work)">Fire Sprinkler (Major)</option>
                            <option value="Demolition Permit Commercial">Demolition (Commercial)</option>
                            <option value="Demolition Permit SFD/Duplex">Demolition (SFD/Duplex)</option>
                            <option value="Electrical Sign New Construction">Electrical Sign</option>
                            <option value="Elevator">Elevator</option>
                            <option value="Grading and Paving">Grading and Paving</option>
                            <option value="Lawn Sprinkler">Lawn Sprinkler</option>
                            <option value="Backflow">Backflow</option>
                            <option value="Barricade">Barricade</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="sub_category">Sub-Category</label>
                        <select id="sub_category">
                            <option value="Single Family">Single Family</option>
                            <option value="Commercial">Commercial</option>
                            <option value="Multi Family">Multi Family</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="work_type">Work Type</label>
                        <select id="work_type">
                            <option value="New Construction">New Construction</option>
                            <option value="Alteration">Alteration</option>
                            <option value="Renovation">Renovation</option>
                            <option value="Addition">Addition</option>
                            <option value="Reconstruction">Reconstruction</option>
                            <option value="Finish Out">Finish Out</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="zip_code">ZIP Code</label>
                        <input type="text" id="zip_code" value="75201" placeholder="e.g. 75201">
                    </div>
                    <div class="form-group">
                        <label for="square_footage">Square Footage</label>
                        <input type="number" id="square_footage" value="5000" min="0" step="100" placeholder="e.g. 5000">
                    </div>
                    <div class="form-group">
                        <label for="occupancy_type">Land Use / Occupancy</label>
                        <select id="occupancy_type">
                            <option value="SINGLE FAMILY DWELLING">Single Family Dwelling</option>
                            <option value="MULTI-FAMILY DWELLING">Multi-Family Dwelling</option>
                            <option value="OFFICE BUILDING" selected>Office Building</option>
                            <option value="RESTAURANT WITHOUT DRIVE-IN SERVICE">Restaurant</option>
                            <option value="GEN MERCHANDISE OR FOOD STORE < 3500 SQ. FT.">Retail Store (Small)</option>
                            <option value="GEN MERCHANDISE OR FOOD STORE > 3500 SQ. FT.">Retail Store (Large)</option>
                            <option value="OFFICE SHOWROOM/WAREHOUSE">Office/Showroom/Warehouse</option>
                            <option value="VACANT FLOOR SPACE">Vacant Floor Space</option>
                            <option value="COMMUNICATIONS">Communications</option>
                            <option value="CHURCH">Church</option>
                            <option value="HOTEL/MOTEL">Hotel/Motel</option>
                            <option value="HOSPITAL / MEDICAL CLINIC">Hospital/Medical Clinic</option>
                            <option value="SCHOOL">School</option>
                            <option value="COMMERCIAL">Commercial (General)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="contractor_city">Contractor City</label>
                        <select id="contractor_city">
                            <option value="DALLAS">Dallas</option>
                            <option value="FORT WORTH">Fort Worth</option>
                            <option value="PLANO">Plano</option>
                            <option value="IRVING">Irving</option>
                            <option value="GARLAND">Garland</option>
                            <option value="MESQUITE">Mesquite</option>
                            <option value="ARLINGTON">Arlington</option>
                            <option value="RICHARDSON">Richardson</option>
                            <option value="CARROLLTON">Carrollton</option>
                            <option value="UNKNOWN">Other / Unknown</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="issue_month">Issue Month</label>
                        <select id="issue_month">
                            <option value="1">January</option>
                            <option value="2">February</option>
                            <option value="3">March</option>
                            <option value="4">April</option>
                            <option value="5">May</option>
                            <option value="6" selected>June</option>
                            <option value="7">July</option>
                            <option value="8">August</option>
                            <option value="9">September</option>
                            <option value="10">October</option>
                            <option value="11">November</option>
                            <option value="12">December</option>
                        </select>
                    </div>
                    <div class="form-group full">
                        <button type="submit" class="btn" id="submitBtn">
                            Predict Permit Value
                        </button>
                    </div>
                </div>
            </form>
        </div>

        <!-- Result Card -->
        <div class="card result" id="resultCard">
            <h2><span class="icon">&#10003;</span> Prediction Result</h2>
            <div class="result-header">
                <div class="result-value" id="resultValue">$0</div>
                <span class="result-class" id="resultClass">—</span>
            </div>
            <div class="result-details">
                <div class="detail-item"><div class="dl">Permit Category</div><div class="dv" id="rCat">—</div></div>
                <div class="detail-item"><div class="dl">Work Type</div><div class="dv" id="rWork">—</div></div>
                <div class="detail-item"><div class="dl">Sub-Category</div><div class="dv" id="rSub">—</div></div>
                <div class="detail-item"><div class="dl">Stage 1 (P(positive))</div><div class="dv" id="rProb">—</div></div>
            </div>
            <div class="ci-bar" id="ciBar" style="display:none;">
                <div class="ci-label">95% Confidence Interval</div>
                <div class="ci-track"><div class="ci-fill" id="ciFill" style="width:0%"></div></div>
                <div class="ci-labels"><span id="ciLow">—</span><span id="ciHigh">—</span></div>
            </div>
        </div>

        <div class="footer">
            ASDS 6302 Final Project &mdash; University of Texas at Arlington &mdash;
            <a href="/health">GET /health</a> &nbsp;|&nbsp; <a href="javascript:void(0)" onclick="showJSON()">POST /predict (JSON)</a>
        </div>
    </div>

    <script>
        function buildPermitType() {
            const cat = document.getElementById('permit_category').value;
            const sub = document.getElementById('sub_category').value;
            const work = document.getElementById('work_type').value;
            // Non-standard categories that don't follow the (CODE) pattern
            const nonStandard = ['Demolition Permit Commercial', 'Demolition Permit SFD/Duplex',
                'Electrical Sign New Construction', 'Barricade', 'Backflow'];
            if (nonStandard.includes(cat)) {
                return cat;
            }
            const codeMap = { 'Building':'BU','Electrical':'EL','Plumbing':'PL','Mechanical':'ME',
                'Fence':'FN','Swimming Pool':'SP','Fire Alarm':'FA','Fire Sprinkler (Major Work)':'FS',
                'Elevator':'EV','Grading and Paving':'GP','Lawn Sprinkler':'LS' };
            const code = codeMap[cat] || 'BU';
            return cat + ' (' + code + ') ' + sub + ' ' + work;
        }

        async function submitPrediction(e) {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Predicting...';

            const payload = {
                permit_type: buildPermitType(),
                zip_code: document.getElementById('zip_code').value,
                square_footage: parseFloat(document.getElementById('square_footage').value) || 0,
                occupancy_type: document.getElementById('occupancy_type').value,
                contractor_city: document.getElementById('contractor_city').value,
                issue_month: parseInt(document.getElementById('issue_month').value),
                issue_year: 2020
            };

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                showResult(data);
            } catch (err) {
                alert('Error: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = 'Predict Permit Value';
            }
        }

        function showResult(data) {
            const card = document.getElementById('resultCard');
            card.classList.add('show');

            document.getElementById('resultValue').textContent = data.predicted_value_formatted;
            const classEl = document.getElementById('resultClass');
            classEl.textContent = data.permit_class === 'construction' ? 'Construction' : 'Administrative ($0)';
            classEl.className = 'result-class ' + data.permit_class;

            document.getElementById('rCat').textContent = data.parsed_input.permit_category;
            document.getElementById('rWork').textContent = data.parsed_input.work_type;
            document.getElementById('rSub').textContent = data.parsed_input.sub_category;
            document.getElementById('rProb').textContent = (data.stage1_result.probability_positive * 100).toFixed(1) + '%';

            const ciBar = document.getElementById('ciBar');
            if (data.confidence_interval_95) {
                ciBar.style.display = 'block';
                document.getElementById('ciLow').textContent = data.confidence_interval_95.lower_formatted;
                document.getElementById('ciHigh').textContent = data.confidence_interval_95.upper_formatted;
                // Animate fill bar (position of predicted value within CI range)
                const low = data.confidence_interval_95.lower_usd;
                const high = data.confidence_interval_95.upper_usd;
                const pct = ((data.predicted_value_usd - low) / (high - low)) * 100;
                setTimeout(() => { document.getElementById('ciFill').style.width = Math.min(100, Math.max(5, pct)) + '%'; }, 100);
            } else {
                ciBar.style.display = 'none';
            }

            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        function showJSON() {
            const payload = {
                permit_type: buildPermitType(),
                zip_code: document.getElementById('zip_code').value,
                square_footage: parseFloat(document.getElementById('square_footage').value) || 0,
                occupancy_type: document.getElementById('occupancy_type').value,
                contractor_city: document.getElementById('contractor_city').value,
                issue_month: parseInt(document.getElementById('issue_month').value),
                issue_year: 2020
            };
            alert('POST /predict\\n\\n' + JSON.stringify(payload, null, 2));
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Landing page with API documentation."""
    return LANDING_HTML


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": "Two-Stage Hurdle Model",
        "stage1": "XGBoost Classifier (Acc=0.977, AUC=0.992)",
        "stage2": "XGBoost Regressor (R2=0.866)",
        "features": len(FEATURE_COLS),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict permit value using the Two-Stage Hurdle Model.
    
    Input: JSON with permit details
    Output: JSON with predicted value, stage results, and confidence info
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
        
        # Validate required fields
        required = ["permit_type"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        
        # Build feature vector (63 features for Stage 1)
        feature_vector, parsed_permit = build_feature_vector(data)
        X_cls = feature_vector.reshape(1, -1)
        
        # Build Stage 2 feature vector (62 features — excludes area_value_ratio)
        X_reg = np.array([feature_vector[FEATURE_COLS.index(c)] for c in FEATURE_COLS_REG], dtype=np.float64).reshape(1, -1)
        
        # Stage 1: Classify zero vs. positive
        stage1_pred = int(stage1_clf.predict(X_cls)[0])
        stage1_prob = float(stage1_clf.predict_proba(X_cls)[0][1])  # P(zero-value)
        
        if stage1_pred == 1:
            # Administrative / zero-value permit
            result = {
                "predicted_value_usd": 0.0,
                "predicted_value_formatted": "$0",
                "permit_class": "administrative",
                "permit_class_description": "This permit is predicted to be an administrative/zero-value permit (e.g., sign installation, demolition, barricade).",
                "confidence": round((stage1_prob) * 100, 1),
                "stage1_result": {
                    "prediction": "zero-value",
                    "probability_zero": round(stage1_prob, 4),
                    "probability_positive": round(1 - stage1_prob, 4),
                },
                "stage2_result": None,
                "parsed_input": {
                    "permit_category": parsed_permit["permit_category"],
                    "sub_category": parsed_permit["sub_category"],
                    "work_type": parsed_permit["work_type"],
                    "permit_code": parsed_permit["permit_code"],
                    "zip_code": data.get("zip_code", "N/A"),
                    "square_footage": data.get("square_footage", 0),
                    "occupancy_type": data.get("occupancy_type", "N/A"),
                }
            }
        else:
            # Positive-value construction permit — run Stage 2 with 62-feature vector
            log_value_pred = float(stage2_reg.predict(X_reg)[0])
            value_usd = float(np.exp(log_value_pred))
            
            # Confidence band (approximate using training RMSE = 0.685)
            rmse = 0.685
            lower_bound = float(np.exp(log_value_pred - 1.96 * rmse))
            upper_bound = float(np.exp(log_value_pred + 1.96 * rmse))
            
            result = {
                "predicted_value_usd": round(value_usd, 2),
                "predicted_value_formatted": f"${value_usd:,.0f}",
                "permit_class": "construction",
                "permit_class_description": "This permit is predicted to be a positive-value construction project.",
                "confidence_interval_95": {
                    "lower_usd": round(lower_bound, 2),
                    "upper_usd": round(upper_bound, 2),
                    "lower_formatted": f"${lower_bound:,.0f}",
                    "upper_formatted": f"${upper_bound:,.0f}",
                },
                "stage1_result": {
                    "prediction": "positive-value",
                    "probability_zero": round(stage1_prob, 4),
                    "probability_positive": round(1 - stage1_prob, 4),
                },
                "stage2_result": {
                    "log_value_predicted": round(log_value_pred, 4),
                    "value_usd": round(value_usd, 2),
                },
                "parsed_input": {
                    "permit_category": parsed_permit["permit_category"],
                    "sub_category": parsed_permit["sub_category"],
                    "work_type": parsed_permit["work_type"],
                    "permit_code": parsed_permit["permit_code"],
                    "zip_code": data.get("zip_code", "N/A"),
                    "square_footage": data.get("square_footage", 0),
                    "occupancy_type": data.get("occupancy_type", "N/A"),
                }
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    print(f"\n  Starting Dallas Permit Value Predictor API on port {port}...")
    print("  Endpoints:")
    print("    GET  /        - API info page")
    print("    GET  /health  - Health check")
    print("    POST /predict - Predict permit value")
    print()
    app.run(host="0.0.0.0", port=port, debug=debug)

