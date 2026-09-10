"""Test the Flask API with 3 scenarios."""
import urllib.request
import json

BASE = "http://127.0.0.1:5000"

tests = [
    {
        "name": "High-Value: Commercial New Construction",
        "data": {
            "permit_type": "Building (BU) Commercial New Construction",
            "zip_code": "75201",
            "square_footage": 25000,
            "occupancy_type": "COMMERCIAL OFFICE BUILDING",
            "contractor_name": "ROGERS OBRIEN CONSTRUCTION",
            "contractor_city": "DALLAS",
            "issue_month": 6,
            "issue_year": 2024
        }
    },
    {
        "name": "Mid-Value: Single Family Renovation",
        "data": {
            "permit_type": "Building (BU) Single Family Renovation",
            "zip_code": "75228",
            "square_footage": 1800,
            "occupancy_type": "SINGLE FAMILY DWELLING",
            "contractor_city": "DALLAS",
            "issue_month": 3,
            "issue_year": 2024
        }
    },
    {
        "name": "Zero-Value: Electrical Sign New Construction",
        "data": {
            "permit_type": "Electrical Sign New Construction",
            "zip_code": "75247",
            "square_footage": 0,
            "occupancy_type": "COMMERCIAL RETAIL",
            "issue_month": 9,
            "issue_year": 2024
        }
    },
]

for test in tests:
    print(f"\n{'='*60}")
    print(f"TEST: {test['name']}")
    print(f"{'='*60}")
    
    payload = json.dumps(test["data"]).encode()
    req = urllib.request.Request(
        f"{BASE}/predict",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"ERROR: {e}")
        if hasattr(e, 'read'):
            print(e.read().decode())
