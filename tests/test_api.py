import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Make the app module importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_prediction_endpoint():
    # Test payload with realistic credit risk inputs
    payload = {
        "util_ratio": 0.45,
        "age": 35,
        "late_30_59": 0,
        "late_60_89": 0,
        "late_90_plus": 0,
        "debt_ratio": 0.35,
        "monthly_income": 6500.0,
        "open_credit_lines": 8,
        "real_estate_loans": 1,
        "dependents": 2.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "probability" in data
    assert "verdict" in data
    assert "risk_level" in data
    assert "top_reasons" in data
    assert "roadmap_to_yes" in data
    
    # Assert probability is a valid percentage (0.0 to 1.0)
    assert 0.0 <= data["probability"] <= 1.0
    # Assert verdict is business logic-friendly
    assert data["verdict"] in ["Approved", "Declined"]
    assert data["risk_level"] in ["Low Risk", "High Risk"]
    
    # Assert SHAP reasons contains top reasons (up to 5)
    assert len(data["top_reasons"]) > 0
    assert "feature" in data["top_reasons"][0]
    assert "impact" in data["top_reasons"][0]
    
    print("API prediction test passed successfully!")
    print("API Response:", data)
