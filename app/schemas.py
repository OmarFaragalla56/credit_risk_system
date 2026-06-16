from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class BorrowerInput(BaseModel):
    """Data validation schema for incoming API requests."""
    util_ratio: float = Field(default=0.30, description="Revolving Utilization")
    age: int = Field(default=45, ge=18, description="Borrower Age")
    late_30_59: int = Field(default=0)
    late_60_89: int = Field(default=0)
    late_90_plus: int = Field(default=0)
    debt_ratio: float = Field(default=0.35)
    monthly_income: float = Field(default=5000.0)
    open_credit_lines: int = Field(default=5)
    real_estate_loans: int = Field(default=0)
    dependents: int = Field(default=0)

class PredictionResponse(BaseModel):
    """Data validation schema for outbound API responses."""
    probability: float
    verdict: str
    risk_level: str
    top_reasons: List[Dict[str, Any]]
    roadmap_to_yes: Optional[str] = None
