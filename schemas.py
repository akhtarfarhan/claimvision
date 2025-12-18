from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

# ==================== ENUMS for Validation ====================

class Severity(str, Enum):
    """Damage severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DamageType(str, Enum):
    """Types of damage detected by CV model"""
    DENT = "dent"
    CRACK = "crack"
    SCRATCH = "scratch"
    BROKEN = "broken"
    TORN_TAPE = "torn_tape"
    OTHER = "other"

class ReviewStatus(str, Enum):
    """Review workflow status"""
    AUTO = "auto"
    REVIEW_PENDING = "review_pending"
    APPROVED = "approved"
    REJECTED = "rejected"

# ==================== SUB-SCHEMAS ====================

class DamageBox(BaseModel):
    """Bounding box coordinates (normalized 0-1)"""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)

class CVOutput(BaseModel):
    """Computer Vision model output"""
    image_id: str
    damage_types: List[DamageType]
    damage_boxes: List[DamageBox]
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)

class LogisticsMetadata(BaseModel):
    """Logistics and shipment metadata"""
    route: str
    carrier: str
    expected_delivery: datetime
    actual_delivery: datetime
    
    @field_validator('actual_delivery')
    @classmethod
    def validate_delivery_times(cls, v, info):
        """Ensure actual delivery is after expected delivery"""
        values = info.data
        if 'expected_delivery' in values and v < values['expected_delivery']:
            raise ValueError('Actual delivery cannot be before expected delivery')
        return v

class LLMRequest(BaseModel):
    """LLM generation parameters"""
    prompt_template_id: str = Field(default="claim-template-v1")
    max_tokens: int = Field(default=256, ge=1, le=1024)

class StructuredOutput(BaseModel):
    """Structured LLM output"""
    claim_id: str
    claim_category: str
    severity_score: int = Field(..., ge=1, le=5)
    suggested_compensation_usd: float = Field(..., ge=0.0)
    action: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class LLMResponse(BaseModel):
    """Complete LLM response"""
    claim_report: str
    structured: StructuredOutput
    raw_llm_text: str

class Review(BaseModel):
    """Human review information"""
    status: ReviewStatus = Field(default=ReviewStatus.AUTO)
    reviewer_id: Optional[str] = Field(default=None)
    review_notes: Optional[str] = Field(default=None)
    finalized_at: Optional[datetime] = Field(default=None)
    
    @field_validator('status')
    @classmethod
    def validate_review_fields(cls, v, info):
        """Validate review fields based on status"""
        values = info.data
        if v == ReviewStatus.APPROVED and not values.get('finalized_at'):
            raise ValueError('finalized_at is required when status is "approved"')
        if v == ReviewStatus.REJECTED and not values.get('review_notes'):
            raise ValueError('review_notes are required when status is "rejected"')
        return v

# ==================== MAIN SCHEMAS ====================

class CVLLMIntegrationSchema(BaseModel):
    """Unified schema for the end-to-end claim generation system."""
    shipment_id: str
    vendor_id: str
    tracking_id: str
    timestamp: datetime
    location: str
    cv_output: CVOutput
    logistics_metadata: LogisticsMetadata
    llm_request: LLMRequest = Field(default_factory=LLMRequest)
    llm_response: Optional[LLMResponse] = Field(default=None)
    review: Optional[Review] = Field(default_factory=Review)
    
    model_config = ConfigDict(extra="allow")

# ==================== HELPER FUNCTIONS ====================

def generate_sample_json() -> Dict[str, Any]:
    """Generate a sample JSON matching the schema for testing."""
    now = datetime.now()
    return {
        "shipment_id": "SHIP-2025-00123",
        "vendor_id": "VEND-45678",
        "tracking_id": "TRK-789012",
        "timestamp": now.isoformat(),
        "location": "Warehouse A",
        "cv_output": {
            "image_id": "img_001",
            "damage_types": ["dent", "crack"],
            "damage_boxes": [{"x": 0.12, "y": 0.34, "w": 0.2, "h": 0.1}],
            "severity": "medium",
            "confidence": 0.87
        },
        "logistics_metadata": {
            "route": "NYC -> LA",
            "carrier": "UPS",
            "expected_delivery": (now - timedelta(days=1)).isoformat(),
            "actual_delivery": now.isoformat()
        },
        "llm_request": {
            "prompt_template_id": "claim-template-v1",
            "max_tokens": 256
        }
    }