from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from llm_service import generate_full_claim

app = FastAPI(title="ClaimVision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-claim")
async def generate_claim(
    shipment_id: Optional[str] = Form(None),
    tracking_id: Optional[str] = Form(None),
    carrier: Optional[str] = Form(None),
    route: Optional[str] = Form(None),
    expected_delivery: Optional[str] = Form(None),
    actual_delivery: Optional[str] = Form(None),
    incident_location: Optional[str] = Form(None),
    damage_severity: Optional[str] = Form(None),
    damage_types: Optional[str] = Form(None),
    damage_description: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None),
):
    claim_input = {
        "shipment_id": shipment_id or "Not provided",
        "tracking_id": tracking_id or "Not provided",
        "carrier": carrier or "Not provided",
        "route": route or "Not provided",
        "expected_delivery": expected_delivery or "Not provided",
        "actual_delivery": actual_delivery or "Not provided",
        "incident_location": incident_location or "Not provided",
        "damage_severity": damage_severity or "Not specified",
        "damage_types": damage_types or "Not specified",
        "damage_description": damage_description or "No description provided",
    }

    claim_text = generate_full_claim(claim_input)

    return {"claim_report": claim_text}