from fastapi import FastAPI, Form, UploadFile, File
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware

# 🔥 Import your LLM function
from llm_service import generate_full_claim

app = FastAPI(title="ClaimVision API")

# ✅ Allow frontend to talk to backend
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

    # 👇 IMAGES ARE OPTIONAL (FIX)
    images: Optional[List[UploadFile]] = File(None),
):
    """
    Generates a damage claim using LLM.
    Images are OPTIONAL.
    """

    # Prepare structured input for LLM
    claim_input = {
        "shipment_id": shipment_id,
        "tracking_id": tracking_id,
        "carrier": carrier,
        "route": route,
        "expected_delivery": expected_delivery,
        "actual_delivery": actual_delivery,
        "incident_location": incident_location,
        "damage_severity": damage_severity,
        "damage_types": damage_types,
        "damage_description": damage_description,
    }

    # 🔥 Call Phi-3 (already loaded in llm_service)
    claim_text = generate_full_claim(claim_input)

    return {
        "status": "success",
        "generated_claim": claim_text
    }
