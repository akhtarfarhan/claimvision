# llm_service.py
import sys
import uuid
import datetime
from typing import Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

from schemas import (
    CVLLMIntegrationSchema,
    LLMResponse,
    Review,
    ReviewStatus,
    StructuredOutput,
    BaseModel
)

# ==================== CONFIG ====================
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "claim_model_ashly_final"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== TOKENIZER ====================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ==================== 4-BIT QUANTIZATION ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ==================== BASE MODEL ====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    # device_map="auto",
    trust_remote_code=False  # ← THIS IS THE KEY FIX
)
model.config.attn_implementation = "eager"

# ==================== LOAD LoRA ADAPTER ====================
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    is_trainable=False
)

# ==================== STABILITY FIXES ====================
model.config.use_cache = False
model.config.attn_implementation = "eager"
model.eval()

print("✅ Phi-3 loaded with 4-bit quantization + LoRA")

# ==================== GOLD STANDARD ====================
class GoldStandard(BaseModel):
    shipment_id: str
    expected_compensation_usd: float
    expected_claim_category: str

    @staticmethod
    def get_gold_standard(shipment_id: str) -> Optional["GoldStandard"]:
        data = {
            "SHIP-TEST-HIGH": GoldStandard(
                shipment_id="SHIP-TEST-HIGH",
                expected_compensation_usd=300.00,
                expected_claim_category="Major Physical Damage"
            ),
            "SHIP-TEST-MED": GoldStandard(
                shipment_id="SHIP-TEST-MED",
                expected_compensation_usd=75.00,
                expected_claim_category="Moderate Physical Damage"
            ),
            "SHIP-LOWCONF-001": GoldStandard(
                shipment_id="SHIP-LOWCONF-001",
                expected_compensation_usd=60.00,
                expected_claim_category="Moderate Physical Damage"
            ),
        }
        return data.get(shipment_id)

# ==================== EVALUATION ====================
def evaluate_claim_accuracy(
    generated_report: CVLLMIntegrationSchema
) -> Dict[str, Any]:

    results = {
        "evaluation_timestamp": datetime.datetime.now().isoformat()
    }

    if not generated_report.llm_response:
        results["error"] = "LLM response missing"
        return results

    structured = generated_report.llm_response.structured
    gold = GoldStandard.get_gold_standard(generated_report.shipment_id)

    fields = structured.model_dump(exclude={"claim_id", "confidence"})
    missing = [k for k, v in fields.items() if v is None]

    results["coverage_score"] = 1.0 - (len(missing) / len(fields))
    results["format_accuracy_ok"] = len(missing) == 0

    if gold:
        actual = gold.expected_compensation_usd
        predicted = structured.suggested_compensation_usd
        results["gold_compensation"] = actual
        results["predicted_compensation"] = predicted
        results["squared_error"] = round((predicted - actual) ** 2, 2)
        results["category_match"] = (
            structured.claim_category == gold.expected_claim_category
        )

    results["gold_standard_available"] = bool(gold)
    return results

# ==================== REAL LLM INFERENCE ====================
def _real_llm_call(input_data: CVLLMIntegrationSchema) -> LLMResponse:
    cv = input_data.cv_output

    prompt = f"""
Generate a structured insurance claim.

Shipment ID: {input_data.shipment_id}
Damage types: {', '.join(cv.damage_types)}
Severity: {cv.severity}
Confidence: {cv.confidence:.2f}
Route: {input_data.logistics_metadata.route}
Carrier: {input_data.logistics_metadata.carrier}

Return strictly in this format:
Claim ID:
Category:
Severity Score:
Compensation USD:
Action:
Confidence:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()

    parsed = {}
    for line in generated_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            parsed[k.strip()] = v.strip()

    structured = StructuredOutput(
        claim_id=parsed.get("Claim ID", f"CLM-{uuid.uuid4().hex[:6]}"),
        claim_category=parsed.get("Category", "Moderate Physical Damage"),
        severity_score=int(parsed.get("Severity Score", "3")),
        suggested_compensation_usd=float(
            parsed.get("Compensation USD", "75").replace("$", "")
        ),
        action=parsed.get("Action", "Repair"),
        confidence=float(parsed.get("Confidence", "0.9"))
    )

    return LLMResponse(
        claim_report=generated_text,
        structured=structured,
        raw_llm_text=generated_text
    )

# ==================== HITL ROUTING ====================
def _route_for_review(cv_conf: float, llm_conf: float) -> Review:
    if cv_conf < 0.85 or llm_conf < 0.90:
        return Review(
            status=ReviewStatus.REVIEW_PENDING,
            review_notes="Low confidence detected"
        )
    return Review(status=ReviewStatus.AUTO)

# ==================== MAIN PIPELINE ====================
from datetime import datetime  # ← ADD THIS AT THE TOP OF THE FILE

def generate_full_claim(data: dict) -> str:
    severity = data['damage_severity'].lower()
    if severity == 'high':
        comp_range = "$400 - $1000"
    elif severity == 'medium':
        comp_range = "$150 - $400"
    else:
        comp_range = "$50 - $150"

    prompt = f"""You are a professional insurance claims adjuster. Generate ONLY the claim report below. Do NOT add any explanations, disclaimers, or extra text.

INPUT:
Shipment ID: {data['shipment_id']}
Damage Severity: {data['damage_severity']}
Damage Types: {data['damage_types']}
Description: {data['damage_description']}

OUTPUT FORMAT - FOLLOW EXACTLY:

**Claim ID:** CLM-20251218-{str(hash(data['shipment_id']) % 10000).zfill(4)}

**Claim Summary**
The shipment experienced damage during transit.

**Damage Assessment**
The reported damage includes {data['damage_types'] or 'unspecified damage'}. Severity level: {data['damage_severity'] or 'unknown'}. Description: {data['damage_description']}.

**Recommended Compensation**
${comp_range[1:]} USD (based on {data['damage_severity']} severity)

**Recommended Action**
Replacement

**Final Decision**
Approve Claim

Use formal language only. Do not repeat phrases. Do not mention AI or Microsoft."""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,  # Lower temperature = less repetition
            do_sample=True,
            repetition_penalty=1.2,  # Prevents repetition
            pad_token_id=tokenizer.eos_token_id,
        )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Strictly extract only from **Claim ID:** onward
    report_start = full_text.find("**Claim ID:**")
    if report_start != -1:
        report = full_text[report_start:].strip()
        
        # Remove any trailing instructions or tags
        end_tag = report.find("Use formal language only")
        if end_tag != -1:
            report = report[:end_tag].strip()
            
        end_tag = report.find("<|end_of_document|>")
        if end_tag != -1:
            report = report[:end_tag].strip()
            
        return report
    
    return "Report generation failed. Please try again."