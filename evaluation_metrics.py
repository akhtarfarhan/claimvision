import datetime
from typing import Dict, Any, Optional
from schemas import CVLLMIntegrationSchema, StructuredOutput
from pydantic import BaseModel


class GoldStandard(BaseModel):
    """Human-labeled ground truth for evaluation"""
    shipment_id: str
    expected_compensation_usd: float
    expected_claim_category: str

    @staticmethod
    def get_gold_standard(shipment_id: str) -> Optional['GoldStandard']:
        """Gold standard dataset – includes all test cases"""
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


def evaluate_claim_accuracy(generated_report: CVLLMIntegrationSchema) -> Dict[str, Any]:
    """
    Main evaluation function – returns all required metrics:
    - coverage_score
    - format_accuracy_ok
    - squared_error (for RMSE)
    - gold_compensation, predicted_compensation
    - category_match
    """
    results = {
        "evaluation_timestamp": datetime.datetime.now().isoformat()
    }

    # Basic safety check
    if not generated_report.llm_response:
        results["error"] = "LLM response missing"
        return results

    structured = generated_report.llm_response.structured
    gold = GoldStandard.get_gold_standard(generated_report.shipment_id)

    # 1. Coverage & Format Accuracy
    fields = structured.model_dump(exclude={'claim_id', 'confidence'})
    missing_fields = [k for k, v in fields.items() if v is None]
    total_fields = len(fields)
    results["coverage_score"] = 1.0 if total_fields == 0 else 1.0 - (len(missing_fields) / total_fields)
    results["format_accuracy_ok"] = len(missing_fields) == 0

    # 2. RMSE / Monetary Error (only if gold standard exists)
    if gold:
        actual = gold.expected_compensation_usd
        predicted = structured.suggested_compensation_usd
        squared_error = (predicted - actual) ** 2

        results["gold_compensation"] = actual
        results["predicted_compensation"] = predicted
        results["squared_error"] = round(squared_error, 2)
        results["category_match"] = (structured.claim_category == gold.expected_claim_category)
        results["gold_standard_available"] = True
    else:
        results["gold_standard_available"] = False

    return results


# Optional: Quick test when running this file directly
if __name__ == "__main__":
    print("eval_metrics.py loaded – evaluation framework ready!")
    print("Gold standards available for:")
    for sid in ["SHIP-TEST-HIGH", "SHIP-TEST-MED", "SHIP-LOWCONF-001"]:
        print(f"  → {sid}")