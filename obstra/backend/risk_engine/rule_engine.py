from typing import Dict, Any, Tuple

class RuleEngine:
    def __init__(self):
        pass

    def evaluate(self, action_type: str, payload: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluates deterministic hard rules.
        Returns:
            Tuple of (risk_score [0.0 to 1.0], reason)
        """
        # Rule 1: Dosage limit (Healthcare demo)
        if "dosage" in payload:
            try:
                dosage = float(payload["dosage"].replace("mg", ""))
                if dosage > 2000:
                    return 1.0, "Dosage exceeds maximum safe limit of 2000mg"
            except (ValueError, AttributeError):
                pass
        
        # Rule 2: High-value financial transfer
        if action_type == "transfer_funds" and "amount" in payload:
            try:
                amount = float(payload["amount"])
                if amount > 10000000: # 1 Crore
                    return 1.0, "Transfer amount > 1 Crore blocked"
                if amount > 1000000: # 10 Lakhs
                    return 0.7, "Transfer amount > 10 Lakhs requires review"
            except ValueError:
                pass
        
        # Rule 3: Database DELETE without WHERE
        if action_type == "execute_sql" and "query" in payload:
            query = str(payload["query"]).upper()
            if "DELETE FROM" in query and "WHERE" not in query:
                return 1.0, "Destructive database operation (DELETE without WHERE) detected"

        # Safe fallback
        return 0.0, "Passed all deterministic hard rules"
