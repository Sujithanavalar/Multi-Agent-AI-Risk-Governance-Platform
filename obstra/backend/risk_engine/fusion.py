from typing import Dict, Any, Tuple
from .rule_engine import RuleEngine
from .isolation_forest import MLAnomalyDetector
from .context_scorer import ContextScorer

class RiskFusionEngine:
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.ml_detector = MLAnomalyDetector()
        self.context_scorer = ContextScorer()

    def evaluate_action(self, agent_id: str, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs all 3 risk algorithms and fuses the scores.
        """
        # 1. Evaluate Rule Engine (Weight: 0.4)
        rule_score, rule_reason = self.rule_engine.evaluate(action_type, payload)
        
        # 2. Evaluate ML Anomaly (Weight: 0.4)
        ml_score, ml_reason = self.ml_detector.evaluate(action_type, payload)
        
        # 3. Evaluate Context (Weight: 0.2)
        ctx_score, ctx_reason = self.context_scorer.evaluate(agent_id, action_type, payload)
        
        # Fusion Formula
        final_score = (ml_score * 0.4) + (rule_score * 0.4) + (ctx_score * 0.2)
        
        # Determine Status
        # 0.0 - 0.3 -> APPROVED
        # 0.3 - 0.7 -> PENDING (Consensus required)
        # 0.7 - 1.0 -> BLOCKED
        
        # Hard Rule Override: If Rule Engine says 1.0, it's always blocked regardless of ML
        if rule_score == 1.0:
            final_score = 1.0
            
        status = "APPROVED"
        message = "Action is within safe parameters."
        
        if final_score >= 0.7:
            status = "BLOCKED"
            message = f"High Risk Detected. Primary driver: {rule_reason if rule_score >= ml_score else ml_reason}"
        elif final_score >= 0.3:
            status = "PENDING"
            message = "Medium Risk. Action pushed to consensus queue for review."

        return {
            "status": status,
            "final_score": round(final_score, 2),
            "breakdown": {
                "rule_engine": {"score": rule_score, "reason": rule_reason},
                "isolation_forest": {"score": ml_score, "reason": ml_reason},
                "context_scorer": {"score": ctx_score, "reason": ctx_reason}
            },
            "message": message
        }
