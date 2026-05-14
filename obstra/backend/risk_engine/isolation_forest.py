import numpy as np
from sklearn.ensemble import IsolationForest
import random
from typing import Dict, Any, Tuple

class MLAnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False
        self._train_synthetic_baseline()

    def _train_synthetic_baseline(self):
        """
        Train the model on 'normal' synthetic agent behavior.
        Features: [payload_size, parameter_count]
        """
        print("[Risk Engine] Training Isolation Forest on synthetic normal data...")
        
        # Generate 1000 normal actions
        # Payload size normally between 100 and 1000 bytes
        # Parameter count normally between 1 and 5
        X_train = []
        for _ in range(1000):
            payload_size = random.normalvariate(500, 150)
            param_count = random.randint(1, 5)
            X_train.append([payload_size, param_count])
            
        # Add a few known anomalies so it learns boundaries
        # Huge payload
        X_train.append([15000, 2])
        # Too many parameters
        X_train.append([500, 50])
            
        self.model.fit(np.array(X_train))
        self.is_trained = True
        print("[Risk Engine] Isolation Forest training complete.")

    def _extract_features(self, payload: Dict[str, Any]) -> list:
        # Approximate payload size by string length
        payload_size = len(str(payload))
        param_count = len(payload.keys())
        return [payload_size, param_count]

    def evaluate(self, action_type: str, payload: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluates the action against the Isolation Forest.
        Returns:
            Tuple of (risk_score [0.0 to 1.0], reason)
        """
        if not self.is_trained:
            return 0.0, "Model not trained"
            
        features = np.array([self._extract_features(payload)])
        
        # decision_function returns > 0 for normal, < 0 for anomaly
        # Typically between -0.5 and 0.5. We map this to 0-1 risk score.
        raw_score = self.model.decision_function(features)[0]
        
        # Normalise: more negative = more anomalous = higher risk
        # We invert it so higher score = higher risk
        risk_score = 0.5 - raw_score
        
        # Clip between 0 and 1
        risk_score = max(0.0, min(1.0, float(risk_score)))
        
        reason = "Behavior looks normal"
        if risk_score > 0.6:
            reason = f"Anomalous payload characteristics detected (score: {risk_score:.2f})"
            
        return round(risk_score, 2), reason
