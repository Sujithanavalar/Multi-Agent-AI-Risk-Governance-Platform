import time
from typing import Dict, Any, Tuple, List

class AgentState:
    def __init__(self):
        self.action_history: List[Dict[str, Any]] = []

class ContextScorer:
    def __init__(self):
        # In a real distributed system, this would be backed by Redis
        self.agent_states: Dict[str, AgentState] = {}

    def evaluate(self, agent_id: str, action_type: str, payload: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluates sequential context risk.
        Returns:
            Tuple of (risk_score [0.0 to 1.0], reason)
        """
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = AgentState()
            
        state = self.agent_states[agent_id]
        
        # Log this action with timestamp
        current_time = time.time()
        state.action_history.append({
            "action": action_type,
            "timestamp": current_time
        })
        
        # Prune history older than 60 seconds
        state.action_history = [
            a for a in state.action_history 
            if current_time - a["timestamp"] < 60
        ]
        
        # ── Context Rule 1: High Velocity ──
        # More than 10 actions in 60 seconds is highly suspicious
        if len(state.action_history) >= 10:
            return 0.8, f"High velocity: {len(state.action_history)} actions in last 60 seconds"
            
        # ── Context Rule 2: Rapid Repetitive Reads ──
        # Catching data exfiltration attempts
        read_actions = [a for a in state.action_history if "READ" in str(a["action"]).upper()]
        if len(read_actions) >= 5:
            return 0.6, "Suspicious sequence: 5+ consecutive READ actions detected"
            
        return 0.1, "Action sequence context is normal"
