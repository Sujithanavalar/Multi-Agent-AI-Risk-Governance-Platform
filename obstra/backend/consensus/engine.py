import json
import uuid
from typing import Dict, Any, Optional
from db.redis_client import redis_client

class ConsensusEngine:
    def __init__(self):
        # 5 minutes timeout as per spec (300 seconds)
        self.TIMEOUT_SECONDS = 300

    def queue_action(self, agent_id: str, action_type: str, payload: Dict[str, Any], risk_score: float) -> str:
        """
        Puts a medium-risk action into the Redis queue for consensus review.
        Returns a unique consensus_id.
        """
        consensus_id = f"pending_{uuid.uuid4().hex}"
        
        data = {
            "agent_id": agent_id,
            "action_type": action_type,
            "payload": payload,
            "risk_score": risk_score,
            "status": "PENDING_REVIEW"
        }
        
        # Store in Redis with TTL. If it expires, it is auto-blocked.
        redis_client.set_ex(consensus_id, self.TIMEOUT_SECONDS, json.dumps(data))
        
        return consensus_id

    def resolve_action(self, consensus_id: str, approved: bool, reviewer: str, reason: str) -> Optional[Dict[str, Any]]:
        """
        Resolves a pending action.
        """
        data_str = redis_client.get(consensus_id)
        if not data_str:
            return None # Action expired or doesn't exist
            
        data = json.loads(data_str)
        data["status"] = "APPROVED" if approved else "BLOCKED"
        data["reviewer"] = reviewer
        data["review_reason"] = reason
        
        # Remove from pending queue
        redis_client.delete(consensus_id)
        
        return data
