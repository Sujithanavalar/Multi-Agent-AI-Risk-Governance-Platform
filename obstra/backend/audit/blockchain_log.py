import hashlib
import json
from sqlalchemy.orm import Session
from db import models

class HashChainedAuditLog:
    def __init__(self):
        pass
        
    def _calculate_hash(self, previous_hash: str, payload_str: str) -> str:
        """
        Calculates SHA-256 hash of the previous hash combined with the current data payload.
        This creates the tamper-proof cryptographic chain.
        """
        combined = f"{previous_hash}{payload_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def secure_log(self, db: Session, action_log: models.ActionLog) -> models.ActionLog:
        """
        Secures an ActionLog entry by chaining its hash to the previous entry.
        """
        # 1. Fetch the absolute latest log entry to get the previous_hash
        # Use order_by(desc) and limit(1)
        last_entry = db.query(models.ActionLog).order_by(models.ActionLog.id.desc()).first()
        
        previous_hash = "GENESIS_BLOCK"
        if last_entry and last_entry.current_hash:
            previous_hash = last_entry.current_hash
            
        # 2. Prepare the deterministic payload string
        payload_data = {
            "agent_id": action_log.agent_id,
            "action": action_log.action_type,
            "score": action_log.final_risk_score,
            "status": action_log.status
        }
        # Serialize with sorted keys to guarantee deterministic string
        payload_str = json.dumps(payload_data, sort_keys=True)
        
        # 3. Calculate current hash
        current_hash = self._calculate_hash(previous_hash, payload_str)
        
        # 4. Attach to model
        action_log.previous_hash = previous_hash
        action_log.current_hash = current_hash
        
        # Save to DB
        db.add(action_log)
        db.commit()
        db.refresh(action_log)
        
        return action_log
