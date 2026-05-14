import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import engine, SessionLocal
from db import models
from sqlalchemy.orm import Session
import secrets

def create_test_data():
    db = SessionLocal()
    try:
        # Create test agents (only original columns)
        agent1 = models.Agent(
            name="DiagnosisAgent",
            framework="langchain",
            owner="HospitalA",
            token=f"sm_{secrets.token_hex(16)}"
        )
        agent2 = models.Agent(
            name="PrescriptionAgent",
            framework="crewai",
            owner="HospitalA",
            token=f"sm_{secrets.token_hex(16)}"
        )
        
        db.add(agent1)
        db.add(agent2)
        db.commit()
        db.refresh(agent1)
        db.refresh(agent2)
        
        # Create test action logs
        log1 = models.ActionLog(
            agent_id=agent1.id,
            action_type="READ_FILE",
            payload={"file": "patient_123.pdf"},
            isolation_forest_score=0.12,
            rule_engine_score=0.1,
            context_score=0.08,
            final_risk_score=0.10,
            status="APPROVED",
            reason="Low risk - normal file read"
        )
        
        log2 = models.ActionLog(
            agent_id=agent2.id,
            action_type="DOSAGE",
            payload={"dosage": 3000, "patient": "patient_123"},
            isolation_forest_score=0.85,
            rule_engine_score=0.92,
            context_score=0.88,
            final_risk_score=0.88,
            status="BLOCKED",
            reason="Dosage exceeds maximum allowed limit"
        )
        
        log3 = models.ActionLog(
            agent_id=agent1.id,
            action_type="MODIFY_PERMISSIONS",
            payload={"target": "agent_456", "permissions": ["ADMIN"]},
            isolation_forest_score=0.72,
            rule_engine_score=0.78,
            context_score=0.65,
            final_risk_score=0.72,
            status="PENDING",
            reason="Suspicious permission change request"
        )
        
        db.add(log1)
        db.add(log2)
        db.add(log3)
        db.commit()
        
        print("Test data created successfully!")
        print(f"Agents created: {agent1.id}, {agent2.id}")
        print(f"Logs created: {log1.id}, {log2.id}, {log3.id}")
        
    except Exception as e:
        print(f"Error creating test data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_data()
