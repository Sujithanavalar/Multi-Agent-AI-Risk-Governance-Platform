from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

from db import models
from db.database import engine, get_db
from sqlalchemy.orm import Session
from agents import registry
from interceptor.middleware import TokenAuthMiddleware
from risk_engine.fusion import RiskFusionEngine
from consensus.engine import ConsensusEngine
from audit.blockchain_log import HashChainedAuditLog

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Obstra",
    description="Real-Time Multi-Agent AI Governance Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TokenAuthMiddleware)

risk_engine = RiskFusionEngine()
consensus_engine = ConsensusEngine()
audit_log = HashChainedAuditLog()

app.include_router(registry.router)

class ActionRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any]

class AgentRegisterRequestExtended(BaseModel):
    name: str
    framework: str
    owner: Optional[str] = None
    description: Optional[str] = None
    permissions: List[str] = []

class ConsensusRequest(BaseModel):
    consensus_id: str
    approved: bool
    reviewer: str
    reason: str

class ThreatResolveRequest(BaseModel):
    threat_id: int
    resolved: bool
    resolved_by: str
    reason: str

class PolicyCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    rules: Dict[str, Any]
    is_active: bool = True

@app.post("/agents/register-extended", tags=["Agents"], status_code=201)
def register_agent_extended(request: AgentRegisterRequestExtended, db: Session = Depends(get_db)):
    import secrets
    token = f"sm_{secrets.token_hex(16)}"
    
    new_agent = models.Agent(
        name=request.name,
        framework=request.framework,
        owner=request.owner,
        token=token,
        description=request.description,
        permissions=request.permissions
    )
    
    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)
    
    return new_agent

@app.get("/agents/{agent_id}", tags=["Agents"])
def get_agent(agent_id: int, db: Session = Depends(get_db)):
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.get("/agents/{agent_id}/actions", tags=["Agents"])
def get_agent_actions(agent_id: int, db: Session = Depends(get_db)):
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    actions = db.query(models.ActionLog).filter(models.ActionLog.agent_id == agent_id).order_by(models.ActionLog.timestamp.desc()).limit(20).all()
    return actions

@app.post("/actions/evaluate", tags=["Actions"])
async def evaluate_action(request: Request, action: ActionRequest, db: Session = Depends(get_db)):
    agent = request.state.agent
    
    evaluation = risk_engine.evaluate_action(
        agent_id=str(agent.id),
        action_type=action.action_type,
        payload=action.payload
    )
    
    consensus_id = None
    if evaluation["status"] == "PENDING":
        consensus_id = consensus_engine.queue_action(
            agent_id=str(agent.id),
            action_type=action.action_type,
            payload=action.payload,
            risk_score=evaluation["final_score"]
        )
    
    action_log = models.ActionLog(
        agent_id=agent.id,
        action_type=action.action_type,
        payload=action.payload,
        isolation_forest_score=evaluation["breakdown"]["isolation_forest"]["score"],
        rule_engine_score=evaluation["breakdown"]["rule_engine"]["score"],
        context_score=evaluation["breakdown"]["context_scorer"]["score"],
        final_risk_score=evaluation["final_score"],
        status=evaluation["status"],
        reason=evaluation["message"]
    )
    audit_log.secure_log(db, action_log)
    
    if evaluation["status"] == "BLOCKED":
        threat = models.Threat(
            agent_id=agent.id,
            action_log_id=action_log.id,
            threat_type="RISK_BLOCKED",
            severity="HIGH",
            description=evaluation["message"]
        )
        db.add(threat)
        db.commit()
    
    return {
        "status": evaluation["status"],
        "consensus_id": consensus_id,
        "agent_name": agent.name,
        "action_type": action.action_type,
        "risk_score": evaluation["final_score"],
        "message": evaluation["message"],
        "breakdown": evaluation["breakdown"]
    }

@app.post("/consensus/resolve", tags=["Consensus"])
def resolve_consensus(request: ConsensusRequest, db: Session = Depends(get_db)):
    resolved_data = consensus_engine.resolve_action(
        consensus_id=request.consensus_id,
        approved=request.approved,
        reviewer=request.reviewer,
        reason=request.reason
    )
    
    if not resolved_data:
        return {"error": "Action expired, already resolved, or not found"}
    
    archived = models.ArchivedAction(
        original_action_id=0,
        agent_id=int(resolved_data["agent_id"]),
        action_type=resolved_data["action_type"],
        payload=resolved_data["payload"],
        final_risk_score=resolved_data["risk_score"],
        status="PENDING",
        resolution="APPROVED" if request.approved else "BLOCKED",
        resolved_by=request.reviewer,
        reason=request.reason
    )
    db.add(archived)
    db.commit()
    
    return {"status": "SUCCESS", "resolved_data": resolved_data}

@app.get("/consensus/pending", tags=["Consensus"])
def get_pending_actions():
    from db.redis_client import redis_client
    import json
    
    pending = []
    if redis_client.use_mock:
        for k, v in redis_client.mock_store.items():
            data = json.loads(v["value"])
            data["consensus_id"] = k
            pending.append(data)
    else:
        for key in redis_client.client.scan_iter("pending_*"):
            data = json.loads(redis_client.client.get(key))
            data["consensus_id"] = key
            pending.append(data)
            
    return pending

@app.get("/audit/logs", tags=["Audit"])
def get_audit_logs(db: Session = Depends(get_db)):
    logs = db.query(models.ActionLog).order_by(models.ActionLog.id.desc()).limit(100).all()
    return logs

@app.get("/threats", tags=["Threats"])
def get_threats(db: Session = Depends(get_db), status: Optional[str] = "active"):
    query = db.query(models.Threat)
    if status:
        query = query.filter(models.Threat.status == status)
    threats = query.order_by(models.Threat.detected_at.desc()).all()
    return threats

@app.post("/threats/resolve", tags=["Threats"])
def resolve_threat(request: ThreatResolveRequest, db: Session = Depends(get_db)):
    threat = db.query(models.Threat).filter(models.Threat.id == request.threat_id).first()
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    threat.status = "resolved" if request.resolved else "active"
    threat.resolved_at = datetime.now() if request.resolved else None
    db.commit()
    
    return {"status": "SUCCESS", "threat": threat}

@app.get("/policies", tags=["Policies"])
def get_policies(db: Session = Depends(get_db)):
    policies = db.query(models.Policy).order_by(models.Policy.created_at.desc()).all()
    return policies

@app.post("/policies", tags=["Policies"], status_code=201)
def create_policy(request: PolicyCreateRequest, db: Session = Depends(get_db)):
    policy = models.Policy(
        name=request.name,
        description=request.description,
        rules=request.rules,
        is_active=request.is_active
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy

@app.get("/archived-actions", tags=["Archive"])
def get_archived_actions(db: Session = Depends(get_db)):
    archived = db.query(models.ArchivedAction).order_by(models.ArchivedAction.resolved_at.desc()).limit(100).all()
    return archived

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}
