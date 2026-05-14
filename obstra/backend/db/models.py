from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Boolean, Text
from sqlalchemy.sql import func
from db.database import Base

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    framework = Column(String, nullable=False) # e.g., langchain, crewai
    owner = Column(String, nullable=True)
    token = Column(String, unique=True, index=True, nullable=False)
    registered_at = Column(DateTime(timezone=True), server_default=func.now())

class ActionLog(Base):
    __tablename__ = "action_logs"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    action_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=True)
    
    isolation_forest_score = Column(Float, nullable=True)
    rule_engine_score = Column(Float, nullable=True)
    context_score = Column(Float, nullable=True)
    final_risk_score = Column(Float, nullable=True)
    
    status = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    previous_hash = Column(String, nullable=True)
    current_hash = Column(String, nullable=True)

class Threat(Base):
    __tablename__ = "threats"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    action_log_id = Column(Integer, ForeignKey("action_logs.id"), nullable=True)
    threat_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="active")
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)

class Policy(Base):
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    rules = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ArchivedAction(Base):
    __tablename__ = "archived_actions"

    id = Column(Integer, primary_key=True, index=True)
    original_action_id = Column(Integer, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    action_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=True)
    final_risk_score = Column(Float, nullable=True)
    status = Column(String, nullable=False)
    resolution = Column(String, nullable=False)
    resolved_by = Column(String, nullable=True)
    resolved_at = Column(DateTime(timezone=True), server_default=func.now())
    reason = Column(Text, nullable=True)
