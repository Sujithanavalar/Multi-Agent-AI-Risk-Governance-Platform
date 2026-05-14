from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
import secrets
from typing import List

from db.database import get_db
from db import models

router = APIRouter(prefix="/agents", tags=["Agents"])

class AgentRegisterRequest(BaseModel):
    name: str
    framework: str
    owner: str | None = None

class AgentResponse(BaseModel):
    id: int
    name: str
    framework: str
    owner: str | None = None
    token: str
    status: str

    class Config:
        from_attributes = True

@router.post("/register", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def register_agent(request: AgentRegisterRequest, db: Session = Depends(get_db)):
    # Generate a unique secure token for the agent
    token = f"sm_{secrets.token_hex(16)}"
    
    new_agent = models.Agent(
        name=request.name,
        framework=request.framework,
        owner=request.owner,
        token=token
    )
    
    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)
    
    return new_agent

@router.get("/", response_model=List[AgentResponse])
def list_agents(db: Session = Depends(get_db)):
    agents = db.query(models.Agent).all()
    return agents
