from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session
from db.database import SessionLocal
from db import models

class TokenAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # We only want to intercept action evaluation endpoints
        if request.url.path.startswith("/actions/evaluate"):
            auth_header = request.headers.get("Authorization")
            
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid Authorization header"}
                )
                
            token = auth_header.split(" ")[1]
            
            # Verify token
            db: Session = SessionLocal()
            try:
                agent = db.query(models.Agent).filter(models.Agent.token == token).first()
                if not agent:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid agent token"}
                    )
                if agent.status != "active":
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Agent is suspended"}
                    )
                
                # Store agent in request state for the route handler
                request.state.agent = agent
            finally:
                db.close()
                
        # Proceed with the request
        response = await call_next(request)
        return response
