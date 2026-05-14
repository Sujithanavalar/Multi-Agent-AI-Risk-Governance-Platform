# Obstra: Real-Time Multi-Agent AI Governance Platform

This plan outlines the architectural shift from a standalone simulation project to a production-ready middleware (Obstra) that connects to real-world AI agents (built using LangChain, CrewAI, AutoGen, etc.). This ensures Obstra functions as a true governance layer for AI systems in regulated industries.

## User Review Required

> [!IMPORTANT]
> The current project code (`engine/detection_engine.py`, etc.) is based on an older "LenZ" prototype using Flask and WebSockets. We will be transitioning to a modern stack (FastAPI) as requested, which means replacing the existing codebase entirely. Please confirm if it's okay to start fresh in the `obstra/` directory while keeping the old files around for reference.

## Open Questions

> [!WARNING]
> 1. Which database would you like to use for the initial local development of the registry? PostgreSQL (as planned for the full scale) or SQLite for now to quickly test Module 1?
> 2. For the SDK integration with LangChain/CrewAI, would you prefer us to build the SDK hooks as a Python package that can be imported (e.g., `import obstra.sdk`), or a REST API that agents ping directly?

## Proposed Changes

We will begin with **Module 1: Agent Registry + FastAPI skeleton**.

### Obstra Core Backend (Module 1)

#### [NEW] obstra/backend/main.py
- The core FastAPI application instance.
- Sets up routing and initialization.
- Includes a basic health-check endpoint.

#### [NEW] obstra/backend/agents/registry.py
- Handles agent registration logic.
- Generates unique authentication tokens for external AI agents when they register with Obstra.
- Manages permissions and agent metadata (e.g., framework used, allowed actions).

#### [NEW] obstra/backend/agents/sdk.py
- The Python SDK that developers using LangChain or CrewAI will import.
- Provides functions like `register_agent()` and `log_action()` to communicate with the Obstra backend.

#### [NEW] obstra/backend/interceptor/middleware.py
- FastAPI middleware to intercept all HTTP actions.
- Validates the agent's token before passing the action to the risk engine (to be implemented in Module 2).

#### [NEW] obstra/backend/db/models.py
- SQLAlchemy models for the Agent Registry (e.g., `Agent`, `ActionLog`).

#### [NEW] obstra/backend/requirements.txt
- Updated dependencies for the new stack (FastAPI, Uvicorn, SQLAlchemy, Pydantic, etc.).

## Verification Plan

### Automated Tests
- Run the FastAPI server (`uvicorn obstra.backend.main:app --reload`).
- Execute a test script to simulate a LangChain agent registering itself via the new SDK and receiving a token.

### Manual Verification
- Manually test the `/register` endpoint using curl or Postman to ensure tokens are correctly issued and stored in the database.
