import requests
from typing import Any, Dict, Optional

class ObstraSDK:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None

    def register(self, name: str, framework: str, owner: str = None) -> str:
        """Register the agent with Obstra and get a secure token."""
        url = f"{self.base_url}/agents/register"
        payload = {
            "name": name,
            "framework": framework,
            "owner": owner
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 201:
            data = response.json()
            self.token = data.get("token")
            print(f"[Obstra] Agent registered successfully. Token: {self.token[:10]}...")
            return self.token
        else:
            raise Exception(f"Failed to register agent: {response.text}")

    def log_action(self, action_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send an action to Obstra for risk evaluation."""
        if not self.token:
            raise Exception("Agent not registered. Call register() first.")
            
        url = f"{self.base_url}/actions/evaluate"
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        data = {
            "action_type": action_type,
            "payload": payload
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to evaluate action: {response.text}")

# --- FRAMEWORK INTEGRATION HOOKS ---

try:
    from langchain.callbacks.base import BaseCallbackHandler

    class ObstraLangChainCallback(BaseCallbackHandler):
        """
        Plug-and-play callback for LangChain agents.
        Usage: 
            agent.run(input, callbacks=[ObstraLangChainCallback(sdk)])
        """
        def __init__(self, sdk: ObstraSDK):
            self.sdk = sdk
            
        def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
            """Intercept whenever the LangChain agent tries to use a tool/action."""
            print(f"[Obstra Interceptor] LangChain Agent attempting to use tool: {serialized.get('name')}")
            
            # Send the intended action to Obstra for scoring BEFORE it executes
            decision = self.sdk.log_action(
                action_type=serialized.get("name", "unknown_tool"),
                payload={"input": input_str}
            )
            
            if decision.get("status") == "BLOCKED":
                raise Exception(f"Obstra blocked this action: {decision.get('message')}")
            
            print("[Obstra Interceptor] Action APPROVED. Proceeding...")

except ImportError:
    pass

class ObstraCrewAIHook:
    """
    Plugin for CrewAI agents. 
    CrewAI allows defining step_callbacks.
    Usage:
        Agent(..., step_callback=ObstraCrewAIHook(sdk).callback)
    """
    def __init__(self, sdk: ObstraSDK):
        self.sdk = sdk
        
    def callback(self, agent_output: Any):
        """Intercept the output of a CrewAI agent step."""
        print(f"[Obstra Interceptor] CrewAI Agent step completed. Evaluating...")
        
        # Log the step action
        decision = self.sdk.log_action(
            action_type="CREW_AI_STEP",
            payload={"output": str(agent_output)}
        )
        
        if decision.get("status") == "BLOCKED":
            print(f"WARNING: Obstra flagged this step as dangerous: {decision.get('message')}")
            # CrewAI doesn't easily stop execution via callbacks, but we can raise an error
            raise Exception("Obstra Security Exception: Dangerous action blocked.")
