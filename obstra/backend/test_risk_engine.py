from agents.sdk import ObstraSDK
import time

# 1. Initialize the SDK
sdk = ObstraSDK(base_url="http://localhost:8001")

print("\n--- Registering Agent ---")
token = sdk.register(name="FinanceAgent", framework="crewai", owner="BankCorp")
print(f"Token: {token}")

# Helper to print results cleanly
def print_decision(action, response):
    print(f"\n[ACTION] {action}")
    print(f"  Status: {response['status']} (Score: {response['risk_score']})")
    print(f"  Reason: {response['message']}")

print("\n--- 1. Testing Normal Action ---")
resp = sdk.log_action("query_balance", {"account_id": "ACC-123", "parameters": 1})
print_decision("Query Balance", resp)

print("\n--- 2. Testing Hard Rule Violation (Transfer > 1 Crore) ---")
resp = sdk.log_action("transfer_funds", {"amount": 15000000, "to_account": "ACC-999"})
print_decision("Transfer 1.5 Crore", resp)

print("\n--- 3. Testing ML Anomaly (Huge Payload) ---")
huge_payload = {f"key_{i}": "x" * 100 for i in range(200)}
resp = sdk.log_action("batch_export", huge_payload)
print_decision("Batch Export (Huge Payload)", resp)

print("\n--- 4. Testing Context Scorer (5+ Rapid READs) ---")
for i in range(6):
    resp = sdk.log_action(f"READ_FILE_{i}", {"file": f"secret_doc_{i}.pdf"})
    print(f"  READ {i} -> Score: {resp['risk_score']}")
    
print("\nFinal READ attempt should be blocked/pending:")
print_decision("Final Rapid READ", resp)
