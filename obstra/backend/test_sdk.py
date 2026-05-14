from agents.sdk import ObstraSDK

# 1. Initialize the SDK
sdk = ObstraSDK(base_url="http://localhost:8001")

print("Registering a new LangChain agent...")
token = sdk.register(name="ClinicalDiagnosisAgent", framework="langchain", owner="HospitalA")
print(f"Received Token: {token}")

print("\nSending a test action for evaluation...")
action_payload = {
    "patient_id": "P-12345",
    "action": "read_medical_record",
    "reason": "routine checkup"
}
response = sdk.log_action(action_type="DATA_READ", payload=action_payload)
print("Response from Obstra:", response)
