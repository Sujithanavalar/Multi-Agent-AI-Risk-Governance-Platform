import requests
import json
import sqlite3
import os

BASE_URL = "http://localhost:8001"

print("\n--- 1. Testing Consensus Engine (Medium Risk Action) ---")
# Simulate a PENDING action (5+ READs to trigger context scorer)
from agents.sdk import ObstraSDK
sdk = ObstraSDK(base_url=BASE_URL)
sdk.register("ComplianceBot", "langchain", "AuditDept")

for i in range(5):
    sdk.log_action(f"READ_FILE", {"file": "patient_data.pdf"})

resp = sdk.log_action("READ_FILE", {"file": "patient_data.pdf"})
print("Response:", resp)
consensus_id = resp.get("consensus_id")
print(f"Action queued with Consensus ID: {consensus_id}")

if consensus_id:
    print("\n--- 2. Operator Approving Action ---")
    approve_payload = {
        "consensus_id": consensus_id,
        "approved": True,
        "reviewer": "Human_Operator_1",
        "reason": "Verified it's a routine audit sweep"
    }
    app_resp = requests.post(f"{BASE_URL}/consensus/resolve", json=approve_payload)
    print("Approval Response:", app_resp.json())
else:
    print("Failed to get consensus_id.")

print("\n--- 3. Verifying Hash-Chained Audit Log ---")
# We'll connect directly to the SQLite database to verify the hashes
DB_PATH = os.path.join(os.path.dirname(__file__), "obstra.db")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT id, action_type, status, previous_hash, current_hash FROM action_logs ORDER BY id DESC LIMIT 5")
rows = cursor.fetchall()
for row in reversed(rows):
    print(f"Log ID: {row[0]} | Action: {row[1]} | Status: {row[2]}")
    print(f"  Prev Hash: {row[3][:16]}... | Curr Hash: {row[4][:16]}...")

conn.close()
