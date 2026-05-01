import json
import time
import random
import requests
from datetime import datetime

DETECTION_ENGINE_URL = "http://localhost:5001/decision"

# ── NORMAL DECISION TEMPLATES ──────────────────────────────────────────────
def generate_normal_decision(agent_id, step):
    return {
        "agent_id": agent_id,
        "decision_type": "deployment_approval",
        "scope": random.randint(100, 2000),
        "parameter_delta": round(random.uniform(0.01, 0.10), 3),
        "timing_offset": random.randint(5, 30),
        "payload_size": random.randint(512, 8192),
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "anomalous": False
    }

# ── ANOMALOUS DECISION TEMPLATES ───────────────────────────────────────────
def generate_anomalous_decision(agent_id, step):
    return {
        "agent_id": agent_id,
        "decision_type": "deployment_approval",
        "scope": random.randint(30000, 50000),
        "parameter_delta": round(random.uniform(0.80, 1.50), 3),
        "timing_offset": random.randint(1, 3),
        "payload_size": random.randint(90000, 120000),
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "anomalous": True
    }

# ── AGENT FUNCTIONS ────────────────────────────────────────────────────────
def agent_a(step, anomalous=False):
    print(f"[Agent A] Evaluating update — step {step}")
    decision = generate_anomalous_decision("agent_a", step) if anomalous else generate_normal_decision("agent_a", step)
    return decision

def agent_b(prev_decision, step, anomalous=False):
    print(f"[Agent B] Validating parameters — step {step}")
    decision = generate_anomalous_decision("agent_b", step) if anomalous else generate_normal_decision("agent_b", step)
    decision["prev_scope"] = prev_decision["scope"]
    return decision

def agent_c(prev_decision, step, anomalous=False):
    print(f"[Agent C] Approving deployment — step {step}")
    decision = generate_anomalous_decision("agent_c", step) if anomalous else generate_normal_decision("agent_c", step)
    decision["prev_scope"] = prev_decision["scope"]
    decision["final_approval"] = True
    return decision

# ── SEND TO DETECTION ENGINE ───────────────────────────────────────────────
def send_decision(decision):
    try:
        response = requests.post(DETECTION_ENGINE_URL, json=decision, timeout=5)
        result = response.json()
        status = result.get("status", "UNKNOWN")
        score = result.get("layer1_score", "N/A")
        print(f"  → LenZ: {status} | Score: {score}")
        return result
    except requests.exceptions.ConnectionError:
        print("  → Detection engine not running yet. Start engine first.")
        return None
    except Exception as e:
        print(f"  → Error: {e}")
        return None

# ── MAIN SIMULATION ────────────────────────────────────────────────────────
def run_simulation():
    print("=" * 60)
    print("LenZ — Agent Simulator Starting")
    print("=" * 60)

    # PHASE 1 — BASELINE (150 normal decisions)
    print("\n[BASELINE PHASE] Generating normal decisions for 150 steps...")
    print("LenZ is learning normal behavior.\n")

    for step in range(1, 151):
        decision_a = agent_a(step, anomalous=False)
        decision_b = agent_b(decision_a, step, anomalous=False)
        decision_c = agent_c(decision_b, step, anomalous=False)
        send_decision(decision_c)
        time.sleep(0.3)

        if step % 25 == 0:
            print(f"  [Baseline] {step}/150 decisions sent\n")

    print("\n[BASELINE COMPLETE] Switching to live monitoring...\n")
    time.sleep(2)

    # PHASE 2 — NORMAL OPERATIONS (20 normal decisions)
    print("[LIVE MONITORING] Normal operations running...\n")

    for step in range(151, 171):
        decision_a = agent_a(step, anomalous=False)
        decision_b = agent_b(decision_a, step, anomalous=False)
        decision_c = agent_c(decision_b, step, anomalous=False)
        send_decision(decision_c)
        time.sleep(1)

    print("\n[ANOMALY INJECTION] CrowdStrike pattern firing...\n")
    time.sleep(1)

    # PHASE 3 — ANOMALY INJECTION (10 anomalous decisions)
    for step in range(171, 181):
        decision_a = agent_a(step, anomalous=True)
        decision_b = agent_b(decision_a, step, anomalous=True)
        decision_c = agent_c(decision_b, step, anomalous=True)
        send_decision(decision_c)
        time.sleep(1)

    print("\n[SIMULATION COMPLETE]")

if __name__ == "__main__":
    run_simulation()
