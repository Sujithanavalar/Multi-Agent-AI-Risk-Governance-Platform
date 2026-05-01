import json
import sqlite3
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import jsonschema
import networkx as nx

app = Flask(__name__)
app.config['SECRET_KEY'] = 'LenZ2026'
socketio = SocketIO(app, cors_allowed_origins="*")

# ── PATHS ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "lenz_audit.db")
os.makedirs(DATA_DIR, exist_ok=True)
_OLD_DB_PATH = os.path.join(DATA_DIR, "safeagent_audit.db")
if os.path.exists(_OLD_DB_PATH) and not os.path.exists(DB_PATH):
    try:
        os.rename(_OLD_DB_PATH, DB_PATH)
        print("[DB] Migrated safeagent_audit.db → lenz_audit.db")
    except Exception as e:
        print(f"[DB] Migration failed: {e}")

# ── STATE ──────────────────────────────────────────────────────────────────
baseline_data     = []
decision_log      = []
model             = None
baseline_complete = False
baseline_mean     = {}
BASELINE_SIZE     = 150
ANOMALY_THRESHOLD = 0.65
TRAJECTORY_WINDOW = 10
pending_decision  = None

# ── SCHEMA FOR LAYER 2 ─────────────────────────────────────────────────────
DECISION_SCHEMA = {
    "type": "object",
    "required": ["scope", "parameter_delta", "payload_size", "timing_offset"],
    "properties": {
        "scope":           {"type": "number", "minimum": 0,   "maximum": 5000},
        "parameter_delta": {"type": "number", "minimum": 0,   "maximum": 0.15},
        "payload_size":    {"type": "number", "minimum": 512, "maximum": 8192},
        "timing_offset":   {"type": "number", "minimum": 0,   "maximum": 60},
    }
}

# ── COMMUNICATION GRAPH ────────────────────────────────────────────────────
comm_graph = nx.DiGraph()
comm_graph.add_nodes_from(["Agent A", "Agent B", "Agent C"])

# ── DATABASE SETUP ─────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS decisions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            agent_id         TEXT,
            scope            REAL,
            parameter_delta  REAL,
            payload_size     REAL,
            timing_offset    REAL,
            layer1_score     REAL,
            layer1_status    TEXT,
            layer2_status    TEXT,
            layer2_detail    TEXT,
            layer3_status    TEXT,
            layer3_detail    TEXT,
            final_status     TEXT,
            operator_action  TEXT,
            operator_reason  TEXT,
            action_timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] Audit database initialised.")

def log_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO decisions (
            timestamp, agent_id, scope, parameter_delta,
            payload_size, timing_offset, layer1_score,
            layer1_status, layer2_status, layer2_detail,
            layer3_status, layer3_detail, final_status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        data["timestamp"], data["agent_id"], data["scope"],
        data["parameter_delta"], data["payload_size"], data["timing_offset"],
        data["layer1_score"], data["layer1_status"],
        data["layer2_status"], data["layer2_detail"],
        data["layer3_status"], data["layer3_detail"],
        data["final_status"]
    ))
    conn.commit()
    last_id = c.lastrowid
    conn.close()
    return last_id

def update_operator_action(decision_id, action, reason):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE decisions
        SET operator_action=?, operator_reason=?, action_timestamp=?
        WHERE id=?
    ''', (action, reason, datetime.now().isoformat(), decision_id))
    conn.commit()
    conn.close()

# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────
def extract_features(decision):
    return [
        float(decision.get("scope", 0)),
        float(decision.get("parameter_delta", 0)),
        float(decision.get("timing_offset", 0)),
        float(decision.get("payload_size", 0)),
    ]

# ── LAYER 1 — ISOLATION FOREST ─────────────────────────────────────────────
def layer1_score(decision):
    global model
    if model is None:
        return 0.0, "NO_MODEL"
    features = np.array([extract_features(decision)])
    raw_score = model.decision_function(features)[0]
    # Normalise: more negative = more anomalous → map to 0–1
    normalised = float(1 - (raw_score - (-0.5)) / (0.5 - (-0.5)))
    normalised = max(0.0, min(1.0, normalised))
    status = "FLAGGED" if normalised >= ANOMALY_THRESHOLD else "PASS"
    return round(normalised, 3), status

# ── LAYER 2 — SCHEMA + GRAPH ───────────────────────────────────────────────
def layer2_validate(decision):
    violations = []
    try:
        jsonschema.validate(instance=decision, schema=DECISION_SCHEMA)
    except jsonschema.ValidationError as e:
        violations.append(e.message)

    # Update communication graph
    edge_color = "red" if violations else "green"
    comm_graph.add_edge(
        "Agent B", "Agent C",
        color=edge_color,
        timestamp=decision.get("timestamp"),
        scope=decision.get("scope"),
        step=decision.get("step", 0)
    )

    if violations:
        return "INVALID", " | ".join(violations)
    return "VALID", "All fields within expected ranges"

# ── LAYER 3 — CHAIN TRAJECTORY ─────────────────────────────────────────────
def layer3_trajectory(decision):
    global decision_log, baseline_mean
    decision_log.append({
        "scope": decision.get("scope", 0),
        "parameter_delta": decision.get("parameter_delta", 0),
        "step": decision.get("step", 0)
    })

    if len(decision_log) < TRAJECTORY_WINDOW:
        return "SAFE", "Insufficient data for trajectory analysis"

    window = decision_log[-TRAJECTORY_WINDOW:]
    df = pd.DataFrame(window)
    rolling_scope_mean  = df["scope"].mean()
    rolling_param_mean  = df["parameter_delta"].mean()

    baseline_scope = baseline_mean.get("scope", 1000)
    baseline_param = baseline_mean.get("parameter_delta", 0.05)

    scope_ratio = rolling_scope_mean / baseline_scope if baseline_scope > 0 else 1
    param_ratio = rolling_param_mean / baseline_param if baseline_param > 0 else 1

    if scope_ratio >= 3.0 or param_ratio >= 5.0:
        detail = (f"Scope trajectory {scope_ratio:.1f}x baseline | "
                  f"Parameter trajectory {param_ratio:.1f}x baseline")
        return "UNSAFE", detail

    return "SAFE", f"Scope {scope_ratio:.2f}x baseline | Param {param_ratio:.2f}x baseline"

# ── MAIN DECISION ENDPOINT ─────────────────────────────────────────────────
@app.route('/decision', methods=['POST'])
def receive_decision():
    global baseline_data, model, baseline_complete, baseline_mean, pending_decision
    decision = request.get_json()

    # ── BASELINE PHASE ──────────────────────────────────────────────────────
    if not baseline_complete:
        baseline_data.append(extract_features(decision))
        socketio.emit('baseline_update', {
            "count": len(baseline_data),
            "total": BASELINE_SIZE
        })

        if len(baseline_data) >= BASELINE_SIZE:
            X = np.array(baseline_data)
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X)
            baseline_complete = True

            # Compute baseline means for Layer 3
            df_base = pd.DataFrame(baseline_data,
                columns=["scope","parameter_delta","timing_offset","payload_size"])
            baseline_mean = {
                "scope":           df_base["scope"].mean(),
                "parameter_delta": df_base["parameter_delta"].mean(),
            }

            print(f"\n[LENZ] Baseline complete. Model trained on {BASELINE_SIZE} decisions.")
            print(f"[LENZ] Baseline scope mean: {baseline_mean['scope']:.1f}")
            print(f"[LENZ] Baseline param mean: {baseline_mean['parameter_delta']:.4f}")
            print("[LENZ] Live monitoring ACTIVE.\n")

            socketio.emit('baseline_complete', {
                "message": "Baseline established. Live monitoring active.",
                "baseline_scope_mean": round(baseline_mean["scope"], 1),
                "baseline_param_mean": round(baseline_mean["parameter_delta"], 4),
            })

        return jsonify({"status": "BASELINE", "message": "Decision recorded for baseline"})

    # ── LIVE MONITORING PHASE ───────────────────────────────────────────────
    l1_score, l1_status = layer1_score(decision)
    l2_status, l2_detail = layer2_validate(decision)
    l3_status, l3_detail = layer3_trajectory(decision)

    # Final verdict
    if l1_status == "FLAGGED" or l2_status == "INVALID" or l3_status == "UNSAFE":
        final_status = "BLOCKED"
    else:
        final_status = "APPROVED"

    log_data = {
        "timestamp":      decision.get("timestamp", datetime.now().isoformat()),
        "agent_id":       decision.get("agent_id", "unknown"),
        "scope":          decision.get("scope", 0),
        "parameter_delta":decision.get("parameter_delta", 0),
        "payload_size":   decision.get("payload_size", 0),
        "timing_offset":  decision.get("timing_offset", 0),
        "layer1_score":   l1_score,
        "layer1_status":  l1_status,
        "layer2_status":  l2_status,
        "layer2_detail":  l2_detail,
        "layer3_status":  l3_status,
        "layer3_detail":  l3_detail,
        "final_status":   final_status,
    }

    decision_id = log_to_db(log_data)
    log_data["decision_id"] = decision_id

    # Emit to dashboard
    socketio.emit('new_decision', log_data)

    if final_status == "BLOCKED":
        pending_decision = log_data
        alert_text = build_alert_text(decision, l1_score, l2_detail, l3_detail)
        socketio.emit('decision_blocked', {
            "decision_id": decision_id,
            "alert_text":  alert_text,
            "log_data":    log_data
        })
        print(f"[BLOCKED] Score: {l1_score} | L2: {l2_status} | L3: {l3_status}")
    else:
        print(f"[APPROVED] Score: {l1_score} | L2: {l2_status} | L3: {l3_status}")

    return jsonify({
        "status":       final_status,
        "decision_id":  decision_id,
        "layer1_score": l1_score,
        "layer1_status":l1_status,
        "layer2_status":l2_status,
        "layer3_status":l3_status,
    })

# ── ALERT TEXT BUILDER ─────────────────────────────────────────────────────
def build_alert_text(decision, score, l2_detail, l3_detail):
    scope = decision.get("scope", 0)
    param = decision.get("parameter_delta", 0)
    return (
        f"Deployment decision BLOCKED\n"
        f"Deviation score: {score} (threshold: {ANOMALY_THRESHOLD})\n"
        f"Proposed scope: {int(scope):,} machines\n"
        f"Layer 2: {l2_detail}\n"
        f"Layer 3: {l3_detail}\n"
        f"Action required: Review and choose an action below."
    )

# ── OPERATOR INTERVENTION ENDPOINTS ───────────────────────────────────────
@app.route('/approve', methods=['POST'])
def approve():
    data = request.get_json()
    decision_id = data.get("decision_id")
    update_operator_action(decision_id, "APPROVED", "Operator approved")
    socketio.emit('operator_action', {
        "decision_id": decision_id,
        "action": "APPROVED",
        "message": "Decision approved by operator. Agent resuming."
    })
    print(f"[OPERATOR] Decision {decision_id} APPROVED")
    return jsonify({"status": "ok"})

@app.route('/reject', methods=['POST'])
def reject():
    data = request.get_json()
    decision_id = data.get("decision_id")
    reason      = data.get("reason", "No reason provided")
    update_operator_action(decision_id, "REJECTED", reason)
    socketio.emit('operator_action', {
        "decision_id": decision_id,
        "action": "REJECTED",
        "reason": reason,
        "message": f"Decision rejected. Reason: {reason}"
    })
    print(f"[OPERATOR] Decision {decision_id} REJECTED — {reason}")
    return jsonify({"status": "ok"})

@app.route('/escalate', methods=['POST'])
def escalate():
    data = request.get_json()
    decision_id = data.get("decision_id")
    update_operator_action(decision_id, "ESCALATED", "Escalated to senior review")
    socketio.emit('operator_action', {
        "decision_id": decision_id,
        "action": "ESCALATED",
        "message": "Decision escalated. Awaiting senior review."
    })
    print(f"[OPERATOR] Decision {decision_id} ESCALATED")
    return jsonify({"status": "ok"})

# ── GRAPH DATA ENDPOINT ────────────────────────────────────────────────────
@app.route('/graph', methods=['GET'])
def get_graph():
    edges = []
    for u, v, data in comm_graph.edges(data=True):
        edges.append({
            "from": u, "to": v,
            "color": data.get("color", "green"),
            "scope": data.get("scope", 0),
            "step":  data.get("step", 0),
        })
    return jsonify({"edges": edges, "nodes": list(comm_graph.nodes())})

# ── AUDIT LOG ENDPOINT ─────────────────────────────────────────────────────
@app.route('/audit', methods=['GET'])
def get_audit():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 50")
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    return jsonify([dict(zip(columns, row)) for row in rows])

# ── STATUS ENDPOINT ────────────────────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "baseline_complete": baseline_complete,
        "baseline_count":    len(baseline_data),
        "decisions_logged":  len(decision_log),
        "model_trained":     model is not None,
    })

# ── RUN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("=" * 60)
    print("LenZ — Detection Engine Starting on port 5001")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
