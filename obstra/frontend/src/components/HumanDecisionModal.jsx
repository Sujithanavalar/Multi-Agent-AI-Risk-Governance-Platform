import React, { useState } from 'react';
import { X, CheckCircle, XCircle, AlertOctagon, Clock, AlertTriangle } from 'lucide-react';
import api from '../api/client';

function RiskBar({ label, value, max = 1 }) {
  const pct = Math.min(100, ((value || 0) / max) * 100);
  const color = value >= 0.7 ? 'var(--danger)' : value >= 0.3 ? 'var(--warning)' : 'var(--safe)';
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: '0.78rem', fontWeight: 700, color }}>{(value || 0).toFixed(3)}</span>
      </div>
      <div className="risk-bar-track">
        <div className="risk-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function HumanDecisionModal({ action, agentLabel, onClose, onApplied }) {
  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState('');
  const [result, setResult] = useState(null);

  const score = action?.final_risk_score ?? 0;
  const scoreColor = score >= 0.7 ? 'var(--danger)' : score >= 0.3 ? 'var(--warning)' : 'var(--safe)';

  async function decide(decision) {
    setLoading(true);
    try {
      await api.post(`/actions/${action.id}/human-decision`, {
        decision,
        reviewer: 'operator',
        notes: notes || decision,
      });
      setResult(decision);
      onApplied?.();
    } catch (e) {
      alert(e?.response?.data?.detail || 'Request failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()} style={{ maxWidth: 680 }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 20 }}>
          <div>
            <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', margin: 0, fontSize: '1.1rem' }}>Action Review</h3>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>
              ID: {action.id} · {agentLabel || `Agent #${action.agent_id}`}
            </div>
          </div>
          <button onClick={onClose} className="btn btn-ghost" style={{ padding: 8 }}>
            <X size={18} />
          </button>
        </div>

        {result ? (
          <div className={`banner banner-${result === 'ALLOW' ? 'success' : result === 'BLOCK' ? 'danger' : 'warning'}`} style={{ justifyContent: 'center', padding: 24, flexDirection: 'column', gap: 8, textAlign: 'center' }}>
            <div style={{ fontWeight: 700, fontSize: '1rem' }}>
              {result === 'ALLOW' ? '✅ Action Approved' : result === 'BLOCK' ? '🚫 Action Blocked' : '⚠️ Alert Recorded'}
            </div>
            <div style={{ fontSize: '0.85rem', opacity: 0.8 }}>Decision logged to Supabase audit trail</div>
            <button className="btn" onClick={onClose} style={{ marginTop: 8, alignSelf: 'center' }}>Close</button>
          </div>
        ) : (
          <>
            {/* Risk Score */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, padding: '16px 20px', background: 'var(--bg-surface)', borderRadius: 10, border: '1px solid var(--border)' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '2.5rem', fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700, color: scoreColor, lineHeight: 1 }}>
                  {score.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Risk Score</div>
              </div>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <RiskBar label="ML Anomaly" value={action.isolation_forest_score} />
                <RiskBar label="Rule Engine" value={action.rule_engine_score} />
                <RiskBar label="Context" value={action.context_score} />
              </div>
            </div>

            {/* Action Info */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
              <div style={{ padding: '12px 14px', background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Action Type</div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{action.action_type}</div>
              </div>
              <div style={{ padding: '12px 14px', background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>Status</div>
                <span className={`badge badge-${String(action.status).toLowerCase()}`}>{action.status}</span>
              </div>
            </div>

            {/* Payload */}
            {action.payload && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Payload</div>
                <pre style={{ fontSize: '0.78rem', maxHeight: 140, overflow: 'auto' }}>
                  {JSON.stringify(action.payload, null, 2)}
                </pre>
              </div>
            )}

            {/* Reason */}
            {action.reason && (
              <div className="banner banner-warning" style={{ marginBottom: 16 }}>
                <AlertTriangle size={16} style={{ flexShrink: 0 }} />
                <span style={{ fontSize: '0.83rem' }}>{action.reason}</span>
              </div>
            )}

            {/* Notes */}
            <div style={{ marginBottom: 20 }}>
              <label>Decision Notes (optional)</label>
              <textarea
                rows={2}
                placeholder="Add context for the audit trail..."
                value={notes}
                onChange={e => setNotes(e.target.value)}
              />
            </div>

            {/* Decision Buttons */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
              <button className="btn btn-success" disabled={loading} onClick={() => decide('ALLOW')} style={{ flexDirection: 'column', padding: 14, gap: 6 }}>
                <CheckCircle size={20} />
                <span>Allow</span>
              </button>
              <button className="btn btn-danger" disabled={loading} onClick={() => decide('BLOCK')} style={{ flexDirection: 'column', padding: 14, gap: 6 }}>
                <XCircle size={20} />
                <span>Block</span>
              </button>
              <button className="btn btn-warning" disabled={loading} onClick={() => decide('ALERT')} style={{ flexDirection: 'column', padding: 14, gap: 6 }}>
                <AlertOctagon size={20} />
                <span>Alert</span>
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
