import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity, AlertTriangle, CheckSquare, TrendingUp,
  ArrowRight, Bot, Shield, Clock, Zap, RefreshCw
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import HumanDecisionModal from '../components/HumanDecisionModal';

function getRiskColor(score) {
  if (score >= 0.7) return 'var(--danger)';
  if (score >= 0.3) return 'var(--warning)';
  return 'var(--safe)';
}

function getRiskLabel(score) {
  if (score >= 0.7) return 'HIGH';
  if (score >= 0.3) return 'MED';
  return 'LOW';
}

function DecisionBadge({ status }) {
  const s = String(status || '').toLowerCase();
  let cls = 'badge';
  if (s === 'blocked') cls += ' badge-blocked';
  else if (s === 'pending') cls += ' badge-pending';
  else if (s === 'approved' || s === 'allowed') cls += ' badge-allowed';
  else if (s === 'escalated') cls += ' badge-escalated';
  else cls += ' badge-low';
  return <span className={cls}>{status}</span>;
}

function TimeAgo({ ts }) {
  if (!ts) return <span style={{ color: 'var(--text-muted)' }}>—</span>;
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>just now</span>;
  if (mins < 60) return <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{mins}m ago</span>;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{hrs}h ago</span>;
  return <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{new Date(ts).toLocaleDateString()}</span>;
}

export default function DashboardOverview() {
  const navigate = useNavigate();
  const [summary, setSummary] = useState(null);
  const [agents, setAgents] = useState([]);
  const [recentActions, setRecentActions] = useState([]);
  const [threats, setThreats] = useState([]);
  const [modalAction, setModalAction] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [sumRes, agentsRes, logsRes, threatsRes] = await Promise.all([
        api.get('/dashboard/summary'),
        api.get('/agents/'),
        api.get('/audit/logs', { params: { limit: 15 } }),
        api.get('/threats', { params: { status: 'active' } }),
      ]);
      setSummary(sumRes.data);
      setAgents(agentsRes.data);
      setRecentActions(logsRes.data);
      setThreats(threatsRes.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, [refresh]);

  const agentMap = {};
  agents.forEach(a => { agentMap[a.id] = a.name; });

  const stats = summary ? [
    { label: 'Active Agents',    value: summary.active_agents,          hint: `${summary.total_agents} total`, color: 'var(--primary)', icon: Bot, nav: '/agent-mesh' },
    { label: 'Open Threats',     value: summary.open_threats,           hint: 'Active anomalies', color: 'var(--danger)', icon: AlertTriangle, nav: '/threats' },
    { label: 'Pending Review',   value: summary.pending_consensus,      hint: 'Awaiting decision', color: 'var(--warning)', icon: Clock, nav: '/approvals' },
    { label: 'Governed Actions', value: summary.total_action_logs,      hint: `${summary.blocked_actions_logged} blocked`, color: 'var(--secondary)', icon: Shield, nav: '/audit-trail' },
    { label: 'Blocked Actions',  value: summary.blocked_actions_logged, hint: 'Auto-blocked', color: 'var(--danger)', icon: Zap, nav: '/risk-analysis' },
    { label: 'System Health',    value: '98%',                          hint: 'All systems nominal', color: 'var(--safe)', icon: Activity, nav: '/dashboard' },
  ] : [];

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '60vh', flexDirection: 'column', gap: 16 }}>
        <div className="spinner" style={{ width: 32, height: 32 }} />
        <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Loading OBSTRA dashboard…</span>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Command Center</h1>
          <p>Live AI governance overview — every action monitored in real time</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="live-indicator">
            <div className="live-dot" />
            LIVE
          </div>
          <button className="btn" onClick={refresh} style={{ gap: 6 }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="grid-auto" style={{ marginBottom: 28 }}>
        {stats.map(card => (
          <button
            key={card.label}
            className="stat-card"
            onClick={() => navigate(card.nav)}
            style={{ cursor: 'pointer', textAlign: 'left', background: 'none', fontFamily: 'inherit', width: '100%' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
              <span className="stat-label">{card.label}</span>
              <div style={{ padding: 8, borderRadius: 8, background: `${card.color}18` }}>
                <card.icon size={16} color={card.color} />
              </div>
            </div>
            <div className="stat-value" style={{ color: card.color }}>{card.value}</div>
            <div className="stat-hint">{card.hint}</div>
          </button>
        ))}
      </div>

      {/* Main content split */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 24 }}>

        {/* Recent Actions */}
        <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
            <h3 className="section-title" style={{ margin: 0 }}>
              <Activity size={18} color="var(--primary)" /> Recent Actions
            </h3>
            <button className="btn btn-ghost" onClick={() => navigate('/risk-analysis')} style={{ gap: 6, fontSize: '0.8rem' }}>
              View All <ArrowRight size={14} />
            </button>
          </div>
          <div>
            {recentActions.length === 0 ? (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                No actions logged yet
              </div>
            ) : recentActions.map(action => {
              const score = action.final_risk_score ?? 0;
              const riskColor = getRiskColor(score);
              return (
                <button
                  key={action.id}
                  onClick={() => setModalAction(action)}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr auto auto auto',
                    alignItems: 'center',
                    gap: 16,
                    padding: '14px 24px',
                    borderBottom: '1px solid var(--border)',
                    background: 'none',
                    border: 'none',
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                    width: '100%',
                    textAlign: 'left',
                    transition: 'background 0.15s',
                    fontFamily: 'inherit',
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(30,45,74,0.3)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'none'}
                >
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                      {action.action_type}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 3 }}>
                      {agentMap[action.agent_id] || `Agent #${action.agent_id}`}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 6, height: 6, borderRadius: '50%', background: riskColor }} />
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: riskColor }}>
                      {score.toFixed(2)}
                    </span>
                  </div>
                  <DecisionBadge status={action.status} />
                  <TimeAgo ts={action.timestamp} />
                </button>
              );
            })}
          </div>
        </div>

        {/* Right panel: Threats + Pending */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* Active Threats */}
          <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '18px 20px', borderBottom: '1px solid var(--border)' }}>
              <h3 className="section-title" style={{ margin: 0, fontSize: '0.9rem' }}>
                <AlertTriangle size={16} color="var(--danger)" /> Active Threats
              </h3>
              <button className="btn btn-ghost" onClick={() => navigate('/threats')} style={{ fontSize: '0.75rem', padding: '6px 10px' }}>
                View All <ArrowRight size={12} />
              </button>
            </div>
            {threats.length === 0 ? (
              <div style={{ padding: 28, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <Shield size={28} style={{ marginBottom: 8, opacity: 0.4 }} />
                <div>No active threats</div>
              </div>
            ) : threats.slice(0, 4).map(t => (
              <div key={t.id} style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--danger)', marginTop: 6, flexShrink: 0, animation: 'pulse-dot 1.5s infinite' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 3 }}>
                    {t.threat_type || t.anomaly_type || 'Anomaly Detected'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.description || 'Suspicious activity detected'}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                    {t.detected_at ? new Date(t.detected_at).toLocaleTimeString() : '—'}
                  </div>
                </div>
                <span className="badge badge-blocked" style={{ fontSize: '0.65rem', flexShrink: 0 }}>
                  {t.severity || 'HIGH'}
                </span>
              </div>
            ))}
          </div>

          {/* Quick Stats */}
          <div className="glass-card">
            <h3 className="section-title" style={{ fontSize: '0.9rem' }}>
              <TrendingUp size={16} color="var(--secondary)" /> System Stats
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                { label: 'Actions Today', value: summary?.total_action_logs ?? 0, color: 'var(--primary)' },
                { label: 'Avg Risk Score', value: '0.34', color: 'var(--warning)' },
                { label: 'Blocked Rate', value: summary ? `${((summary.blocked_actions_logged / Math.max(summary.total_action_logs, 1)) * 100).toFixed(1)}%` : '0%', color: 'var(--danger)' },
                { label: 'Agents Active', value: summary?.active_agents ?? 0, color: 'var(--safe)' },
              ].map(s => (
                <div key={s.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{s.label}</span>
                  <span style={{ fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700, color: s.color, fontSize: '0.95rem' }}>
                    {s.value}
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>

      {modalAction && (
        <HumanDecisionModal
          action={modalAction}
          agentLabel={agentMap[modalAction.agent_id]}
          onClose={() => setModalAction(null)}
          onApplied={refresh}
        />
      )}
    </div>
  );
}
