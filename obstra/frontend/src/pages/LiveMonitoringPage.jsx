import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Radio, Activity, Pause, Play, Trash2, Bot, RefreshCw } from 'lucide-react';
import api from '../api/client';

function getRiskColor(score) {
  if (score >= 0.7) return 'var(--danger)';
  if (score >= 0.3) return 'var(--warning)';
  return 'var(--safe)';
}

function EventRow({ event }) {
  const score = event.final_risk_score ?? event.risk_score ?? 0;
  const riskColor = getRiskColor(score);
  const status = String(event.status || '').toLowerCase();
  const statusClass = status === 'blocked' ? 'badge-blocked' : status === 'pending' ? 'badge-pending' : 'badge-allowed';

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '90px 1fr auto auto auto',
      gap: 12,
      alignItems: 'center',
      padding: '10px 16px',
      borderLeft: `3px solid ${riskColor}`,
      borderBottom: '1px solid var(--border)',
      transition: 'background 0.15s',
      animation: 'toast-in 0.3s ease',
    }}
      onMouseEnter={e => e.currentTarget.style.background = 'rgba(30,45,74,0.3)'}
      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
    >
      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
        {event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}
      </span>
      <div>
        <span style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)' }}>
          {event.action_type}
        </span>
        {event.agentName && (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginLeft: 8 }}>
            {event.agentName}
          </span>
        )}
        {event.reason && (
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 400 }}>
            {event.reason}
          </div>
        )}
      </div>
      <span style={{ fontSize: '0.8rem', fontWeight: 700, color: riskColor }}>
        {score.toFixed(2)}
      </span>
      <span className={`badge ${statusClass}`} style={{ fontSize: '0.65rem' }}>{event.status}</span>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: riskColor, flexShrink: 0 }} />
    </div>
  );
}

function AgentCard({ agent, selected, onClick }) {
  const isActive = agent.is_active;
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '14px 16px',
        background: selected ? 'rgba(0,180,255,0.1)' : 'var(--bg-surface)',
        border: `1px solid ${selected ? 'rgba(0,180,255,0.35)' : 'var(--border)'}`,
        borderRadius: 10,
        cursor: 'pointer',
        textAlign: 'left',
        width: '100%',
        transition: 'all 0.15s',
        fontFamily: 'inherit',
      }}
    >
      <div style={{
        width: 36, height: 36, borderRadius: '50%',
        background: isActive ? 'rgba(0,180,255,0.15)' : 'rgba(61,79,107,0.3)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
      }}>
        <Bot size={16} color={isActive ? 'var(--primary)' : 'var(--text-muted)'} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, fontSize: '0.875rem', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {agent.name}
        </div>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: 2 }}>
          {agent.framework || agent.provider || 'internal'}
        </div>
      </div>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: isActive ? 'var(--safe)' : 'var(--text-muted)', flexShrink: 0, animation: isActive ? 'live-pulse 2s infinite' : 'none' }} />
    </button>
  );
}

export default function LiveMonitoringPage() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [events, setEvents] = useState([]);
  const [paused, setPaused] = useState(false);
  const [lastPoll, setLastPoll] = useState(null);
  const feedRef = useRef(null);
  const pausedRef = useRef(false);
  pausedRef.current = paused;

  const fetchAgents = useCallback(async () => {
    try {
      const r = await api.get('/agents/');
      setAgents(r.data);
    } catch {}
  }, []);

  const fetchEvents = useCallback(async () => {
    if (pausedRef.current) return;
    try {
      const params = { limit: 50 };
      if (selectedAgent) params.agent_id = selectedAgent.id;
      const r = await api.get('/audit/logs', { params });
      const agentsMap = {};
      agents.forEach(a => { agentsMap[a.id] = a.name; });
      const mapped = r.data.map(e => ({ ...e, agentName: agentsMap[e.agent_id] || `#${e.agent_id}` }));
      setEvents(mapped);
      setLastPoll(new Date());
    } catch {}
  }, [selectedAgent, agents]);

  useEffect(() => { fetchAgents(); }, [fetchAgents]);

  useEffect(() => {
    fetchEvents();
    const t = setInterval(fetchEvents, 3000);
    return () => clearInterval(t);
  }, [fetchEvents]);

  // Auto-scroll
  useEffect(() => {
    if (!paused && feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [events, paused]);

  const displayEvents = selectedAgent
    ? events.filter(e => String(e.agent_id) === String(selectedAgent.id))
    : events;

  const highRisk = events.filter(e => (e.final_risk_score || 0) >= 0.7).length;
  const pending = events.filter(e => String(e.status).toLowerCase() === 'pending').length;
  const blocked = events.filter(e => String(e.status).toLowerCase() === 'blocked').length;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Radio size={24} color="var(--primary)" /> Live Monitoring
          </h1>
          <p>Real-time stream of all agent activity — updates every 3 seconds</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {lastPoll && (
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Last update: {lastPoll.toLocaleTimeString()}
            </span>
          )}
          <div className="live-indicator">
            <div className="live-dot" style={{ background: paused ? 'var(--warning)' : 'var(--safe)' }} /> {paused ? 'PAUSED' : 'LIVE'}
          </div>
          <button className={`btn ${paused ? 'btn-success' : 'btn-warning'}`} onClick={() => setPaused(p => !p)} style={{ gap: 6 }}>
            {paused ? <><Play size={14} /> Resume</> : <><Pause size={14} /> Pause</>}
          </button>
          <button className="btn" onClick={() => { setEvents([]); fetchEvents(); }} style={{ gap: 6 }}>
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Quick stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 24 }}>
        {[
          { label: 'Total Events', value: events.length, color: 'var(--primary)' },
          { label: 'High Risk', value: highRisk, color: 'var(--danger)' },
          { label: 'Pending', value: pending, color: 'var(--warning)' },
          { label: 'Blocked', value: blocked, color: 'var(--danger)' },
        ].map(s => (
          <div key={s.label} className="glass-card" style={{ padding: 16, textAlign: 'center' }}>
            <div style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '1.8rem', fontWeight: 700, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{s.label}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 24 }}>

        {/* Agent list */}
        <div>
          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <h3 className="section-title" style={{ margin: 0, fontSize: '0.875rem' }}>
                <Activity size={15} color="var(--secondary)" /> Agents
              </h3>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{agents.length} total</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <AgentCard
                agent={{ name: 'All Agents', is_active: true, framework: 'all agents' }}
                selected={!selectedAgent}
                onClick={() => setSelectedAgent(null)}
              />
              {agents.map(a => (
                <AgentCard key={a.id} agent={a} selected={selectedAgent?.id === a.id} onClick={() => setSelectedAgent(a)} />
              ))}
            </div>
          </div>
        </div>

        {/* Live feed */}
        <div className="glass-card" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--border)' }}>
            <h3 className="section-title" style={{ margin: 0, fontSize: '0.9rem' }}>
              Activity Stream {selectedAgent ? `— ${selectedAgent.name}` : '— All Agents'}
            </h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <button className="btn btn-ghost" style={{ fontSize: '0.75rem', padding: '6px 10px' }} onClick={() => setEvents([])}>
                <Trash2 size={12} /> Clear
              </button>
            </div>
          </div>

          <div
            ref={feedRef}
            style={{ flex: 1, overflowY: 'auto', maxHeight: '60vh', minHeight: 300 }}
          >
            {displayEvents.length === 0 ? (
              <div style={{ padding: 60, textAlign: 'center', color: 'var(--text-muted)' }}>
                <Radio size={36} style={{ marginBottom: 12, opacity: 0.3 }} />
                <div>Waiting for agent activity…</div>
                <div style={{ fontSize: '0.8rem', marginTop: 8 }}>Events appear here as agents act</div>
              </div>
            ) : displayEvents.map(e => (
              <EventRow key={e.id} event={e} />
            ))}
          </div>

          {/* Risk legend */}
          <div style={{ display: 'flex', gap: 20, padding: '12px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
            {[
              { label: 'Low Risk (<0.3)', color: 'var(--safe)' },
              { label: 'Medium Risk (0.3–0.7)', color: 'var(--warning)' },
              { label: 'High Risk (>0.7)', color: 'var(--danger)' },
            ].map(l => (
              <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: l.color }} />
                <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{l.label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
