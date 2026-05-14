import React, { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, CheckCircle, RefreshCw, Filter } from 'lucide-react';
import api from '../api/client';

function SeverityBadge({ severity }) {
  const s = String(severity || 'medium').toLowerCase();
  const cls = s === 'critical' ? 'badge-blocked' : s === 'high' ? 'badge-escalated' : s === 'medium' ? 'badge-pending' : 'badge-low';
  return <span className={`badge ${cls}`}>{severity || 'medium'}</span>;
}

function ThreatCard({ threat, onResolve }) {
  const [resolving, setResolving] = useState(false);

  async function resolve() {
    setResolving(true);
    try {
      await api.post('/threats/resolve', { threat_id: threat.id, resolved: true, resolved_by: 'operator', reason: 'Resolved via dashboard' });
      onResolve();
    } catch (e) {
      alert(e?.response?.data?.detail || 'Failed to resolve');
    } finally {
      setResolving(false);
    }
  }

  const isActive = threat.status === 'active' || !threat.resolved;
  const sev = String(threat.severity || 'high').toLowerCase();
  const borderColor = sev === 'critical' ? 'var(--danger)' : sev === 'high' ? 'var(--warning)' : sev === 'medium' ? '#FF9500' : 'var(--text-muted)';

  return (
    <div style={{
      background: 'var(--bg-card)',
      border: `1px solid ${borderColor}44`,
      borderLeft: `4px solid ${borderColor}`,
      borderRadius: 12,
      padding: 20,
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      transition: 'all 0.2s',
    }}
      onMouseEnter={e => e.currentTarget.style.borderLeftColor = borderColor}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', flex: 1 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: isActive ? 'rgba(255,59,92,0.1)' : 'rgba(61,79,107,0.2)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            <AlertTriangle size={18} color={isActive ? 'var(--danger)' : 'var(--text-muted)'} />
          </div>
          <div>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem', marginBottom: 4 }}>
              {threat.threat_type || threat.anomaly_type || 'Anomaly Detected'}
            </div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {threat.description || 'Suspicious activity detected by risk engine'}
            </div>
          </div>
        </div>
        <SeverityBadge severity={threat.severity} />
      </div>

      <div style={{ display: 'flex', gap: 16, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
        <span>Agent #{threat.agent_id}</span>
        <span>·</span>
        <span>{threat.detected_at ? new Date(threat.detected_at).toLocaleString() : '—'}</span>
      </div>

      {isActive && (
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-success" style={{ gap: 6, fontSize: '0.82rem' }} onClick={resolve} disabled={resolving}>
            <CheckCircle size={15} />
            {resolving ? 'Resolving…' : 'Resolve Threat'}
          </button>
        </div>
      )}
    </div>
  );
}

export default function ThreatsDetectedPage() {
  const [activeThreats, setActiveThreats] = useState([]);
  const [resolvedThreats, setResolvedThreats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showResolved, setShowResolved] = useState(false);
  const [severityFilter, setSeverityFilter] = useState('all');

  const fetchThreats = useCallback(async () => {
    try {
      const [activeRes, resolvedRes] = await Promise.all([
        api.get('/threats', { params: { status: 'active' } }),
        api.get('/threats', { params: { status: 'resolved' } }),
      ]);
      setActiveThreats(activeRes.data);
      setResolvedThreats(resolvedRes.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchThreats();
    const t = setInterval(fetchThreats, 8000);
    return () => clearInterval(t);
  }, [fetchThreats]);

  const filter = threats => severityFilter === 'all' ? threats : threats.filter(t => String(t.severity || '').toLowerCase() === severityFilter);
  const filteredActive = filter(activeThreats);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <AlertTriangle size={24} color="var(--danger)" /> Threats
          </h1>
          <p>Real-time anomaly feed — all high-risk or auto-blocked agent actions</p>
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <div className="live-indicator"><div className="live-dot" /> LIVE</div>
          <button className="btn" onClick={fetchThreats} style={{ gap: 6 }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 28 }}>
        {[
          { label: 'Active Threats', value: activeThreats.length, color: 'var(--danger)' },
          { label: 'Critical', value: activeThreats.filter(t => t.severity === 'CRITICAL' || t.severity === 'critical').length, color: 'var(--danger)' },
          { label: 'High', value: activeThreats.filter(t => t.severity === 'HIGH' || t.severity === 'high').length, color: 'var(--warning)' },
          { label: 'Resolved Today', value: resolvedThreats.length, color: 'var(--safe)' },
        ].map(s => (
          <div key={s.label} className="glass-card" style={{ padding: 16, textAlign: 'center' }}>
            <div style={{ fontFamily: 'Space Grotesk, sans-serif', fontSize: '2rem', fontWeight: 700, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Filter */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <Filter size={16} color="var(--text-muted)" style={{ alignSelf: 'center' }} />
        {['all', 'critical', 'high', 'medium', 'low'].map(s => (
          <button key={s} className={`btn ${severityFilter === s ? '' : 'btn-ghost'}`}
            style={severityFilter === s ? { background: 'var(--danger-glow)', borderColor: 'var(--danger)', color: 'var(--danger)' } : {}}
            onClick={() => setSeverityFilter(s)}>
            {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      {/* Active Threats */}
      <div className="glass-card" style={{ marginBottom: 24, padding: '20px 24px' }}>
        <h3 className="section-title">
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <div className="live-dot" /> Active Threats
          </span>
          <span style={{ fontWeight: 400, color: 'var(--text-muted)', fontSize: '0.85rem', marginLeft: 8 }}>({filteredActive.length})</span>
        </h3>

        {loading ? (
          <div style={{ textAlign: 'center', padding: 40 }}><div className="spinner" style={{ margin: '0 auto' }} /></div>
        ) : filteredActive.length === 0 ? (
          <div style={{ textAlign: 'center', padding: 48 }}>
            <CheckCircle size={40} color="var(--safe)" style={{ marginBottom: 12, opacity: 0.7 }} />
            <h3 style={{ color: 'var(--safe)', marginBottom: 8 }}>All Clear</h3>
            <p style={{ color: 'var(--text-muted)' }}>No active threats detected</p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: 16 }}>
            {filteredActive.map(t => <ThreatCard key={t.id} threat={t} onResolve={fetchThreats} />)}
          </div>
        )}
      </div>

      {/* Resolved */}
      <div className="glass-card" style={{ padding: '16px 24px' }}>
        <button
          style={{ display: 'flex', width: '100%', justifyContent: 'space-between', alignItems: 'center', background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontFamily: 'inherit' }}
          onClick={() => setShowResolved(r => !r)}
        >
          <h3 className="section-title" style={{ margin: 0 }}>Resolved Threats ({resolvedThreats.length})</h3>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{showResolved ? 'Collapse ▲' : 'Expand ▼'}</span>
        </button>

        {showResolved && resolvedThreats.length > 0 && (
          <div style={{ marginTop: 20, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: 16 }}>
            {resolvedThreats.map(t => <ThreatCard key={t.id} threat={t} onResolve={fetchThreats} />)}
          </div>
        )}
      </div>
    </div>
  );
}
