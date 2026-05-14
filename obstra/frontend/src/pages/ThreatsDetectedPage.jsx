import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { AlertTriangle, Shield, Clock, Check, X } from 'lucide-react';

const ThreatsDetectedPage = () => {
  const [threats, setThreats] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchThreats();
  }, []);

  const fetchThreats = async () => {
    try {
      const res = await axios.get('http://localhost:8000/threats');
      setThreats(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const resolveThreat = async (threatId, resolved) => {
    try {
      await axios.post('http://localhost:8000/threats/resolve', {
        threat_id: threatId,
        resolved: resolved,
        resolved_by: 'Admin',
        reason: resolved ? 'Resolved and allowed' : 'Threat blocked'
      });
      fetchThreats();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Threats Detected</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Real-time threat detection feed</p>
      </header>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '48px' }}>
          <p style={{ color: 'var(--text-secondary)' }}>Loading threats...</p>
        </div>
      ) : threats.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '48px' }}>
          <Shield size={48} color="var(--success)" />
          <p style={{ color: 'var(--text-secondary)', marginTop: '16px' }}>No active threats detected</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {threats.map(threat => (
            <div 
              key={threat.id}
              className="glass-card"
              style={{ 
                borderLeft: `4px solid ${threat.status === 'active' ? 'var(--danger)' : 'var(--success)'}`
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '24px' }}>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                    {threat.status === 'active' ? <AlertTriangle size={20} color="var(--danger)" /> : <Check size={20} color="var(--success)" />}
                    <span style={{ fontWeight: 600, fontSize: '1.1rem', color: 'var(--text-primary)' }}>
                      Agent #{threat.agent_id} • {threat.threat_type}
                    </span>
                    <span className={`badge badge-${threat.status === 'active' ? 'pending' : 'success'}`}>
                      {threat.status}
                    </span>
                  </div>
                  
                  <p style={{ color: 'var(--text-secondary)', marginBottom: '12px' }}>{threat.description}</p>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 600, color: threat.severity === 'HIGH' ? 'var(--danger)' : 'var(--warning)' }}>
                        Severity: {threat.severity}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)' }}>
                      <Clock size={14} />
                      {new Date(threat.detected_at).toLocaleString()}
                    </div>
                  </div>
                </div>
                
                {threat.status === 'active' && (
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button 
                      onClick={() => resolveThreat(threat.id, true)}
                      className="btn btn-success"
                      style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
                    >
                      <Check size={16} /> Allow
                    </button>
                    <button 
                      onClick={() => resolveThreat(threat.id, false)}
                      className="btn btn-danger"
                      style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
                    >
                      <X size={16} /> Block
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ThreatsDetectedPage;
