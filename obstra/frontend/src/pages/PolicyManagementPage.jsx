import React from 'react';
import { Settings, Shield, AlertTriangle, Plus, Edit2, Trash2 } from 'lucide-react';

const PolicyManagementPage = () => {
  const [rules, setRules] = React.useState([
    { id: 1, condition: 'IF transfer > 1000000', action: 'REVIEW', enabled: true },
    { id: 2, condition: 'IF dosage > 2000mg', action: 'BLOCK', enabled: true },
    { id: 3, condition: 'IF DELETE without WHERE', action: 'BLOCK', enabled: true },
    { id: 4, condition: 'IF 10+ READ_FILE in 60s', action: 'REVIEW', enabled: true },
  ]);

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Policy Management</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Configure detection policies and rules</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
            <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Shield size={20} /> Rule Engine
            </h3>
            <button className="btn btn-primary">
              <Plus size={16} /> Add Rule
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {rules.map(rule => (
              <div key={rule.id} style={{ 
                display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                padding: '16px', background: '#FAFAFA', borderRadius: '8px', 
                border: '1px solid var(--border-subtle)' 
              }}>
                <div>
                  <p style={{ fontWeight: 600, marginBottom: '4px', color: 'var(--text-primary)' }}>
                    {rule.condition}
                  </p>
                  <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    Action: <span style={{ fontWeight: 600, color: rule.action === 'BLOCK' ? 'var(--danger)' : 'var(--warning)' }}>
                      {rule.action}
                    </span>
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <div style={{ 
                    width: '12px', height: '12px', borderRadius: '50%', 
                    background: rule.enabled ? 'var(--success)' : 'var(--text-muted)' 
                  }}></div>
                  <button className="btn" style={{ padding: '8px', background: 'white', border: '1px solid var(--border-highlight)' }}>
                    <Edit2 size={16} />
                  </button>
                  <button className="btn" style={{ padding: '8px', background: 'white', border: '1px solid var(--border-highlight)', color: 'var(--danger)' }}>
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="glass-card">
            <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Settings size={18} /> Risk Thresholds
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Low Risk</span>
                  <span style={{ color: 'var(--text-primary)' }}>&lt; 0.3</span>
                </div>
                <input type="range" min="0" max="0.5" step="0.05" defaultValue="0.3" style={{ width: '100%' }} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>High Risk</span>
                  <span style={{ color: 'var(--text-primary)' }}>&gt; 0.7</span>
                </div>
                <input type="range" min="0.5" max="1" step="0.05" defaultValue="0.7" style={{ width: '100%' }} />
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Consensus Timeout</span>
                  <span style={{ color: 'var(--text-primary)' }}>5 min</span>
                </div>
                <input type="number" min="1" max="60" defaultValue="5" style={{ 
                  width: '100%', padding: '10px', borderRadius: '4px', 
                  border: '1px solid var(--border-subtle)', fontSize: '1rem' 
                }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PolicyManagementPage;
