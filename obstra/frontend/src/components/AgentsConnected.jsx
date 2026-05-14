import React from 'react';
import { Cpu, Server, Users } from 'lucide-react';

const AgentsConnected = ({ agents }) => {
  return (
    <div className="glass-card">
      <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Cpu size={20} color="var(--accent-cyan)" />
        Agents Connected
      </h3>
      
      {agents.length === 0 ? (
        <p style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '24px' }}>
          No agents registered yet.
        </p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '12px' }}>
          {agents.map((agent) => (
            <div 
              key={agent.id}
              style={{
                padding: '16px',
                border: '1px solid var(--border-subtle)',
                borderRadius: '8px',
                background: '#FAFAFA'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: 'var(--primary-red)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white'
                }}>
                  <Server size={16} />
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ fontWeight: 600, fontSize: '0.9rem', margin: 0, color: 'var(--text-primary)' }}>
                    {agent.name}
                  </p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', margin: 0 }}>
                    {agent.framework}
                  </p>
                </div>
              </div>
              {agent.owner && (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0 }}>
                  <Users size={12} style={{ display: 'inline', marginRight: '4px' }} />
                  {agent.owner}
                </p>
              )}
              <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', margin: '8px 0 0 0', fontFamily: 'monospace' }}>
                ID: {agent.id}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentsConnected;
