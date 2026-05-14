import React from 'react';
import { Link2, Globe, Server, Bot, Shield, Users, Activity } from 'lucide-react';

const IntegrationsPage = () => {
  const integrations = [
    { name: 'LangChain', icon: Bot, description: 'Seamless integration with LangChain agents', status: 'Available' },
    { name: 'CrewAI', icon: Users, description: 'Plugin for CrewAI multi-agent systems', status: 'Available' },
    { name: 'AutoGen', icon: Globe, description: 'Integration with Microsoft AutoGen', status: 'Coming Soon' },
    { name: 'LlamaIndex', icon: Server, description: 'Connect with LlamaIndex agents', status: 'Available' },
    { name: 'OpenAI Assistants', icon: Bot, description: 'Governance for OpenAI Assistants', status: 'In Development' },
    { name: 'Prometheus', icon: Activity, description: 'Monitoring and alerting integration', status: 'Available' },
  ];

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Integrations</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Connect Obstra with your existing tools and frameworks</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
        {integrations.map((integration, idx) => (
          <div key={idx} className="glass-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
              <div style={{ 
                width: '48px', height: '48px', borderRadius: '8px', 
                background: 'var(--accent-cyan)', opacity: 0.1, 
                display: 'flex', alignItems: 'center', justifyContent: 'center' 
              }}>
                <integration.icon size={24} color="var(--accent-cyan)" />
              </div>
              <div>
                <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>{integration.name}</h4>
                <span style={{ 
                  fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase',
                  color: integration.status === 'Available' ? 'var(--success)' : 'var(--warning)'
                }}>
                  {integration.status}
                </span>
              </div>
            </div>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '16px' }}>
              {integration.description}
            </p>
            <button className="btn" style={{ 
              width: '100%', 
              background: 'white', 
              border: '1px solid var(--border-highlight)',
              color: integration.status === 'Available' ? 'var(--text-primary)' : 'var(--text-muted)'
            }} disabled={integration.status !== 'Available'}>
              <Link2 size={16} /> Connect
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default IntegrationsPage;
