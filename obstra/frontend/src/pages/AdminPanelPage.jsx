import React from 'react';
import { Users, Shield, Settings, Key, Activity, Plus, Edit2, Trash2 } from 'lucide-react';

const AdminPanelPage = () => {
  const [agents, setAgents] = React.useState([
    { id: 1, name: 'DiagnosisAgent', framework: 'LangChain', owner: 'HospitalA', status: 'active' },
    { id: 2, name: 'PrescriptionAgent', framework: 'CrewAI', owner: 'HospitalA', status: 'active' },
    { id: 3, name: 'FinanceAgent', framework: 'LangChain', owner: 'BankCorp', status: 'suspended' },
    { id: 4, name: 'ComplianceBot', framework: 'AutoGen', owner: 'AuditDept', status: 'active' },
  ]);

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Admin Panel</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Manage agents, users, and system settings</p>
      </header>

      <div className="glass-card" style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Users size={20} /> Agent Management
          </h3>
          <button className="btn btn-primary">
            <Plus size={16} /> Add Agent
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Agent ID</th>
                <th>Name</th>
                <th>Framework</th>
                <th>Owner</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {agents.map(agent => (
                <tr key={agent.id}>
                  <td style={{ color: 'var(--text-secondary)' }}>#{agent.id}</td>
                  <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{agent.name}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{agent.framework}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>{agent.owner}</td>
                  <td>
                    <span className={`badge badge-${agent.status === 'active' ? 'approved' : 'blocked'}`}>
                      {agent.status}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <button className="btn" style={{ padding: '8px', background: 'white', border: '1px solid var(--border-highlight)' }}>
                        <Edit2 size={16} />
                      </button>
                      <button className="btn" style={{ padding: '8px', background: 'white', border: '1px solid var(--border-highlight)', color: 'var(--danger)' }}>
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Key size={18} /> API Keys
          </h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Manage API keys for agent authentication
          </p>
          <button className="btn" style={{ background: 'white', border: '1px solid var(--border-highlight)' }}>
            <Key size={16} /> Generate New Key
          </button>
        </div>

        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} /> System Logs
          </h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
            View system activity and audit trails
          </p>
          <button className="btn" style={{ background: 'white', border: '1px solid var(--border-highlight)' }}>
            <Activity size={16} /> View Logs
          </button>
        </div>

        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Settings size={18} /> Security Settings
          </h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Configure security and access controls
          </p>
          <button className="btn" style={{ background: 'white', border: '1px solid var(--border-highlight)' }}>
            <Settings size={16} /> Configure
          </button>
        </div>
      </div>
    </div>
  );
};

export default AdminPanelPage;
