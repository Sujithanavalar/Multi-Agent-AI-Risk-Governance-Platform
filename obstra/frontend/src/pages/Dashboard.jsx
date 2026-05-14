import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import { Bot, CheckSquare, AlertTriangle, Shield } from 'lucide-react';
import Header from '../components/Header';

const Dashboard = () => {
  const { agents, approvals, anomalies, rules, fetchAgents, fetchApprovals, fetchAnomalies, fetchRules } = useStore();

  useEffect(() => {
    fetchAgents();
    fetchApprovals();
    fetchAnomalies();
    fetchRules();
  }, [fetchAgents, fetchApprovals, fetchAnomalies, fetchRules]);

  const stats = [
    { label: 'Total Agents', value: agents.length, icon: Bot, color: '#C74634', bg: '#FEF2F2' },
    { label: 'Pending Approvals', value: approvals.filter(a => a.status === 'pending').length, icon: CheckSquare, color: '#B45309', bg: '#FFFBEB' },
    { label: 'Active Anomalies', value: anomalies.filter(a => !a.resolved).length, icon: AlertTriangle, color: '#C0392B', bg: '#FEF2F2' },
    { label: 'Active Rules', value: rules.filter(r => r.enabled).length, icon: Shield, color: '#1B8A4E', bg: '#EBF5EE' },
  ];

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Dashboard Overview</h1>
          <p>Real-time monitoring of your AI agent governance</p>
        </div>

        <div className="stats-grid">
          {stats.map((stat, idx) => {
            const Icon = stat.icon;
            return (
              <div key={idx} className="stat-card">
                <div className="stat-card-header">
                  <div className="stat-icon" style={{ backgroundColor: stat.bg }}>
                    <Icon style={{ color: stat.color }} size={24} />
                  </div>
                </div>
                <div>
                  <p className="stat-value">{stat.value}</p>
                  <p className="stat-label">{stat.label}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="content-grid">
          <div className="card">
            <div className="card-header">
              <h2>Recent Agents</h2>
            </div>
            {agents.length > 0 ? (
              <div>
                {agents.slice(0, 5).map(agent => (
                  <div key={agent.id} className="agent-item">
                    <div className="agent-info">
                      <h4>{agent.name}</h4>
                      <p>{agent.provider} • {agent.model}</p>
                    </div>
                    <span className={`badge ${agent.status === 'active' ? 'badge-success' : agent.status === 'paused' ? 'badge-muted' : 'badge-danger'}`}>
                      {agent.status}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#9AA5B4' }}>No agents yet</p>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2>Pending Approvals</h2>
            </div>
            {approvals.filter(a => a.status === 'pending').length > 0 ? (
              <div>
                {approvals.filter(a => a.status === 'pending').slice(0, 5).map(approval => (
                  <div key={approval.id} className="agent-item">
                    <div className="agent-info">
                      <h4>Approval Required</h4>
                      <p>Risk: {(approval.risk_score * 100).toFixed(0)}%</p>
                    </div>
                    <span className="badge badge-warning">Pending</span>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#9AA5B4' }}>No pending approvals</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
