import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  Activity, Shield, AlertTriangle, CheckCircle, Users, Clock, Eye, 
  TrendingUp, Database, ArrowRight, XCircle, ThumbsUp, Bell
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const DashboardOverview = () => {
  const [stats, setStats] = useState({
    activeAgents: 0,
    threatsBlocked: 0,
    pendingReviews: 0,
    auditIntegrity: '100%'
  });
  const [recentActions, setRecentActions] = useState([]);
  const [selectedAction, setSelectedAction] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const logsRes = await axios.get('http://localhost:8000/audit/logs');
        const agentsRes = await axios.get('http://localhost:8000/agents/');
        const pendingRes = await axios.get('http://localhost:8000/consensus/pending');
        
        // Sort actions: BLOCKED first, then PENDING, then APPROVED
        const sortedLogs = [...logsRes.data].sort((a, b) => {
          const priority = { BLOCKED: 0, PENDING: 1, APPROVED: 2 };
          return priority[a.status] - priority[b.status];
        }).slice(0, 20);
        
        setRecentActions(sortedLogs);
        setStats({
          activeAgents: agentsRes.data.length,
          threatsBlocked: logsRes.data.filter(l => l.status === 'BLOCKED').length,
          pendingReviews: pendingRes.data.length,
          auditIntegrity: '100%'
        });
      } catch (err) {
        console.error(err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleActionClick = (action) => {
    setSelectedAction(action);
    setShowModal(true);
  };

  const handleManualAction = async (actionType) => {
    try {
      alert(`Manual ${actionType} action sent! (Connect to Supabase backend for full functionality)`);
      setShowModal(false);
    } catch (err) {
      console.error(err);
    }
  };

  const statCards = [
    { 
      label: 'Active Agents', 
      value: stats.activeAgents, 
      icon: Users, 
      color: 'var(--accent-cyan)',
      glow: 'rgba(10, 130, 118, 0.2)',
      action: () => navigate('/agent-mesh')
    },
    { 
      label: 'Threats Blocked', 
      value: stats.threatsBlocked, 
      icon: Shield, 
      color: 'var(--danger)',
      glow: 'rgba(220, 53, 69, 0.2)',
      action: () => navigate('/threats')
    },
    { 
      label: 'Pending Reviews', 
      value: stats.pendingReviews, 
      icon: AlertTriangle, 
      color: 'var(--warning)',
      glow: 'rgba(245, 166, 35, 0.2)',
      action: () => navigate('/approvals')
    },
    { 
      label: 'Audit Integrity', 
      value: stats.auditIntegrity, 
      icon: Database, 
      color: 'var(--success)',
      glow: 'rgba(25, 135, 84, 0.2)',
      action: () => navigate('/audit-trail')
    }
  ];

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Dashboard Overview</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Real-time monitoring and governance for your AI agents</p>
      </header>

      {/* Stats Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        {statCards.map((card, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="glass-card"
            style={{ 
              borderTop: `4px solid ${card.color}`, 
              boxShadow: `0 4px 12px ${card.glow}`,
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onClick={card.action}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-4px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '16px' }}>
              <div style={{ 
                width: '48px', height: '48px', 
                borderRadius: '12px', 
                background: card.glow, 
                display: 'flex', alignItems: 'center', justifyContent: 'center' 
              }}>
                <card.icon size={24} color={card.color} />
              </div>
              <div style={{ flex: 1 }}>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{card.label}</p>
                <p style={{ margin: 0, fontSize: '2.5rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
                  {card.value}
                </p>
              </div>
              <ArrowRight size={20} color="var(--text-muted)" />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recent Actions */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={20} />
            Recent Actions (Last 20)
          </h3>
          <button 
            onClick={() => navigate('/audit-trail')}
            className="btn"
            style={{ fontSize: '0.9rem', padding: '8px 16px' }}
          >
            View All <ArrowRight size={16} style={{ marginLeft: '6px' }} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {recentActions.map((action, idx) => (
            <motion.div
              key={action.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: idx * 0.03 }}
              onClick={() => handleActionClick(action)}
              style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                padding: '16px 20px',
                background: action.status === 'BLOCKED' ? 'rgba(220,53,69,0.03)' : 
                          action.status === 'PENDING' ? 'rgba(245,166,35,0.03)' : 'white',
                borderRadius: '12px',
                border: `1px solid ${action.status === 'BLOCKED' ? 'rgba(220,53,69,0.2)' : 
                              action.status === 'PENDING' ? 'rgba(245,166,35,0.2)' : 'var(--border-subtle)'}`,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = 'var(--accent-cyan)';
                e.currentTarget.style.background = 'rgba(10,130,118,0.02)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = action.status === 'BLOCKED' ? 'rgba(220,53,69,0.2)' : 
                                              action.status === 'PENDING' ? 'rgba(245,166,35,0.2)' : 'var(--border-subtle)';
                e.currentTarget.style.background = action.status === 'BLOCKED' ? 'rgba(220,53,69,0.03)' : 
                                              action.status === 'PENDING' ? 'rgba(245,166,35,0.03)' : 'white';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ 
                  width: '44px', height: '44px', 
                  borderRadius: '10px', 
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  background: action.status === 'BLOCKED' ? 'rgba(220,53,69,0.1)' : 
                            action.status === 'PENDING' ? 'rgba(245,166,35,0.1)' : 'rgba(25,135,84,0.1)'
                }}>
                  {action.status === 'BLOCKED' && <Shield size={20} color="var(--danger)" />}
                  {action.status === 'PENDING' && <AlertTriangle size={20} color="var(--warning)" />}
                  {action.status === 'APPROVED' && <CheckCircle size={20} color="var(--success)" />}
                </div>
                <div>
                  <p style={{ margin: 0, fontWeight: 600, color: 'var(--text-primary)', fontSize: '1rem' }}>
                    {action.action_type}
                  </p>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Users size={14} /> Agent #{action.agent_id}
                    <span style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'var(--text-muted)' }}></span>
                    <Clock size={14} /> {new Date(action.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                <div style={{ textAlign: 'right' }}>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)' }}>Risk Score</p>
                  <p style={{ 
                    margin: 0, 
                    fontWeight: 700, 
                    fontSize: '1.1rem',
                    color: action.final_risk_score > 0.7 ? 'var(--danger)' : 
                          action.final_risk_score > 0.3 ? 'var(--warning)' : 'var(--success)'
                  }}>
                    {action.final_risk_score?.toFixed(2) || '0.00'}
                  </p>
                </div>
                <span className={`badge badge-${action.status.toLowerCase()}`} style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
                  {action.status}
                </span>
                <Eye size={18} color="var(--text-muted)" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Action Detail Modal */}
      {showModal && selectedAction && (
        <div style={{ 
          position: 'fixed', inset: 0, 
          background: 'rgba(0,0,0,0.5)', 
          display: 'flex', alignItems: 'center', justifyContent: 'center', 
          zIndex: 1000, padding: '24px'
        }} onClick={() => setShowModal(false)}>
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card" 
            style={{ width: '100%', maxWidth: '700px', maxHeight: '90vh', overflowY: 'auto' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', paddingBottom: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
              <div>
                <h2 style={{ margin: 0, color: 'var(--text-primary)', fontSize: '1.5rem' }}>Action Details</h2>
                <p style={{ margin: '4px 0 0 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  {selectedAction.action_type} • Agent #{selectedAction.agent_id}
                </p>
              </div>
              <button onClick={() => setShowModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.5rem', color: 'var(--text-muted)' }}>×</button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Quick Info */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div style={{ padding: '16px', background: '#FAFAFA', borderRadius: '10px' }}>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Status</p>
                  <span className={`badge badge-${selectedAction.status.toLowerCase()}`}>{selectedAction.status}</span>
                </div>
                <div style={{ padding: '16px', background: '#FAFAFA', borderRadius: '10px' }}>
                  <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Timestamp</p>
                  <p style={{ margin: 0, color: 'var(--text-primary)', fontWeight: 500 }}>
                    {new Date(selectedAction.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Risk Scores */}
              <div>
                <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={18} /> Risk Breakdown
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: '#FAFAFA', borderRadius: '8px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Isolation Forest (ML)</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedAction.isolation_forest_score?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: '#FAFAFA', borderRadius: '8px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Rule Engine</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedAction.rule_engine_score?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', background: '#FAFAFA', borderRadius: '8px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>Context Scorer</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{selectedAction.context_score?.toFixed(2) || '0.00'}</span>
                  </div>
                  <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '12px', marginTop: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Final Risk Score</span>
                    <span style={{ 
                      fontWeight: 700, 
                      fontSize: '1.3rem',
                      color: selectedAction.final_risk_score > 0.7 ? 'var(--danger)' : 
                            selectedAction.final_risk_score > 0.3 ? 'var(--warning)' : 'var(--success)'
                    }}>
                      {selectedAction.final_risk_score?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Reason */}
              <div>
                <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)' }}>Reason</h4>
                <p style={{ 
                  margin: 0, 
                  padding: '14px', 
                  background: '#FAFAFA', 
                  borderRadius: '10px', 
                  border: '1px solid var(--border-subtle)', 
                  color: 'var(--text-primary)',
                  lineHeight: '1.6'
                }}>
                  {selectedAction.reason || 'No reason provided'}
                </p>
              </div>

              {/* Payload */}
              <div>
                <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)' }}>Payload</h4>
                <pre style={{ 
                  margin: 0, 
                  padding: '14px', 
                  background: '#FAFAFA', 
                  borderRadius: '10px', 
                  border: '1px solid var(--border-subtle)', 
                  fontSize: '0.85rem', 
                  overflowX: 'auto',
                  color: 'var(--text-primary)'
                }}>
                  {JSON.stringify(selectedAction.payload, null, 2)}
                </pre>
              </div>

              {/* Hash Chain */}
              <div>
                <h4 style={{ margin: '0 0 12px 0', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Database size={18} /> Audit Hash Chain
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', padding: '14px', background: '#FAFAFA', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)', minWidth: '110px' }}>Previous Hash:</span>
                    <code style={{ fontSize: '0.8rem', color: 'var(--text-primary)', wordBreak: 'break-all' }}>{selectedAction.previous_hash || 'None (Genesis)'}</code>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)', minWidth: '110px' }}>Current Hash:</span>
                    <code style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)', wordBreak: 'break-all', fontWeight: 600 }}>{selectedAction.current_hash || 'N/A'}</code>
                  </div>
                </div>
              </div>

              {/* Navigation Buttons */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '8px' }}>
                <button 
                  onClick={() => { setShowModal(false); navigate('/agent-mesh'); }}
                  className="btn"
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                >
                  <Users size={16} /> View Agent
                </button>
                <button 
                  onClick={() => { setShowModal(false); navigate('/audit-trail'); }}
                  className="btn"
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
                >
                  <Database size={16} /> View Audit
                </button>
              </div>

              {/* Manual Action Buttons */}
              <div style={{ 
                borderTop: '1px solid var(--border-subtle)', 
                paddingTop: '20px', 
                marginTop: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px'
              }}>
                <h4 style={{ margin: 0, color: 'var(--text-primary)' }}>Manual Actions</h4>
                <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  Take manual control of this agent action
                </p>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                  <button 
                    onClick={() => handleManualAction('BLOCK')}
                    className="btn"
                    style={{ 
                      background: 'rgba(220,53,69,0.05)', 
                      border: '1px solid var(--danger)', 
                      color: 'var(--danger)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px'
                    }}
                  >
                    <XCircle size={16} /> Block
                  </button>
                  <button 
                    onClick={() => handleManualAction('ALLOW')}
                    className="btn"
                    style={{ 
                      background: 'rgba(25,135,84,0.05)', 
                      border: '1px solid var(--success)', 
                      color: 'var(--success)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px'
                    }}
                  >
                    <ThumbsUp size={16} /> Allow
                  </button>
                  <button 
                    onClick={() => handleManualAction('ALERT')}
                    className="btn"
                    style={{ 
                      background: 'rgba(245,166,35,0.05)', 
                      border: '1px solid var(--warning)', 
                      color: 'var(--warning)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px'
                    }}
                  >
                    <Bell size={16} /> Alert
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
};

export default DashboardOverview;
