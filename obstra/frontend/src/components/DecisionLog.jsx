import React, { useState } from 'react';
import { Database, ChevronDown, ChevronUp } from 'lucide-react';
import { motion } from 'framer-motion';

const DecisionLog = ({ logs }) => {
  const [expandedId, setExpandedId] = useState(null);

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="glass-card" id="audit">
      <h3 style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Database size={20} color="var(--primary-terracotta)" />
        Cryptographic Audit Log
      </h3>
      
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Log ID</th>
              <th>Agent ID</th>
              <th>Action Type</th>
              <th>Risk Breakdown</th>
              <th>Status</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <React.Fragment key={log.id}>
                <motion.tr 
                  key={log.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.02 }}
                  style={{ cursor: 'pointer' }}
                  onClick={() => toggleExpand(log.id)}
                >
                  <td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>#{log.id}</td>
                  <td style={{ color: 'var(--accent-cyan)', fontWeight: 500 }}>Agent {log.agent_id}</td>
                  <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{log.action_type}</td>
                  <td>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <span style={{ fontSize: '0.75rem' }}>Final: {log.final_risk_score}</span>
                      {log.breakdown && (
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          Rule: {log.breakdown.rule_engine?.score} | ML: {log.breakdown.isolation_forest?.score}
                        </div>
                      )}
                    </div>
                  </td>
                  <td>
                    <span className={`badge badge-${log.status.toLowerCase()}`}>
                      {log.status}
                    </span>
                  </td>
                  <td>
                    {expandedId === log.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </td>
                </motion.tr>
                {expandedId === log.id && (
                  <motion.tr 
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                  >
                    <td colSpan="6">
                      <div style={{ 
                        padding: '20px', 
                        background: '#FAFAFA',
                        borderBottom: '1px solid var(--border-subtle)',
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                        gap: '24px'
                      }}>
                        <div>
                          <h4 style={{ marginBottom: '12px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Reason</h4>
                          <p style={{ margin: 0, fontSize: '0.9rem' }}>{log.reason}</p>
                        </div>
                        
                        <div>
                          <h4 style={{ marginBottom: '12px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Payload</h4>
                          <pre style={{ 
                            margin: 0, 
                            fontSize: '0.8rem', 
                            background: 'white', 
                            padding: '12px', 
                            borderRadius: '4px',
                            border: '1px solid var(--border-subtle)',
                            overflow: 'auto',
                            maxHeight: '150px'
                          }}>
                            {JSON.stringify(log.payload, null, 2)}
                          </pre>
                        </div>
                        
                        {log.current_hash && (
                          <div>
                            <h4 style={{ marginBottom: '12px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Hash Chain</h4>
                            <div style={{ 
                              fontFamily: 'monospace', 
                              fontSize: '0.75rem',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: '8px'
                            }}>
                              <div>
                                <span style={{ color: 'var(--text-muted)' }}>Previous:</span>
                                <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                                  {log.previous_hash}
                                </div>
                              </div>
                              <div>
                                <span style={{ color: 'var(--text-muted)' }}>Current:</span>
                                <div style={{ color: 'var(--accent-cyan)', wordBreak: 'break-all' }}>
                                  {log.current_hash}
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </td>
                  </motion.tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DecisionLog;
