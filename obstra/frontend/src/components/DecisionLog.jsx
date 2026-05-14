import React, { useState } from 'react';
import { Database, ChevronDown, ChevronUp } from 'lucide-react';
import { motion } from 'framer-motion';

const DecisionLog = ({ logs, onReview }) => {
  const [expandedId, setExpandedId] = useState(null);

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="glass-card" id="audit">
      <h3 style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Database size={20} color="var(--primary-terracotta)" />
        Cryptographic audit log
      </h3>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Agent</th>
              <th>Action</th>
              <th>Risk</th>
              <th>Status</th>
              <th style={{ width: 120 }}>Review</th>
              <th> </th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <React.Fragment key={log.id}>
                <motion.tr
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Math.min(index * 0.015, 0.4) }}
                  style={{ cursor: 'pointer' }}
                  onClick={() => toggleExpand(log.id)}
                >
                  <td style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>#{log.id}</td>
                  <td style={{ color: 'var(--accent-cyan)', fontWeight: 500 }}>Agent {log.agent_id}</td>
                  <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{log.action_type}</td>
                  <td>{log.final_risk_score ?? '—'}</td>
                  <td>
                    <span className={`badge badge-${String(log.status).toLowerCase()}`}>{log.status}</span>
                  </td>
                  <td onClick={(e) => e.stopPropagation()}>
                    {onReview && (
                      <button type="button" className="btn" style={{ padding: '6px 10px', fontSize: '0.8rem' }} onClick={() => onReview(log)}>
                        Decide
                      </button>
                    )}
                  </td>
                  <td>{expandedId === log.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}</td>
                </motion.tr>
                {expandedId === log.id && (
                  <motion.tr initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <td colSpan="7">
                      <div
                        style={{
                          padding: '20px',
                          background: '#FAFAFA',
                          borderBottom: '1px solid var(--border-subtle)',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                          gap: '20px',
                        }}
                      >
                        <div>
                          <h4 style={{ marginBottom: '10px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Reason</h4>
                          <p style={{ margin: 0, fontSize: '0.9rem' }}>{log.reason}</p>
                        </div>

                        <div>
                          <h4 style={{ marginBottom: '10px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Payload</h4>
                          <pre
                            style={{
                              margin: 0,
                              fontSize: '0.8rem',
                              background: 'white',
                              padding: '12px',
                              borderRadius: '6px',
                              border: '1px solid var(--border-subtle)',
                              overflow: 'auto',
                              maxHeight: '180px',
                            }}
                          >
                            {JSON.stringify(log.payload, null, 2)}
                          </pre>
                        </div>

                        {log.current_hash && (
                          <div>
                            <h4 style={{ marginBottom: '10px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Hash chain</h4>
                            <div style={{ fontFamily: 'monospace', fontSize: '0.72rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                              <div>
                                <span style={{ color: 'var(--text-muted)' }}>Previous</span>
                                <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{log.previous_hash}</div>
                              </div>
                              <div>
                                <span style={{ color: 'var(--text-muted)' }}>Current</span>
                                <div style={{ color: 'var(--accent-cyan)', wordBreak: 'break-all' }}>{log.current_hash}</div>
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
