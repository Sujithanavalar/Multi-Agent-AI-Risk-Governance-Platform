import React, { useState } from 'react';
import api from '../api/client';
import { AlertTriangle, Check, X, Eye, FileCode } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PendingApprovals = ({ pending, refreshData }) => {
  const [expandedItem, setExpandedItem] = useState(null);

  const handleResolve = async (id, approved) => {
    try {
      await api.post('/consensus/resolve', {
        consensus_id: id,
        approved: approved,
        reviewer: "AdminDashboard",
        reason: approved ? "Manually approved by operator" : "Blocked by operator"
      });
      refreshData();
    } catch (err) {
      console.error(err);
      alert("Failed to resolve action.");
    }
  };

  return (
    <div className="glass-card" id="pending">
      <h3 style={{ marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--warning)' }}>
        <AlertTriangle size={20} />
        Consensus Queue (Medium Risk)
      </h3>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <AnimatePresence>
          {pending.length === 0 ? (
            <motion.p 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              style={{ color: 'var(--text-secondary)', textAlign: 'center', padding: '24px' }}
            >
              No pending actions requiring consensus.
            </motion.p>
          ) : (
            pending.map((item) => (
              <motion.div 
                key={item.consensus_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -100 }}
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid var(--border-highlight)',
                  borderRadius: '8px',
                  overflow: 'hidden'
                }}
              >
                <div 
                  style={{
                    padding: '16px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    cursor: 'pointer'
                  }}
                  onClick={() => setExpandedItem(expandedItem === item.consensus_id ? null : item.consensus_id)}
                >
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{item.action_type}</span>
                      <span className="badge badge-pending">Risk: {item.risk_score}</span>
                    </div>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                      Agent ID: {item.agent_id}
                    </p>
                  </div>
                  
                  <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <Eye size={16} color="var(--text-secondary)" />
                  </div>
                </div>

                <AnimatePresence>
                  {expandedItem === item.consensus_id && (
                    <motion.div 
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      style={{
                        borderTop: '1px solid var(--border-subtle)',
                        padding: '16px',
                        background: '#FAFAFA'
                      }}
                    >
                      <div style={{ marginBottom: '16px' }}>
                        <h4 style={{ 
                          marginBottom: '12px', 
                          fontSize: '0.9rem', 
                          color: 'var(--text-secondary)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px'
                        }}>
                          <FileCode size={16} />
                          Action Payload
                        </h4>
                        <pre style={{ 
                          margin: 0, 
                          fontSize: '0.8rem', 
                          background: 'white', 
                          padding: '12px', 
                          borderRadius: '4px',
                          border: '1px solid var(--border-subtle)',
                          overflow: 'auto',
                          maxHeight: '200px'
                        }}>
                          {JSON.stringify(item.payload, null, 2)}
                        </pre>
                      </div>

                      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                        <button onClick={() => handleResolve(item.consensus_id, true)} className="btn btn-success">
                          <Check size={16} /> Approve
                        </button>
                        <button onClick={() => handleResolve(item.consensus_id, false)} className="btn btn-danger">
                          <X size={16} /> Reject
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default PendingApprovals;
