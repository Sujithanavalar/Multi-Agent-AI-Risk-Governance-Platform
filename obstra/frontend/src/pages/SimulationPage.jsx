import React, { useState } from 'react';
import { Play, Pause, AlertTriangle, CheckCircle } from 'lucide-react';
import axios from 'axios';

const SimulationPage = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [simulationLog, setSimulationLog] = useState([]);

  const addLogEntry = (entry) => {
    setSimulationLog(prev => [...prev, { ...entry, id: Date.now(), time: new Date().toLocaleTimeString() }]);
  };

  const startSimulation = async () => {
    setIsRunning(true);
    addLogEntry({ type: 'info', message: 'Hospital simulation started' });

    try {
      // Step 1: Register Diagnosis Agent
      addLogEntry({ type: 'info', message: 'Registering DiagnosisAgent...' });
      const agentRes = await axios.post('http://localhost:8001/agents/register', {
        name: 'Simulation-DiagnosisAgent',
        framework: 'langchain',
        owner: 'HospitalA',
        permissions: ['READ_FILE', 'DIAGNOSE']
      });
      addLogEntry({ type: 'success', message: 'DiagnosisAgent registered successfully' });

      await new Promise(r => setTimeout(r, 1000));
      addLogEntry({ type: 'info', message: 'DiagnosisAgent reading patient file...' });
      
      // Step 2: Log READ_FILE
      await new Promise(r => setTimeout(r, 1000));
      const decision1 = await axios.post('http://localhost:8001/actions/evaluate', {
        action_type: 'READ_FILE',
        payload: { file: 'patient_123.pdf' }
      }, {
        headers: { 'X-Obstra-Agent-Token': agentRes.data.token }
      });
      addLogEntry({ type: decision1.data.status === 'APPROVED' ? 'success' : 'danger', message: `READ_FILE: ${decision1.data.status}` });

      // Step 3: Attempt dangerous dosage
      await new Promise(r => setTimeout(r, 1000));
      addLogEntry({ type: 'warning', message: 'PrescriptionAgent attempting: DOSAGE 3000mg' });
      
      await new Promise(r => setTimeout(r, 1000));
      const decision2 = await axios.post('http://localhost:8001/actions/evaluate', {
        action_type: 'DOSAGE',
        payload: { dosage: 3000, patient: 'patient_123' }
      }, {
        headers: { 'X-Obstra-Agent-Token': agentRes.data.token }
      });
      addLogEntry({ type: 'danger', message: `DOSAGE 3000mg: ${decision2.data.status} - ${decision2.data.message}` });

    } catch (err) {
      addLogEntry({ type: 'error', message: `Simulation error: ${err.message}` });
      console.error(err);
    }

    setIsRunning(false);
  };

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Simulation (Demo)</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Interactive multi-agent hospital workflow simulation</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Simulation Controls</h3>
          <button 
            onClick={startSimulation} 
            disabled={isRunning}
            className={isRunning ? 'btn' : 'btn btn-primary'}
            style={{ width: '100%', marginBottom: '16px' }}
          >
            {isRunning ? <><Pause size={16} /> Running...</> : <><Play size={16} /> Start Simulation</>}
          </button>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', background: '#FAFAFA', borderRadius: '8px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: isRunning ? 'var(--success)' : 'var(--text-muted)' }}></div>
              <span style={{ color: 'var(--text-primary)' }}>
                {isRunning ? 'Simulation Active' : 'Ready to Start'}
              </span>
            </div>
          </div>
        </div>

        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Agents</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {['DiagnosisAgent', 'PrescriptionAgent', 'PharmacyAgent', 'ValidatorAgent'].map((agent, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ 
                  width: '32px', height: '32px', borderRadius: '50%', 
                  background: 'var(--accent-cyan)', 
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'white', fontWeight: 600, fontSize: '0.8rem'
                }}>
                  {agent[0]}
                </div>
                <span style={{ color: 'var(--text-primary)' }}>{agent}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="glass-card">
        <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Simulation Log</h3>
        <div style={{ 
          background: '#FAFAFA', 
          borderRadius: '8px', 
          padding: '16px', 
          maxHeight: '400px',
          overflowY: 'auto',
          fontFamily: 'monospace',
          fontSize: '0.85rem'
        }}>
          {simulationLog.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>
              Start the simulation to see the log
            </p>
          ) : (
            simulationLog.map(entry => (
              <div key={entry.id} style={{ marginBottom: '8px', display: 'flex', gap: '12px' }}>
                <span style={{ color: 'var(--text-muted)' }}>[{entry.time}]</span>
                {entry.type === 'success' && <CheckCircle size={14} color="var(--success)" />}
                {entry.type === 'danger' && <AlertTriangle size={14} color="var(--danger)" />}
                {entry.type === 'warning' && <AlertTriangle size={14} color="var(--warning)" />}
                <span style={{ 
                  color: entry.type === 'success' ? 'var(--success)' : 
                         entry.type === 'danger' ? 'var(--danger)' : 
                         entry.type === 'warning' ? 'var(--warning)' : 'var(--text-primary)'
                }}>
                  {entry.message}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default SimulationPage;
