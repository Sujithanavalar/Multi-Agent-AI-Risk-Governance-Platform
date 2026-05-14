import React from 'react';
import DecisionLog from '../components/DecisionLog';
import axios from 'axios';

const AuditTrailPage = () => {
  const [logs, setLogs] = React.useState([]);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        const logsRes = await axios.get('http://localhost:8000/audit/logs');
        setLogs(logsRes.data);
      } catch (err) {
        console.error(err);
      }
    };
    fetchData();
  }, []);

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Audit Trail (Blockchain)</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Tamper-proof audit log with hash chaining</p>
      </header>

      <DecisionLog logs={logs} />
    </div>
  );
};

export default AuditTrailPage;
