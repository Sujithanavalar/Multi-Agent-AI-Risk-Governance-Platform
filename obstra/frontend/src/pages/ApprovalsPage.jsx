import React from 'react';
import PendingApprovals from '../components/PendingApprovals';
import axios from 'axios';

const ApprovalsPage = () => {
  const [pending, setPending] = React.useState([]);

  const fetchData = async () => {
    try {
      const pendingRes = await axios.get('http://localhost:8000/consensus/pending');
      setPending(pendingRes.data);
    } catch (err) {
      console.error(err);
    }
  };

  React.useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Approvals (Consensus)</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Review and approve pending agent actions</p>
      </header>

      <PendingApprovals pending={pending} refreshData={fetchData} />
    </div>
  );
};

export default ApprovalsPage;
