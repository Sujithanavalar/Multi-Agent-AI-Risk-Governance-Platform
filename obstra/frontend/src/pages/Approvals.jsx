import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import Header from '../components/Header';

const Approvals = () => {
  const { approvals, fetchApprovals } = useStore();

  useEffect(() => fetchApprovals(), [fetchApprovals]);

  const handleApprove = async (id) => {
    try {
      await api.post(`/approvals/${id}/approve`);
      await fetchApprovals();
    } catch (err) { console.error(err); }
  };

  const handleReject = async (id) => {
    try {
      await api.post(`/approvals/${id}/reject`);
      await fetchApprovals();
    } catch (err) { console.error(err); }
  };

  const formatDate = (dateStr) => new Date(dateStr).toLocaleString();

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Approvals</h1>
          <p>Review and approve agent actions</p>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Action</th>
                <th>Risk</th>
                <th>Status</th>
                <th>Requested</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {approvals.map((app) => (
                <tr key={app.id}>
                  <td>{app.agent_name || 'Unknown'}</td>
                  <td style={{ maxWidth: 300 }}>{app.request_summary}</td>
                  <td>
                    <span className={`badge ${app.risk_score > 0.7 ? 'badge-danger' : app.risk_score > 0.4 ? 'badge-warning' : 'badge-info'}`}>
                      {(app.risk_score * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${app.status === 'pending' ? 'badge-warning' : app.status === 'approved' ? 'badge-success' : 'badge-muted'}`}>
                      {app.status}
                    </span>
                  </td>
                  <td>{formatDate(app.created_at)}</td>
                  <td>
                    {app.status === 'pending' && (
                      <div style={{ display: 'flex', gap: 8 }}>
                        <button onClick={() => handleApprove(app.id)} className="btn btn-success btn-sm">Approve</button>
                        <button onClick={() => handleReject(app.id)} className="btn btn-danger btn-sm">Reject</button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Approvals;
