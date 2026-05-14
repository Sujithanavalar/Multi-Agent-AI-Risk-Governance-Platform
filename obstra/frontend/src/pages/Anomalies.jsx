import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import Header from '../components/Header';

const Anomalies = () => {
  const { anomalies, fetchAnomalies } = useStore();

  useEffect(() => fetchAnomalies(), [fetchAnomalies]);

  const handleResolve = async (id) => {
    try {
      await api.post(`/anomalies/${id}/resolve`);
      await fetchAnomalies();
    } catch (err) { console.error(err); }
  };

  const formatDate = (dateStr) => new Date(dateStr).toLocaleString();

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Anomalies</h1>
          <p>Detected unusual agent behavior</p>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Status</th>
                <th>Detected</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.map((anom) => (
                <tr key={anom.id}>
                  <td>{anom.agent_name || 'Unknown'}</td>
                  <td>{anom.anomaly_type}</td>
                  <td>
                    <span className={`badge ${anom.severity === 'high' ? 'badge-danger' : anom.severity === 'medium' ? 'badge-warning' : 'badge-info'}`}>
                      {anom.severity}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${anom.resolved ? 'badge-muted' : 'badge-danger'}`}>
                      {anom.resolved ? 'Resolved' : 'Active'}
                    </span>
                  </td>
                  <td>{formatDate(anom.created_at)}</td>
                  <td>
                    {!anom.resolved && (
                      <button onClick={() => handleResolve(anom.id)} className="btn btn-success btn-sm">Resolve</button>
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

export default Anomalies;
