import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { AlertTriangle, CheckCircle } from 'lucide-react';

const AnomaliesPage = () => {
  const { anomalies, fetchAnomalies } = useStore();

  useEffect(() => {
    fetchAnomalies();
  }, [fetchAnomalies]);

  const handleResolve = async (id) => {
    try {
      await api.post(`/anomalies/${id}/resolve`);
      await fetchAnomalies();
    } catch (error) {
      console.error('Error resolving anomaly:', error);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'low': return 'text-obstra-success bg-green-50';
      case 'medium': return 'text-obstra-warning bg-yellow-50';
      case 'high': return 'text-obstra-danger bg-red-50';
      case 'critical': return 'text-obstra-danger bg-red-100';
      default: return 'text-obstra-text-secondary bg-gray-100';
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-obstra-text mb-2">Anomalies</h1>
        <p className="text-obstra-text-secondary">Detected anomalies and security threats</p>
      </div>

      <div className="bg-white rounded-obstra shadow-card border border-obstra-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-obstra-table-header">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Severity</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Description</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Detected At</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-obstra-border">
              {anomalies.map(anomaly => (
                <tr key={anomaly.id} className="hover:bg-obstra-table-header transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <AlertTriangle size={16} className={anomaly.resolved ? 'text-obstra-text-muted' : 'text-obstra-warning'} />
                      <span className="text-obstra-text font-medium">{anomaly.anomaly_type}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 text-xs font-medium rounded-obstra ${getSeverityColor(anomaly.severity)}`}>
                      {anomaly.severity}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-obstra-text-secondary">{anomaly.description || 'No description'}</td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 text-xs font-medium rounded-obstra ${
                      anomaly.resolved ? 'bg-green-50 text-obstra-success' : 'bg-yellow-50 text-obstra-warning'
                    }`}>
                      {anomaly.resolved ? 'Resolved' : 'Open'}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-obstra-text-secondary">
                    {new Date(anomaly.detected_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4">
                    {!anomaly.resolved && (
                      <button
                        onClick={() => handleResolve(anomaly.id)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-obstra-secondary text-white rounded-obstra text-sm font-medium hover:bg-obstra-secondary-hover transition-colors"
                      >
                        <CheckCircle size={14} />
                        Resolve
                      </button>
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

export default AnomaliesPage;
