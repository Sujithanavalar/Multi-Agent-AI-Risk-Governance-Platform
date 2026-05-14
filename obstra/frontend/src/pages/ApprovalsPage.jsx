import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Check, X } from 'lucide-react';

const ApprovalsPage = () => {
  const { approvals, fetchApprovals } = useStore();

  useEffect(() => {
    fetchApprovals();
  }, [fetchApprovals]);

  const handleApprove = async (id) => {
    try {
      await api.post(`/approvals/${id}/approve`);
      await fetchApprovals();
    } catch (error) {
      console.error('Error approving:', error);
    }
  };

  const handleReject = async (id) => {
    try {
      await api.post(`/approvals/${id}/reject`);
      await fetchApprovals();
    } catch (error) {
      console.error('Error rejecting:', error);
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      pending: 'bg-yellow-50 text-obstra-warning',
      approved: 'bg-green-50 text-obstra-success',
      rejected: 'bg-red-50 text-obstra-danger',
      expired: 'bg-gray-100 text-obstra-text-muted',
    };
    return styles[status] || 'bg-gray-100 text-obstra-text-muted';
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-obstra-text mb-2">Approvals</h1>
        <p className="text-obstra-text-secondary">Review and approve/reject pending agent actions</p>
      </div>

      <div className="bg-white rounded-obstra shadow-card border border-obstra-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-obstra-table-header">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">ID</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Agent</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Risk Score</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Requested At</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-obstra-border">
              {approvals.map(approval => (
                <tr key={approval.id} className="hover:bg-obstra-table-header transition-colors">
                  <td className="px-6 py-4 text-sm text-obstra-text-secondary font-mono">{approval.id.slice(0, 8)}...</td>
                  <td className="px-6 py-4 text-obstra-text">{approval.agent_id?.slice(0, 8)}...</td>
                  <td className="px-6 py-4">
                    <span className={`font-medium ${
                      approval.risk_score < 0.3 ? 'text-obstra-success' :
                      approval.risk_score < 0.7 ? 'text-obstra-warning' : 'text-obstra-danger'
                    }`}>
                      {(approval.risk_score * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 text-xs font-medium rounded-obstra ${getStatusBadge(approval.status)}`}>
                      {approval.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-obstra-text-secondary">
                    {new Date(approval.requested_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4">
                    {approval.status === 'pending' && (
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleApprove(approval.id)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-obstra-success text-white rounded-obstra text-sm font-medium hover:opacity-90 transition-opacity"
                        >
                          <Check size={14} />
                          Approve
                        </button>
                        <button
                          onClick={() => handleReject(approval.id)}
                          className="flex items-center gap-1 px-3 py-1.5 bg-obstra-danger text-white rounded-obstra text-sm font-medium hover:opacity-90 transition-opacity"
                        >
                          <X size={14} />
                          Reject
                        </button>
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

export default ApprovalsPage;
