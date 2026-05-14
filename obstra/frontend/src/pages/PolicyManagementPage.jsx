import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Plus, Edit, Trash2, ToggleLeft, ToggleRight } from 'lucide-react';

const CATEGORIES = ['healthcare', 'finance', 'database', 'api_security', 'prompt_safety', 'access_control', 'contextual', 'time_based'];

const PolicyManagementPage = () => {
  const { rules, fetchRules } = useStore();
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    category: 'access_control',
    condition_field: '',
    condition_operator: 'contains',
    condition_value: '',
    action: 'review',
    severity: 'medium',
    enabled: true,
  });

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  const handleCreateRule = async (e) => {
    e.preventDefault();
    try {
      await api.post('/rules', formData);
      await fetchRules();
      setShowModal(false);
      setFormData({
        name: '', category: 'access_control', condition_field: '',
        condition_operator: 'contains', condition_value: '', action: 'review',
        severity: 'medium', enabled: true,
      });
    } catch (error) {
      console.error('Error creating rule:', error);
    }
  };

  const handleToggleRule = async (id) => {
    try {
      await api.patch(`/rules/${id}/toggle`);
      await fetchRules();
    } catch (error) {
      console.error('Error toggling rule:', error);
    }
  };

  const handleDeleteRule = async (id) => {
    if (window.confirm('Are you sure you want to delete this rule?')) {
      try {
        await api.delete(`/rules/${id}`);
        await fetchRules();
      } catch (error) {
        console.error('Error deleting rule:', error);
      }
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-obstra-text mb-2">Policies & Rules</h1>
          <p className="text-obstra-text-secondary">Manage your governance rules and policies</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 bg-obstra-primary text-white px-4 py-2 rounded-obstra font-medium hover:bg-obstra-primary-hover transition-colors"
        >
          <Plus size={18} />
          Add Rule
        </button>
      </div>

      <div className="bg-white rounded-obstra shadow-card border border-obstra-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-obstra-table-header">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Name</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Category</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Condition</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Action</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Severity</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-obstra-border">
              {rules.map(rule => (
                <tr key={rule.id} className="hover:bg-obstra-table-header transition-colors">
                  <td className="px-6 py-4 font-medium text-obstra-text">{rule.name}</td>
                  <td className="px-6 py-4 text-obstra-text-secondary capitalize">{rule.category}</td>
                  <td className="px-6 py-4 text-obstra-text-secondary">
                    {rule.condition_field} {rule.condition_operator} "{rule.condition_value}"
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 text-xs font-medium rounded-obstra ${
                      rule.action === 'block' ? 'bg-red-50 text-obstra-danger' :
                      rule.action === 'review' ? 'bg-yellow-50 text-obstra-warning' :
                      'bg-green-50 text-obstra-success'
                    }`}>
                      {rule.action}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 text-xs font-medium rounded-obstra ${
                      rule.severity === 'critical' ? 'bg-red-50 text-obstra-danger' :
                      rule.severity === 'high' ? 'bg-orange-50 text-obstra-warning' :
                      rule.severity === 'medium' ? 'bg-yellow-50 text-obstra-warning' :
                      'bg-green-50 text-obstra-success'
                    }`}>
                      {rule.severity}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <button
                      onClick={() => handleToggleRule(rule.id)}
                      className="text-obstra-text-secondary hover:text-obstra-primary"
                    >
                      {rule.enabled ? <ToggleRight size={20} /> : <ToggleLeft size={20} />}
                    </button>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <button className="p-1 text-obstra-text-secondary hover:text-obstra-secondary rounded">
                        <Edit size={16} />
                      </button>
                      <button
                        onClick={() => handleDeleteRule(rule.id)}
                        className="p-1 text-obstra-text-secondary hover:text-obstra-danger rounded"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-obstra shadow-xl w-full max-w-lg p-6">
            <h2 className="text-xl font-bold text-obstra-text mb-4">Add New Rule</h2>
            <form onSubmit={handleCreateRule} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-obstra-text mb-1">Rule Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={e => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  required
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-obstra-text mb-1">Category</label>
                  <select
                    value={formData.category}
                    onChange={e => setFormData({ ...formData, category: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  >
                    {CATEGORIES.map(cat => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-obstra-text mb-1">Severity</label>
                  <select
                    value={formData.severity}
                    onChange={e => setFormData({ ...formData, severity: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-obstra-text mb-1">Condition Field</label>
                <input
                  type="text"
                  value={formData.condition_field}
                  onChange={e => setFormData({ ...formData, condition_field: e.target.value })}
                  className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  placeholder="e.g., prompt"
                  required
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-obstra-text mb-1">Operator</label>
                  <select
                    value={formData.condition_operator}
                    onChange={e => setFormData({ ...formData, condition_operator: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  >
                    <option value="contains">contains</option>
                    <option value="not_contains">not contains</option>
                    <option value=">">&gt;</option>
                    <option value="<">&lt;</option>
                    <option value="=">=</option>
                    <option value="regex">regex</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-obstra-text mb-1">Value</label>
                  <input
                    type="text"
                    value={formData.condition_value}
                    onChange={e => setFormData({ ...formData, condition_value: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-obstra-text mb-1">Action</label>
                <select
                  value={formData.action}
                  onChange={e => setFormData({ ...formData, action: e.target.value })}
                  className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                >
                  <option value="allow">Allow</option>
                  <option value="review">Review</option>
                  <option value="block">Block</option>
                </select>
              </div>
              <div className="flex gap-3 justify-end pt-4">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 border border-obstra-border rounded-obstra text-obstra-text hover:bg-obstra-table-header transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 bg-obstra-primary text-white rounded-obstra hover:bg-obstra-primary-hover transition-colors"
                >
                  Create Rule
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default PolicyManagementPage;
