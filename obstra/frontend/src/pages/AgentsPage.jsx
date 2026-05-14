import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Plus, Pause, Play, Bot } from 'lucide-react';

const AgentsPage = () => {
  const { agents, fetchAgents } = useStore();
  const [showModal, setShowModal] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    provider: 'openai',
    model: 'gpt-4',
    api_key: '',
    base_url: '',
    system_prompt: '',
    agent_role: '',
  });

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const handleCreateAgent = async (e) => {
    e.preventDefault();
    try {
      await api.post('/agents', formData);
      await fetchAgents();
      setShowModal(false);
      setFormData({ name: '', provider: 'openai', model: 'gpt-4', api_key: '', base_url: '', system_prompt: '', agent_role: '' });
    } catch (error) {
      console.error('Error creating agent:', error);
    }
  };

  const handlePause = async (agentId) => {
    try {
      await api.post(`/agents/${agentId}/pause`);
      await fetchAgents();
    } catch (error) {
      console.error('Error pausing agent:', error);
    }
  };

  const handleRestart = async (agentId) => {
    try {
      await api.post(`/agents/${agentId}/restart`);
      await fetchAgents();
    } catch (error) {
      console.error('Error restarting agent:', error);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-obstra-text mb-2">Agents</h1>
          <p className="text-obstra-text-secondary">Manage your AI agents</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 bg-obstra-primary text-white px-4 py-2 rounded-obstra font-medium hover:bg-obstra-primary-hover transition-colors"
        >
          <Plus size={18} />
          Add Agent
        </button>
      </div>

      <div className="bg-white rounded-obstra shadow-card border border-obstra-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-obstra-table-header">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Agent</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Provider</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Actions</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Risk Score</th>
                <th className="px-6 py-3 text-left text-xs font-semibold text-obstra-text-secondary uppercase tracking-wider">Total Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-obstra-border">
              {agents.map(agent => (
                <tr key={agent.id} className="hover:bg-obstra-table-header transition-colors">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-obstra-primary/10 rounded-obstra flex items-center justify-center">
                        <Bot className="text-obstra-primary" size={20} />
                      </div>
                      <div>
                        <p className="font-medium text-obstra-text">{agent.name}</p>
                        <p className="text-xs text-obstra-text-muted">{agent.model}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="capitalize text-obstra-text">{agent.provider}</span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 text-xs font-medium rounded-obstra ${
                      agent.status === 'active' ? 'bg-green-50 text-obstra-success' :
                      agent.status === 'paused' ? 'bg-gray-100 text-obstra-text-secondary' :
                      'bg-red-50 text-obstra-danger'
                    }`}>
                      {agent.status}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      {agent.status === 'active' ? (
                        <button
                          onClick={() => handlePause(agent.id)}
                          className="p-2 text-obstra-text-secondary hover:text-obstra-warning hover:bg-obstra-table-header rounded-obstra transition-colors"
                        >
                          <Pause size={16} />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleRestart(agent.id)}
                          className="p-2 text-obstra-text-secondary hover:text-obstra-success hover:bg-obstra-table-header rounded-obstra transition-colors"
                        >
                          <Play size={16} />
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-obstra-border rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            agent.average_risk_score < 0.3 ? 'bg-obstra-success' :
                            agent.average_risk_score < 0.7 ? 'bg-obstra-warning' : 'bg-obstra-danger'
                          }`}
                          style={{ width: `${agent.average_risk_score * 100}%` }}
                        />
                      </div>
                      <span className={`text-sm font-medium ${
                        agent.average_risk_score < 0.3 ? 'text-obstra-success' :
                        agent.average_risk_score < 0.7 ? 'text-obstra-warning' : 'text-obstra-danger'
                      }`}>
                        {(agent.average_risk_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-obstra-text">{agent.total_actions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-obstra shadow-xl w-full max-w-lg p-6">
            <h2 className="text-xl font-bold text-obstra-text mb-4">Add New Agent</h2>
            <form onSubmit={handleCreateAgent} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-obstra-text mb-1">Agent Name</label>
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
                  <label className="block text-sm font-medium text-obstra-text mb-1">Provider</label>
                  <select
                    value={formData.provider}
                    onChange={e => setFormData({ ...formData, provider: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="google">Google Gemini</option>
                    <option value="ollama">Ollama</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-obstra-text mb-1">Model</label>
                  <input
                    type="text"
                    value={formData.model}
                    onChange={e => setFormData({ ...formData, model: e.target.value })}
                    className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                    required
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-obstra-text mb-1">API Key</label>
                <input
                  type="password"
                  value={formData.api_key}
                  onChange={e => setFormData({ ...formData, api_key: e.target.value })}
                  className="w-full px-3 py-2 border border-obstra-border rounded-obstra focus:outline-none focus:ring-2 focus:ring-obstra-secondary"
                  required
                />
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
                  Create Agent
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentsPage;
