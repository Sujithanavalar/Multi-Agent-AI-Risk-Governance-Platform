import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Plus, Pause, Play, Trash2, Bot } from 'lucide-react';
import Header from '../components/Header';

const Agents = () => {
  const { agents, fetchAgents } = useStore();
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({
    name: '', provider: 'openai', model: 'gpt-4', description: '',
    policies: [], rules: [], escalation_threshold: 0.7
  });

  useEffect(() => fetchAgents(), [fetchAgents]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await api.post('/agents', form);
      await fetchAgents();
      setShowModal(false);
      setForm({ name: '', provider: 'openai', model: 'gpt-4', description: '', policies: [], rules: [], escalation_threshold: 0.7 });
    } catch (err) { console.error(err); }
  };

  const handleToggleStatus = async (id, current) => {
    try {
      await api.put(`/agents/${id}`, { status: current === 'active' ? 'paused' : 'active' });
      await fetchAgents();
    } catch (err) { console.error(err); }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this agent?')) return;
    try {
      await api.delete(`/agents/${id}`);
      await fetchAgents();
    } catch (err) { console.error(err); }
  };

  const getRiskColor = (score) => {
    if (score > 0.8) return '#C0392B';
    if (score > 0.4) return '#B45309';
    return '#1B8A4E';
  };

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Agents</h1>
          <p>Manage and monitor your AI agents</p>
        </div>

        <div className="page-actions">
          <div></div>
          <button onClick={() => setShowModal(true)} className="btn btn-primary">
            <Plus size={16} />
            New Agent
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Provider</th>
                <th>Model</th>
                <th>Status</th>
                <th>Risk Score</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {agents.map((agent) => (
                <tr key={agent.id}>
                  <td style={{ fontWeight: 600 }}>{agent.name}</td>
                  <td>{agent.provider}</td>
                  <td>{agent.model}</td>
                  <td>
                    <span className={`badge ${agent.status === 'active' ? 'badge-success' : 'badge-muted'}`}>
                      {agent.status}
                    </span>
                  </td>
                  <td>
                    <div className="risk-score">
                      <div className="risk-bar">
                        <div className="risk-bar-fill" style={{ width: `${(agent.risk_score || 0) * 100}%`, backgroundColor: getRiskColor(agent.risk_score || 0) }}></div>
                      </div>
                      <span>{((agent.risk_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button onClick={() => handleToggleStatus(agent.id, agent.status)} className="btn btn-secondary btn-sm">
                        {agent.status === 'active' ? <Pause size={14} /> : <Play size={14} />}
                      </button>
                      <button onClick={() => handleDelete(agent.id)} className="btn btn-danger btn-sm">
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {showModal && (
          <div className="modal-overlay" onClick={() => setShowModal(false)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h2>Create New Agent</h2>
              </div>
              <form onSubmit={handleSubmit}>
                <div className="modal-body">
                  <div className="form-group">
                    <label className="form-label">Name</label>
                    <input
                      className="form-input"
                      value={form.name}
                      onChange={(e) => setForm({ ...form, name: e.target.value })}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Provider</label>
                    <select
                      className="form-select"
                      value={form.provider}
                      onChange={(e) => setForm({ ...form, provider: e.target.value })}
                    >
                      <option value="openai">OpenAI</option>
                      <option value="anthropic">Anthropic</option>
                      <option value="gemini">Google Gemini</option>
                      <option value="ollama">Ollama</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Model</label>
                    <input
                      className="form-input"
                      value={form.model}
                      onChange={(e) => setForm({ ...form, model: e.target.value })}
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Description</label>
                    <textarea
                      className="form-textarea"
                      value={form.description}
                      onChange={(e) => setForm({ ...form, description: e.target.value })}
                      rows={3}
                    />
                  </div>
                </div>
                <div className="modal-footer">
                  <button type="button" onClick={() => setShowModal(false)} className="btn btn-secondary">Cancel</button>
                  <button type="submit" className="btn btn-primary">Create</button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Agents;
