import React, { useEffect, useState } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Plus, Trash2 } from 'lucide-react';
import Header from '../components/Header';

const PoliciesAndRules = () => {
  const { policies, rules, fetchPolicies, fetchRules } = useStore();
  const [activeTab, setActiveTab] = useState('rules');
  const [showModal, setShowModal] = useState(false);
  const [form, setForm] = useState({
    name: '', description: '', conditions: '', action: 'warn', enabled: true, severity: 'medium', type: 'content'
  });

  useEffect(() => {
    fetchPolicies();
    fetchRules();
  }, [fetchPolicies, fetchRules]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (activeTab === 'rules') {
        await api.post('/rules', form);
        await fetchRules();
      } else {
        await api.post('/policies', {
          ...form,
          rules: []
        });
        await fetchPolicies();
      }
      setShowModal(false);
      setForm({
        name: '', description: '', conditions: '', action: 'warn', enabled: true, severity: 'medium', type: 'content'
      });
    } catch (err) { console.error(err); }
  };

  const handleToggle = async (id, current, type) => {
    try {
      if (type === 'rules') {
        await api.put(`/rules/${id}`, { enabled: !current });
        await fetchRules();
      } else {
        await api.put(`/policies/${id}`, { enabled: !current });
        await fetchPolicies();
      }
    } catch (err) { console.error(err); }
  };

  const handleDelete = async (id, type) => {
    if (!window.confirm(`Delete this ${type === 'rules' ? 'rule' : 'policy'}?`)) return;
    try {
      if (type === 'rules') {
        await api.delete(`/rules/${id}`);
        await fetchRules();
      } else {
        await api.delete(`/policies/${id}`);
        await fetchPolicies();
      }
    } catch (err) { console.error(err); }
  };

  const data = activeTab === 'rules' ? rules : policies;

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Policies & Rules</h1>
          <p>Define governance policies and rules</p>
        </div>

        <div style={{ marginBottom: 24, display: 'flex', gap: 8 }}>
          <button
            onClick={() => setActiveTab('rules')}
            className={`btn ${activeTab === 'rules' ? 'btn-primary' : 'btn-secondary'}`}
          >
            Rules
          </button>
          <button
            onClick={() => setActiveTab('policies')}
            className={`btn ${activeTab === 'policies' ? 'btn-primary' : 'btn-secondary'}`}
          >
            Policies
          </button>
        </div>

        <div className="page-actions">
          <div></div>
          <button onClick={() => setShowModal(true)} className="btn btn-primary">
            <Plus size={16} />
            New {activeTab === 'rules' ? 'Rule' : 'Policy'}
          </button>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Description</th>
                {activeTab === 'rules' && <th>Action</th>}
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item) => (
                <tr key={item.id}>
                  <td style={{ fontWeight: 600 }}>{item.name}</td>
                  <td>{item.description}</td>
                  {activeTab === 'rules' && (
                    <td>
                      <span className="badge badge-info">{item.action}</span>
                    </td>
                  )}
                  <td>
                    <span className={`badge ${item.enabled ? 'badge-success' : 'badge-muted'}`}>
                      {item.enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button onClick={() => handleToggle(item.id, item.enabled, activeTab)} className="btn btn-secondary btn-sm">
                        {item.enabled ? 'Disable' : 'Enable'}
                      </button>
                      <button onClick={() => handleDelete(item.id, activeTab)} className="btn btn-danger btn-sm">
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
                <h2>Create New {activeTab === 'rules' ? 'Rule' : 'Policy'}</h2>
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
                    <label className="form-label">Description</label>
                    <textarea
                      className="form-textarea"
                      value={form.description}
                      onChange={(e) => setForm({ ...form, description: e.target.value })}
                      rows={3}
                    />
                  </div>
                  {activeTab === 'rules' && (
                    <>
                      <div className="form-group">
                        <label className="form-label">Type</label>
                        <select
                          className="form-select"
                          value={form.type}
                          onChange={(e) => setForm({ ...form, type: e.target.value })}
                        >
                          <option value="content">Content</option>
                          <option value="rate">Rate</option>
                          <option value="context">Context</option>
                          <option value="custom">Custom</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Conditions</label>
                        <textarea
                          className="form-textarea"
                          value={form.conditions}
                          onChange={(e) => setForm({ ...form, conditions: e.target.value })}
                          rows={3}
                          placeholder="e.g., prompt contains 'secret'"
                        />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Action</label>
                        <select
                          className="form-select"
                          value={form.action}
                          onChange={(e) => setForm({ ...form, action: e.target.value })}
                        >
                          <option value="warn">Warn</option>
                          <option value="block">Block</option>
                          <option value="require_approval">Require Approval</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Severity</label>
                        <select
                          className="form-select"
                          value={form.severity}
                          onChange={(e) => setForm({ ...form, severity: e.target.value })}
                        >
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                        </select>
                      </div>
                    </>
                  )}
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

export default PoliciesAndRules;
