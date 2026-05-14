import React, { useState, useEffect } from 'react';
import { Plug, Bot, Globe, Server, Activity, Plus, Save, X, Send, MessageSquare, Eye, EyeOff, CheckCircle, XCircle } from 'lucide-react';
import api from '../api/client';

const PROVIDERS = [
  { id: 'openai',    label: 'OpenAI (GPT)',       color: '#10A37F' },
  { id: 'claude',    label: 'Anthropic (Claude)', color: '#D4712F' },
  { id: 'gemini',    label: 'Google Gemini',      color: '#4285F4' },
  { id: 'ollama',    label: 'Ollama (Local)',      color: '#7C3AED' },
  { id: 'openrouter',label: 'OpenRouter',          color: '#6366F1' },
  { id: 'custom',    label: 'Custom API',          color: '#7A8BAA' },
];

function providerColor(p) {
  return PROVIDERS.find(x => x.id === p)?.color || '#7A8BAA';
}
function providerLabel(p) {
  return PROVIDERS.find(x => x.id === p)?.label || (p || 'Custom');
}

function AgentCard({ agent, onEdit, onSelect, selected }) {
  const color = providerColor(agent.provider);
  return (
    <div
      style={{
        background: 'var(--bg-card)',
        border: `1px solid ${selected ? color + '88' : 'var(--border)'}`,
        borderRadius: 12,
        padding: 20,
        cursor: 'pointer',
        transition: 'all 0.2s',
        borderTop: `3px solid ${color}`,
      }}
      onClick={() => onSelect(agent)}
      onMouseEnter={e => e.currentTarget.style.borderColor = color + '88'}
      onMouseLeave={e => { if (!selected) e.currentTarget.style.borderColor = 'var(--border)'; }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 38, height: 38, borderRadius: 8, background: color + '22', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Bot size={18} color={color} />
          </div>
          <div>
            <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.95rem' }}>{agent.name}</div>
            <div style={{ fontSize: '0.75rem', color, marginTop: 2, fontWeight: 600 }}>{providerLabel(agent.provider)}</div>
          </div>
        </div>
        <span className={`badge ${agent.is_active ? 'badge-allowed' : 'badge-low'}`} style={{ fontSize: '0.65rem' }}>
          {agent.is_active ? 'Active' : 'Inactive'}
        </span>
      </div>
      <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
        {agent.description || agent.system_prompt?.substring(0, 80) || 'No description'}
      </p>
      <button className="btn" style={{ width: '100%', fontSize: '0.8rem', gap: 6 }} onClick={e => { e.stopPropagation(); onEdit(agent); }}>
        Configure
      </button>
    </div>
  );
}

function ChatPanel({ agent, onClose }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);

  async function send() {
    if (!input.trim()) return;
    const userMsg = { role: 'user', content: input };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setSending(true);
    try {
      const res = await api.post(`/integrations/${agent.id}/send`, { prompt: input });
      const reply = res.data?.response || res.data?.output || 'Response received';
      const riskScore = res.data?.risk_score ?? null;
      const blocked = res.data?.blocked || false;
      setMessages(m => [...m, {
        role: 'assistant', content: reply,
        risk_score: riskScore, blocked,
      }]);
    } catch (e) {
      const detail = e?.response?.data?.detail || 'Request failed or blocked by OBSTRA';
      setMessages(m => [...m, { role: 'error', content: detail }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(8,12,20,0.85)', backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: 24 }}
      onClick={onClose}
    >
      <div className="modal-box" style={{ maxWidth: 640, maxHeight: '80vh', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, flexShrink: 0 }}>
          <div>
            <h3 style={{ fontFamily: 'Space Grotesk, sans-serif', margin: 0 }}>Chat — {agent.name}</h3>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 3 }}>Every message is intercepted and scored by OBSTRA before reaching {providerLabel(agent.provider)}</div>
          </div>
          <button className="btn btn-ghost" style={{ padding: 8 }} onClick={onClose}><X size={18} /></button>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 16, minHeight: 200, maxHeight: 400 }}>
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
              <MessageSquare size={32} style={{ marginBottom: 10, opacity: 0.3 }} />
              <div>Send a prompt to test OBSTRA governance</div>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%' }}>
              <div style={{
                padding: '10px 14px',
                borderRadius: m.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                background: m.role === 'user' ? 'rgba(0,180,255,0.15)' : m.role === 'error' ? 'var(--danger-glow)' : 'var(--bg-surface)',
                border: `1px solid ${m.role === 'user' ? 'rgba(0,180,255,0.3)' : m.role === 'error' ? 'rgba(255,59,92,0.3)' : 'var(--border)'}`,
                color: m.role === 'error' ? 'var(--danger)' : 'var(--text-primary)',
                fontSize: '0.875rem',
                lineHeight: 1.6,
              }}>
                {m.content}
              </div>
              {m.risk_score != null && (
                <div style={{ fontSize: '0.72rem', color: m.blocked ? 'var(--danger)' : 'var(--text-muted)', marginTop: 4, textAlign: m.role === 'user' ? 'right' : 'left' }}>
                  {m.blocked ? '🚫 Blocked by OBSTRA' : `✓ Passed · Risk: ${m.risk_score.toFixed(2)}`}
                </div>
              )}
            </div>
          ))}
          {sending && (
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', color: 'var(--text-muted)', fontSize: '0.82rem', paddingLeft: 4 }}>
              <div className="spinner" style={{ width: 14, height: 14 }} /> OBSTRA scoring…
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 10, flexShrink: 0 }}>
          <input
            placeholder="Send a prompt through OBSTRA…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
            disabled={sending}
          />
          <button className="btn btn-primary" onClick={send} disabled={sending || !input.trim()} style={{ flexShrink: 0, gap: 6 }}>
            <Send size={15} /> Send
          </button>
        </div>
      </div>
    </div>
  );
}

function AddAgentModal({ editing, onSave, onClose }) {
  const initial = editing || { name: '', framework: 'external', description: '', api_key: '', provider: 'openai', system_prompt: '', permissions: [] };
  const [form, setForm] = useState(initial);
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setSaving(true);
    try { await onSave(form); onClose(); }
    catch (err) { alert(err?.response?.data?.detail || 'Failed'); }
    finally { setSaving(false); }
  }

  const needsBaseUrl = ['ollama', 'custom'].includes(form.provider);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()} style={{ maxWidth: 580 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <h3 style={{ fontFamily: 'Space Grotesk, sans-serif' }}>{editing ? 'Configure Agent' : 'Connect External Agent'}</h3>
          <button className="btn btn-ghost" style={{ padding: 8 }} onClick={onClose}><X size={18} /></button>
        </div>
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="grid-2">
            <div><label>Agent Name</label><input value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="My GPT Agent" required /></div>
            <div>
              <label>Provider</label>
              <select value={form.provider} onChange={e => setForm(f => ({ ...f, provider: e.target.value }))}>
                {PROVIDERS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
              </select>
            </div>
          </div>
          <div>
            <label>Description</label>
            <input value={form.description} onChange={e => setForm(f => ({ ...f, description: e.target.value }))} placeholder="What does this agent do?" />
          </div>
          <div>
            <label>API Key</label>
            <div style={{ position: 'relative' }}>
              <input type={showKey ? 'text' : 'password'} value={form.api_key} onChange={e => setForm(f => ({ ...f, api_key: e.target.value }))} placeholder={editing?.has_api_key ? 'Leave blank to keep existing key' : 'sk-...'} style={{ paddingRight: 40 }} />
              <button type="button" onClick={() => setShowKey(s => !s)} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          {needsBaseUrl && (
            <div><label>Base URL</label><input value={form.base_url || ''} onChange={e => setForm(f => ({ ...f, base_url: e.target.value }))} placeholder="http://localhost:11434" /></div>
          )}
          <div><label>System Prompt</label><textarea rows={3} value={form.system_prompt} onChange={e => setForm(f => ({ ...f, system_prompt: e.target.value }))} placeholder="You are a helpful AI assistant..." /></div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 8 }}>
            <button type="button" className="btn" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={saving}><Save size={16} /> {saving ? 'Saving…' : 'Save'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function IntegrationsPage() {
  const [agents, setAgents] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [editingAgent, setEditingAgent] = useState(null);
  const [chatAgent, setChatAgent] = useState(null);
  const [loading, setLoading] = useState(true);

  async function fetchAgents() {
    try { const res = await api.get('/agents/'); setAgents(res.data); }
    catch { } finally { setLoading(false); }
  }

  useEffect(() => { fetchAgents(); }, []);

  async function handleSave(form) {
    if (editingAgent) {
      const { id, token, registered_at, has_api_key, ...rest } = form;
      await api.put(`/agents/${editingAgent.id}`, rest);
    } else {
      await api.post('/agents/register-extended', form);
    }
    await fetchAgents();
  }

  const externalAgents = agents.filter(a => a.provider && a.provider !== 'custom' && a.provider !== null);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1><Plug size={22} style={{ verticalAlign: 'middle', marginRight: 10, color: 'var(--primary)' }} />Integrations</h1>
          <p>Connect external AI agents — every prompt and response is governed by OBSTRA</p>
        </div>
        <button className="btn btn-primary" onClick={() => { setEditingAgent(null); setShowAddModal(true); }} style={{ gap: 6 }}>
          <Plus size={16} /> Connect Agent
        </button>
      </div>

      {/* Provider grid */}
      <div className="glass-card" style={{ marginBottom: 28 }}>
        <h3 className="section-title"><Globe size={16} color="var(--secondary)" /> Supported Providers</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 10 }}>
          {PROVIDERS.map(p => (
            <div key={p.id} style={{ padding: '10px 14px', background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: p.color, flexShrink: 0 }} />
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>{p.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Connected agents */}
      <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 className="section-title" style={{ margin: 0 }}>Connected Agents ({externalAgents.length})</h3>
        {externalAgents.length > 0 && <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Click a card to chat through OBSTRA</span>}
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 60 }}><div className="spinner" style={{ margin: '0 auto' }} /></div>
      ) : externalAgents.length === 0 ? (
        <div className="glass-card" style={{ textAlign: 'center', padding: 60 }}>
          <Plug size={40} style={{ marginBottom: 16, opacity: 0.3 }} />
          <h3 style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>No agents connected</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: 20 }}>Connect an external AI agent to govern its prompts and responses</p>
          <button className="btn btn-primary" onClick={() => setShowAddModal(true)} style={{ gap: 6 }}>
            <Plus size={16} /> Connect First Agent
          </button>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 18 }}>
          {externalAgents.map(agent => (
            <AgentCard
              key={agent.id}
              agent={agent}
              selected={chatAgent?.id === agent.id}
              onEdit={a => { setEditingAgent(a); setShowAddModal(true); }}
              onSelect={a => setChatAgent(chatAgent?.id === a.id ? null : a)}
            />
          ))}
        </div>
      )}

      {showAddModal && (
        <AddAgentModal
          editing={editingAgent}
          onSave={handleSave}
          onClose={() => { setShowAddModal(false); setEditingAgent(null); }}
        />
      )}

      {chatAgent && (
        <ChatPanel agent={chatAgent} onClose={() => setChatAgent(null)} />
      )}
    </div>
  );
}
