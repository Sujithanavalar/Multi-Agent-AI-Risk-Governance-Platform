import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Network, Users, Activity, Eye, Pause, Play, Zap, Bot } from 'lucide-react';
import api from '../api/client';
import * as d3 from 'd3';

const AgentMeshPage = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentActions, setAgentActions] = useState([]);
  const svgRef = useRef(null);

  const fetchAgents = useCallback(async () => {
    try {
      const agentsRes = await api.get('/agents/');
      setAgents(agentsRes.data);
    } catch (err) { console.error(err); }
  }, []);

  useEffect(() => { fetchAgents(); }, [fetchAgents]);

  const handleAgentClick = async (agent) => {
    setSelectedAgent(agent);
    try {
      const res = await api.get(`/agents/${agent.id}/actions`);
      setAgentActions(res.data);
    } catch { setAgentActions([]); }
  };

  const handleDeactivate = async (agentId) => {
    if (!confirm('Deactivate this agent?')) return;
    try { await api.put(`/agents/${agentId}/deactivate`); await fetchAgents(); setSelectedAgent(null); setAgentActions([]); }
    catch { alert('Failed'); }
  };

  useEffect(() => {
    if (!svgRef.current) return;
    const width = svgRef.current.clientWidth || 700;
    const height = 480;
    d3.select(svgRef.current).selectAll('*').remove();
    const svg = d3.select(svgRef.current).attr('width', width).attr('height', height);

    if (agents.length === 0) {
      svg.append('text').attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle').attr('fill', '#3D4F6B').attr('font-size', '1rem')
        .text('No agents registered yet');
      return;
    }

    const links = agents.length > 1
      ? agents.slice(0, -1).map((a, i) => ({ source: a.id, target: agents[i + 1].id }))
      : [];

    const nodes = agents.map(a => ({ ...a }));

    // Glow filter
    const defs = svg.append('defs');
    const filter = defs.append('filter').attr('id', 'glow');
    filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(140))
      .force('charge', d3.forceManyBody().strength(-500))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(55));

    const link = svg.append('g').selectAll('line').data(links).enter().append('line')
      .attr('stroke', '#1E2D4A').attr('stroke-width', 2).attr('stroke-dasharray', '6,4');

    const node = svg.append('g').selectAll('g').data(nodes).enter().append('g')
      .attr('cursor', 'pointer').on('click', (_, d) => handleAgentClick(d));

    node.append('circle').attr('r', 38)
      .attr('fill', d => d.is_active ? 'rgba(0,180,255,0.12)' : 'rgba(61,79,107,0.2)')
      .attr('stroke', d => selectedAgent?.id === d.id ? '#00B4FF' : d.is_active ? 'rgba(0,180,255,0.5)' : '#3D4F6B')
      .attr('stroke-width', d => selectedAgent?.id === d.id ? 3 : 2)
      .attr('filter', d => d.is_active ? 'url(#glow)' : null);

    node.append('text').attr('text-anchor', 'middle').attr('dy', 6)
      .attr('fill', d => d.is_active ? '#00B4FF' : '#3D4F6B').attr('font-weight', 700).attr('font-size', '1.1rem')
      .text(d => d.name.charAt(0).toUpperCase());

    node.append('text').attr('text-anchor', 'middle').attr('dy', 62)
      .attr('fill', '#7A8BAA').attr('font-size', '0.8rem').attr('font-weight', 500)
      .text(d => d.name.length > 12 ? d.name.substring(0, 12) + '…' : d.name);

    node.append('circle').attr('r', 6).attr('cx', 26).attr('cy', -26)
      .attr('fill', d => d.is_active ? '#00E676' : '#3D4F6B');

    simulation.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
  }, [agents, selectedAgent]);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1><Network size={22} style={{ verticalAlign: 'middle', marginRight: 10, color: 'var(--primary)' }} />Agent Mesh</h1>
          <p>Interactive network of all registered agents — click a node to inspect</p>
        </div>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{agents.length} agents registered</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 24 }}>
        <div className="glass-card" style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h3 className="section-title" style={{ margin: 0 }}><Network size={16} color="var(--primary)" /> Network Graph</h3>
            <div style={{ display: 'flex', gap: 14, fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#00E676', display: 'inline-block' }} /> Active
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#3D4F6B', display: 'inline-block' }} /> Inactive
              </span>
            </div>
          </div>
          <div style={{ background: 'rgba(8,12,20,0.6)', borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border)' }}>
            <svg ref={svgRef} style={{ width: '100%', display: 'block' }} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <div className="glass-card">
            <h3 className="section-title"><Eye size={16} color="var(--secondary)" /> Agent Details</h3>
            {selectedAgent ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, paddingBottom: 14, borderBottom: '1px solid var(--border)' }}>
                  <div style={{ width: 48, height: 48, borderRadius: '50%', background: 'rgba(0,180,255,0.15)', border: '2px solid rgba(0,180,255,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)', fontWeight: 700, fontSize: '1.2rem' }}>
                    {selectedAgent.name.charAt(0)}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '1rem' }}>{selectedAgent.name}</div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 2 }}>{selectedAgent.framework || selectedAgent.type || 'internal'}</div>
                  </div>
                </div>

                {[
                  { label: 'Provider', value: selectedAgent.provider },
                  { label: 'Owner', value: selectedAgent.owner },
                  { label: 'Status', value: selectedAgent.is_active ? '● Active' : '● Inactive' },
                ].filter(f => f.value).map(f => (
                  <div key={f.label}>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 3 }}>{f.label}</div>
                    <div style={{ fontSize: '0.875rem', color: f.label === 'Status' ? (selectedAgent.is_active ? 'var(--safe)' : 'var(--text-muted)') : 'var(--text-primary)' }}>{f.value}</div>
                  </div>
                ))}

                {selectedAgent.permissions?.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Permissions</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {selectedAgent.permissions.map((p, i) => (
                        <span key={i} style={{ padding: '3px 10px', background: 'rgba(0,255,200,0.1)', color: 'var(--secondary)', borderRadius: 100, fontSize: '0.72rem', border: '1px solid rgba(0,255,200,0.3)' }}>{p}</span>
                      ))}
                    </div>
                  </div>
                )}

                {selectedAgent.is_active && (
                  <button className="btn btn-danger" onClick={() => handleDeactivate(selectedAgent.id)} style={{ marginTop: 4 }}>
                    Deactivate Agent
                  </button>
                )}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 12px', color: 'var(--text-muted)' }}>
                <Network size={36} style={{ marginBottom: 10, opacity: 0.3 }} />
                <p style={{ fontSize: '0.85rem' }}>Click a node to inspect</p>
              </div>
            )}
          </div>

          {selectedAgent && (
            <div className="glass-card" style={{ maxHeight: 320, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
              <h3 className="section-title" style={{ flexShrink: 0 }}><Activity size={16} color="var(--warning)" /> Recent Actions</h3>
              <div style={{ overflowY: 'auto', flex: 1 }}>
                {agentActions.length === 0 ? (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No actions logged yet</p>
                ) : agentActions.map(action => {
                  const score = action.final_risk_score ?? 0;
                  const statusCls = String(action.status).toLowerCase() === 'blocked' ? 'badge-blocked' : String(action.status).toLowerCase() === 'pending' ? 'badge-pending' : 'badge-allowed';
                  return (
                    <div key={action.id} style={{ padding: '10px 12px', background: 'var(--bg-surface)', borderRadius: 8, marginBottom: 8, borderLeft: `3px solid ${score >= 0.7 ? 'var(--danger)' : score >= 0.3 ? 'var(--warning)' : 'var(--safe)'}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)' }}>{action.action_type}</span>
                        <span className={`badge ${statusCls}`} style={{ fontSize: '0.65rem' }}>{action.status}</span>
                      </div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 4 }}>
                        Score: {score.toFixed(2)} · {action.timestamp ? new Date(action.timestamp).toLocaleTimeString() : '—'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentMeshPage;
