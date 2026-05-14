import React, { useEffect, useState, useRef } from 'react';
import { Network, Users, Activity, Eye } from 'lucide-react';
import axios from 'axios';
import * as d3 from 'd3';

const AgentMeshPage = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentActions, setAgentActions] = useState([]);
  const svgRef = useRef(null);

  useEffect(() => {
    fetchAgents();
  }, []);

  useEffect(() => {
    if (agents.length > 0 && svgRef.current) {
      renderGraph();
    }
  }, [agents]);

  const fetchAgents = async () => {
    try {
      const agentsRes = await axios.get('http://localhost:8001/agents/');
      setAgents(agentsRes.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAgentClick = async (agent) => {
    setSelectedAgent(agent);
    try {
      const logsRes = await axios.get('http://localhost:8000/audit/logs');
      const agentLogs = logsRes.data.filter(log => log.agent_id === agent.id).slice(0, 10);
      setAgentActions(agentLogs);
    } catch (err) {
      console.error(err);
    }
  };

  const renderGraph = () => {
    if (!svgRef.current) return;
    const width = svgRef.current.clientWidth || 600;
    const height = 500;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    if (agents.length === 0) {
      // No agents - show placeholder
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', 'var(--text-muted)')
        .attr('font-size', '1.1rem')
        .text('No agents registered yet');
      return;
    }

    // Create links between agents (simple connections)
    const links = [];
    for (let i = 0; i < agents.length - 1; i++) {
      links.push({ source: agents[i].id, target: agents[i + 1].id });
    }

    const nodes = agents.map(agent => ({
      id: agent.id,
      name: agent.name,
      framework: agent.framework,
      ...agent
    }));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide().radius(50));

    // Draw links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', 'var(--border-subtle)')
      .attr('stroke-width', 2);

    // Draw nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('cursor', 'pointer')
      .on('click', (event, d) => handleAgentClick(d));

    // Node circles
    node.append('circle')
      .attr('r', 35)
      .attr('fill', 'var(--accent-cyan)')
      .attr('stroke', d => selectedAgent?.id === d.id ? 'var(--primary-red)' : 'var(--border-subtle)')
      .attr('stroke-width', d => selectedAgent?.id === d.id ? 3 : 2);

    // Node labels (initials)
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 8)
      .attr('fill', 'white')
      .attr('font-weight', 700)
      .attr('font-size', '1.25rem')
      .text(d => d.name.charAt(0));

    // Node name below
    node.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 60)
      .attr('fill', 'var(--text-primary)')
      .attr('font-size', '0.9rem')
      .attr('font-weight', 500)
      .text(d => d.name.length > 12 ? d.name.substring(0, 12) + '...' : d.name);

    // Update positions
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });
  };

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Agent Mesh</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Live network graph of your AI agents</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px' }}>
        {/* Agents Graph */}
        <div className="glass-card" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Network size={20} /> Live Network
            </h3>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{agents.length} agents connected</span>
          </div>
          
          <div style={{ background: 'white', borderRadius: '12px', overflow: 'hidden' }}>
            <svg ref={svgRef} style={{ width: '100%', display: 'block' }}></svg>
          </div>
          
          <div style={{ marginTop: '20px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent-cyan)' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Active Agent</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '12px', height: '2px', background: 'var(--border-subtle)' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Connection</span>
            </div>
          </div>
        </div>

        {/* Agent Details */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="glass-card">
            <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Eye size={20} /> Agent Details
            </h3>
            
            {selectedAgent ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingBottom: '16px', borderBottom: '1px solid var(--border-subtle)' }}>
                  <div style={{ 
                    width: '60px', height: '60px', 
                    borderRadius: '50%', 
                    background: 'var(--accent-cyan)', 
                    color: 'white', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    fontWeight: 700,
                    fontSize: '1.5rem'
                  }}>
                    {selectedAgent.name.charAt(0)}
                  </div>
                  <div>
                    <p style={{ margin: 0, fontWeight: 700, color: 'var(--text-primary)', fontSize: '1.1rem' }}>{selectedAgent.name}</p>
                    <p style={{ margin: 0, color: 'var(--text-secondary)' }}>{selectedAgent.framework}</p>
                  </div>
                </div>
                
                <div>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Agent ID</p>
                  <p style={{ margin: 0, color: 'var(--text-primary)', fontFamily: 'monospace' }}>#{selectedAgent.id}</p>
                </div>
                
                <div>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Owner</p>
                  <p style={{ margin: 0, color: 'var(--text-primary)' }}>{selectedAgent.owner || 'Unknown'}</p>
                </div>
                
                <div>
                  <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>Permissions</p>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {selectedAgent.permissions?.map((perm, idx) => (
                      <span key={idx} style={{ 
                        padding: '4px 10px', 
                        background: '#E6F7F5', 
                        color: 'var(--accent-cyan)', 
                        borderRadius: '12px', 
                        fontSize: '0.75rem', 
                        fontWeight: 600 
                      }}>
                        {perm}
                      </span>
                    )) || <span style={{ color: 'var(--text-muted)' }}>No permissions set</span>}
                  </div>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--success)' }}></div>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Online & Active</span>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '48px 16px' }}>
                <Users size={48} color="var(--text-muted)" />
                <p style={{ color: 'var(--text-secondary)', marginTop: '16px' }}>
                  Click an agent node to view details
                </p>
              </div>
            )}
          </div>

          {selectedAgent && (
            <div className="glass-card">
              <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Activity size={20} /> Recent Actions
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '300px', overflowY: 'auto' }}>
                {agentActions.length === 0 ? (
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', textAlign: 'center', padding: '16px' }}>No recent actions for this agent</p>
                ) : (
                  agentActions.map((action) => (
                    <div key={action.id} style={{ 
                      padding: '12px', 
                      background: '#FAFAFA', 
                      borderRadius: '8px',
                      borderLeft: `3px solid ${action.status === 'BLOCKED' ? 'var(--danger)' : 
                                        action.status === 'PENDING' ? 'var(--warning)' : 'var(--success)'}`
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{action.action_type}</span>
                        <span className={`badge badge-${action.status.toLowerCase()}`}>{action.status}</span>
                      </div>
                      <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        {new Date(action.timestamp).toLocaleString()}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AgentMeshPage;
