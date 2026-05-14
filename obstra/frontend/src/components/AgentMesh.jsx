import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const AgentMesh = ({ logs }) => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 400 });
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.clientWidth,
        height: 400
      });
    }
  }, []);

  useEffect(() => {
    // Transform logs into nodes and links
    const nodesMap = new Map();
    const links = [];

    // Core Governance Node
    nodesMap.set('Obstra', { id: 'Obstra', group: 'governance', val: 5 });

    logs.forEach((log) => {
      const agentId = `Agent_${log.agent_id}`;
      const actionNode = `Action_${log.id}`;
      
      if (!nodesMap.has(agentId)) {
        nodesMap.set(agentId, { id: agentId, group: 'agent', val: 3 });
      }
      
      nodesMap.set(actionNode, { 
        id: actionNode, 
        group: log.status.toLowerCase(), 
        val: 1.5,
        name: log.action_type
      });

      links.push({ source: agentId, target: actionNode, value: 1 });
      links.push({ 
        source: actionNode, 
        target: 'Obstra', 
        value: 1, 
        color: log.status === 'BLOCKED' ? '#DC3545' : (log.status === 'PENDING' ? '#F5A623' : '#E0E0E0') 
      });
    });

    setGraphData({
      nodes: Array.from(nodesMap.values()),
      links: links
    });
  }, [logs]);

  const getNodeColor = (node) => {
    switch(node.group) {
      case 'governance': return '#C72027'; // Oracle Red
      case 'agent': return '#0A8276'; // Oracle Cyan
      case 'approved': return '#198754';
      case 'blocked': return '#DC3545';
      case 'pending': return '#F5A623';
      default: return '#8C8C8C';
    }
  };

  return (
    <div className="glass-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--accent-cyan)' }}></div>
        Live Multi-Agent Mesh
      </h3>
      <div ref={containerRef} style={{ flex: 1, background: '#FAFAFA', border: '1px solid var(--border-subtle)', borderRadius: '4px', overflow: 'hidden' }}>
        <ForceGraph2D
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          nodeColor={getNodeColor}
          nodeRelSize={4}
          linkColor={(link) => link.color || '#E0E0E0'}
          linkWidth={1.5}
          linkDirectionalParticles={2}
          linkDirectionalParticleSpeed={0.005}
          backgroundColor="transparent"
        />
      </div>
    </div>
  );
};

export default AgentMesh;
