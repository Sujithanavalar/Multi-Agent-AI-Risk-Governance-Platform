import React, { useEffect, useState } from 'react';
import { Activity, TrendingUp, Target, Calendar, Users, Download, FileText } from 'lucide-react';
import api from '../api/client';
import { 
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  Tooltip
} from 'recharts';

const RiskAnalysisPage = () => {
  const [logs, setLogs] = useState([]);
  const [heatmapData, setHeatmapData] = useState([]);
  const [trendData, setTrendData] = useState([]);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const logsRes = await api.get('/audit/logs', { params: { limit: 1000 } });
      const logData = logsRes.data;
      setLogs(logData);

      // Prepare heatmap data
      const agentRiskMap = {};
      logData.forEach(log => {
        const agentId = log.agent_id;
        const hour = new Date(log.timestamp).getHours();
        
        if (!agentRiskMap[agentId]) {
          agentRiskMap[agentId] = {};
        }
        
        if (!agentRiskMap[agentId][hour]) {
          agentRiskMap[agentId][hour] = { count: 0, totalRisk: 0 };
        }
        
        agentRiskMap[agentId][hour].count++;
        agentRiskMap[agentId][hour].totalRisk += log.final_risk_score || 0;
      });

      const heatData = [];
      Object.keys(agentRiskMap).forEach(agentId => {
        Object.keys(agentRiskMap[agentId]).forEach(hour => {
          const avgRisk = agentRiskMap[agentId][hour].totalRisk / agentRiskMap[agentId][hour].count;
          heatData.push({
            agent: parseInt(agentId),
            hour: parseInt(hour),
            risk: avgRisk
          });
        });
      });
      setHeatmapData(heatData);

      // Prepare trend data
      const trendMap = {};
      logData.forEach(log => {
        const date = new Date(log.timestamp).toLocaleDateString();
        if (!trendMap[date]) {
          trendMap[date] = { date, avgRisk: 0, count: 0 };
        }
        trendMap[date].count++;
        trendMap[date].avgRisk += log.final_risk_score || 0;
      });

      const trendArr = Object.values(trendMap).map(d => ({
        date: d.date,
        avgRisk: parseFloat((d.avgRisk / d.count).toFixed(2))
      })).slice(-10);
      setTrendData(trendArr);

    } catch (err) {
      console.error(err);
    }
  };

  const getRiskColor = (risk) => {
    if (risk > 0.7) return '#DC3545'; // Red
    if (risk > 0.3) return '#F5A623'; // Orange
    return '#198754'; // Green
  };

  const getRiskBg = (risk) => {
    if (risk > 0.7) return 'rgba(220, 53, 69, 0.15)';
    if (risk > 0.3) return 'rgba(245, 166, 35, 0.15)';
    return 'rgba(25, 135, 84, 0.15)';
  };

  const exportToCSV = () => {
    const headers = ['ID', 'Agent ID', 'Action Type', 'Final Risk Score', 'Status', 'Timestamp'];
    const rows = logs.map(log => [
      log.id,
      log.agent_id,
      log.action_type,
      log.final_risk_score?.toFixed(2) || '0.00',
      log.status,
      new Date(log.timestamp).toLocaleString()
    ]);
    
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `risk_analysis_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportToPDF = () => {
    alert('PDF export is available via backend integration. For now, please use CSV export or print the page (Ctrl+P / Cmd+P).');
  };

  const agents = [...new Set(heatmapData.map(d => d.agent))].sort((a, b) => a - b);
  const hours = [0, 4, 8, 12, 16, 20];

  return (
    <div>
      <header style={{ marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Risk Analysis</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Comprehensive risk analysis and visualization</p>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button 
            onClick={exportToCSV} 
            className="btn"
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <Download size={16} /> Export to CSV
          </button>
          <button 
            onClick={exportToPDF} 
            className="btn"
            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
          >
            <FileText size={16} /> Export to PDF
          </button>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginBottom: '24px' }}>
        <div className="glass-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ margin: 0, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Calendar size={18} /> Risk Heatmap
            </h3>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Agent vs. Hour of Day
            </span>
          </div>

          {heatmapData.length === 0 ? (
            <div style={{ 
              height: '320px', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              color: 'var(--text-secondary)' 
            }}>
              <Activity size={24} style={{ marginRight: '8px' }} />
              Run the simulation to see risk data
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '60px repeat(24, 1fr)', gap: '2px', minWidth: '800px' }}>
                {/* Header - Hours */}
                <div style={{ padding: '8px', fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}></div>
                {Array.from({ length: 24 }, (_, i) => (
                  <div key={i} style={{ 
                    padding: '8px', 
                    fontSize: '0.75rem', 
                    color: 'var(--text-muted)', 
                    textAlign: 'center',
                    fontWeight: 500
                  }}>
                    {i}:00
                  </div>
                ))}

                {/* Rows - Agents */}
                {agents.map(agentId => (
                  <React.Fragment key={agentId}>
                    <div style={{ 
                      padding: '8px', 
                      fontSize: '0.8rem', 
                      color: 'var(--text-secondary)', 
                      textAlign: 'right',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end'
                    }}>
                      Agent {agentId}
                    </div>
                    {Array.from({ length: 24 }, (_, hour) => {
                      const data = heatmapData.find(d => d.agent === agentId && d.hour === hour);
                      return (
                        <div 
                          key={hour}
                          title={data ? `Risk: ${data.risk.toFixed(2)}` : 'No data'}
                          style={{ 
                            height: '32px', 
                            background: data ? getRiskBg(data.risk) : '#FAFAFA',
                            borderRadius: '4px',
                            border: data ? `2px solid ${getRiskColor(data.risk)}` : '1px solid var(--border-subtle)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            cursor: data ? 'pointer' : 'default',
                            transition: 'all 0.2s'
                          }}
                        >
                          {data && (
                            <span style={{ 
                              fontSize: '0.7rem', 
                              color: getRiskColor(data.risk),
                              fontWeight: 700
                            }}>
                              {data.risk.toFixed(1)}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: '24px', marginTop: '20px', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', background: '#198754', borderRadius: '2px' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Low Risk (&lt;0.3)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', background: '#F5A623', borderRadius: '2px' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Medium Risk (0.3-0.7)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', background: '#DC3545', borderRadius: '2px' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>High Risk (&gt;0.7)</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="glass-card">
            <h3 style={{ marginBottom: '12px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Target size={18} /> Risk Distribution
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Low Risk (&lt;0.3)</span>
                <span style={{ fontWeight: 600, color: 'var(--success)' }}>
                  {logs.filter(l => l.final_risk_score < 0.3).length}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Medium Risk (0.3-0.7)</span>
                <span style={{ fontWeight: 600, color: 'var(--warning)' }}>
                  {logs.filter(l => l.final_risk_score >= 0.3 && l.final_risk_score <= 0.7).length}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>High Risk (&gt;0.7)</span>
                <span style={{ fontWeight: 600, color: 'var(--danger)' }}>
                  {logs.filter(l => l.final_risk_score > 0.7).length}
                </span>
              </div>
            </div>
          </div>

          <div className="glass-card">
            <h3 style={{ marginBottom: '12px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={18} /> Trend Overview
            </h3>
            {trendData.length === 0 ? (
              <div style={{ height: '120px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                No trend data yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Area 
                    type="monotone" 
                    dataKey="avgRisk" 
                    stroke="#DC3545" 
                    fill="rgba(220,53,69,0.1)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <div className="glass-card">
        <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Top Risky Actions</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          {[
            { action: 'DELETE_TABLE', count: logs.filter(l => l.action_type === 'DELETE_TABLE').length, risk: 0.95 },
            { action: 'TRANSFER_FUNDS', count: logs.filter(l => l.action_type === 'TRANSFER_FUNDS').length, risk: 0.82 },
            { action: 'MODIFY_PERMISSIONS', count: logs.filter(l => l.action_type === 'MODIFY_PERMISSIONS').length, risk: 0.78 },
            { action: 'BATCH_EXPORT', count: logs.filter(l => l.action_type === 'BATCH_EXPORT').length, risk: 0.65 },
          ].map((item, idx) => (
            <div key={idx} style={{ padding: '16px', background: '#FAFAFA', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
              <p style={{ fontWeight: 600, marginBottom: '8px', color: 'var(--text-primary)' }}>{item.action}</p>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Count: {item.count}</span>
                <span style={{ fontWeight: 600, color: item.risk > 0.8 ? 'var(--danger)' : 'var(--warning)' }}>
                  Risk: {item.risk}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RiskAnalysisPage;
