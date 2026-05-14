import React from 'react';
import { BarChart3, TrendingUp, Download, FileText } from 'lucide-react';

const ReportsAnalyticsPage = () => {
  const metrics = [
    { label: 'Weekly Anomaly Count', value: '48', change: '+12%' },
    { label: 'Detection Precision', value: '94.2%', change: '+2.1%' },
    { label: 'Consensus Response Time', value: '2.4 min', change: '-18%' },
    { label: 'Audit Integrity Score', value: '100%', change: '0%' },
  ];

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>Reports &amp; Analytics</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Comprehensive reports and analytics</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '24px', marginBottom: '32px' }}>
        {metrics.map((metric, idx) => (
          <div key={idx} className="glass-card">
            <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
              {metric.label}
            </p>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <p style={{ margin: 0, fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-display)' }}>
                {metric.value}
              </p>
              <span style={{ 
                color: metric.change.startsWith('+') ? 'var(--danger)' : 'var(--success)', 
                fontSize: '0.9rem', fontWeight: 600 
              }}>
                {metric.change}
              </span>
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px', marginBottom: '32px' }}>
        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={20} /> Risk Trend Over Time
          </h3>
          <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            Trend chart will appear here
          </div>
        </div>

        <div className="glass-card">
          <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={20} /> Top Risky Workflows
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              { name: 'Financial Transfers', count: 124, risk: 0.78 },
              { name: 'Database Operations', count: 89, risk: 0.72 },
              { name: 'File Access', count: 256, risk: 0.58 },
              { name: 'Prescriptions', count: 67, risk: 0.65 },
            ].map((item, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-primary)' }}>{item.name}</span>
                <span style={{ fontWeight: 600, color: item.risk > 0.7 ? 'var(--danger)' : 'var(--warning)' }}>
                  {item.risk}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="glass-card">
        <h3 style={{ marginBottom: '16px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FileText size={20} /> Export Options
        </h3>
        <div style={{ display: 'flex', gap: '16px' }}>
          <button className="btn" style={{ background: 'white', border: '1px solid var(--border-highlight)' }}>
            <Download size={16} /> Export PDF
          </button>
          <button className="btn" style={{ background: 'white', border: '1px solid var(--border-highlight)' }}>
            <Download size={16} /> Export CSV
          </button>
          <button className="btn btn-primary">
            <FileText size={16} /> Generate Compliance Report
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReportsAnalyticsPage;
