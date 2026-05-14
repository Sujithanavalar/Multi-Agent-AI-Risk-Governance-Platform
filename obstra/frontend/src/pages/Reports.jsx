import React from 'react';
import Header from '../components/Header';

const Reports = () => {
  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Reports</h1>
          <p>Generate compliance and governance reports</p>
        </div>
        <div className="card">
          <p style={{ color: '#9AA5B4', textAlign: 'center', padding: '48px' }}>
            Report generation and downloads will be available here
          </p>
        </div>
      </div>
    </div>
  );
};

export default Reports;
