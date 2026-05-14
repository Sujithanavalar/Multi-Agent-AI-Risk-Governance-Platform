import React from 'react';
import Header from '../components/Header';

const Audit = () => {
  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Audit Trail</h1>
          <p>Complete audit history of all events</p>
        </div>
        <div className="card">
          <p style={{ color: '#9AA5B4', textAlign: 'center', padding: '48px' }}>
            Audit trail data will be displayed here
          </p>
        </div>
      </div>
    </div>
  );
};

export default Audit;
