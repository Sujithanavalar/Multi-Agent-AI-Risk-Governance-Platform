import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import LandingPage from './pages/LandingPage';
import DashboardOverview from './pages/DashboardOverview';
import AgentsPage from './pages/AgentsPage';
import AgentMeshPage from './pages/AgentMeshPage';
import ApprovalsPage from './pages/ApprovalsPage';
import AnomaliesPage from './pages/AnomaliesPage';
import PolicyManagementPage from './pages/PolicyManagementPage';
import AuditTrailPage from './pages/AuditTrailPage';
import ReportsAnalyticsPage from './pages/ReportsAnalyticsPage';
import NotificationsPage from './pages/NotificationsPage';
import RiskAnalysisPage from './pages/RiskAnalysisPage';
import SimulationPage from './pages/SimulationPage';
import ThreatsDetectedPage from './pages/ThreatsDetectedPage';
import LiveMonitoringPage from './pages/LiveMonitoringPage';
import IntegrationsPage from './pages/IntegrationsPage';
import AdminPanelPage from './pages/AdminPanelPage';
import ProfilePage from './pages/ProfilePage';
import './index.css';

// Simple auth check for now
const isAuthenticated = () => {
  return localStorage.getItem('access_token') !== null || true; // Allow demo access
};

function App() {
  return (
    <Router>
      <Routes>
        {/* Landing & Auth */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LandingPage />} />

        {/* Protected Routes */}
        <Route
          path="/*"
          element={
            <div className="app-wrapper">
              <Sidebar />
              <div className="main-area">
                <Header />
                <main className="main-content">
                  <Routes>
                    <Route path="/dashboard" element={<DashboardOverview />} />
                    <Route path="/agents" element={<AgentsPage />} />
                    <Route path="/agent-mesh" element={<AgentMeshPage />} />
                    <Route path="/approvals" element={<ApprovalsPage />} />
                    <Route path="/anomalies" element={<AnomaliesPage />} />
                    <Route path="/policies" element={<PolicyManagementPage />} />
                    <Route path="/audit-trail" element={<AuditTrailPage />} />
                    <Route path="/reports" element={<ReportsAnalyticsPage />} />
                    <Route path="/notifications" element={<NotificationsPage />} />
                    <Route path="/risk-analysis" element={<RiskAnalysisPage />} />
                    <Route path="/simulation" element={<SimulationPage />} />
                    <Route path="/threats" element={<ThreatsDetectedPage />} />
                    <Route path="/live-monitoring" element={<LiveMonitoringPage />} />
                    <Route path="/integrations" element={<IntegrationsPage />} />
                    <Route path="/admin" element={<AdminPanelPage />} />
                    <Route path="/profile" element={<ProfilePage />} />
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </main>
              </div>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
