import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import LandingPage from './pages/LandingPage';
import DashboardOverview from './pages/DashboardOverview';
import AgentMeshPage from './pages/AgentMeshPage';
import ThreatsDetectedPage from './pages/ThreatsDetectedPage';
import RiskAnalysisPage from './pages/RiskAnalysisPage';
import ApprovalsPage from './pages/ApprovalsPage';
import AuditTrailPage from './pages/AuditTrailPage';
import PolicyManagementPage from './pages/PolicyManagementPage';
import ReportsAnalyticsPage from './pages/ReportsAnalyticsPage';
import IntegrationsPage from './pages/IntegrationsPage';
import AdminPanelPage from './pages/AdminPanelPage';
import SimulationPage from './pages/SimulationPage';
import ProfilePage from './pages/ProfilePage';
import { supabase } from './supabase';

const App = () => {
  // Toggle this to false to bypass Supabase auth
  const USE_SUPABASE_AUTH = false;
  const [user, setUser] = useState(USE_SUPABASE_AUTH ? null : true);
  const [loading, setLoading] = useState(USE_SUPABASE_AUTH ? true : false);

  useEffect(() => {
    if (USE_SUPABASE_AUTH) {
      const { data: authListener } = supabase.auth.onAuthStateChange((event, session) => {
        setUser(session?.user || null);
        setLoading(false);
      });

      const checkSession = async () => {
        const { data: { session } } = await supabase.auth.getSession();
        setUser(session?.user || null);
        setLoading(false);
      };
      checkSession();

      return () => {
        authListener?.subscription?.unsubscribe();
      };
    }
  }, []);

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        color: 'var(--primary-red)',
        fontSize: '1.2rem'
      }}>
        Loading Obstra...
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={
          user ? <Navigate to="/dashboard" replace /> : <LandingPage />
        } />
        <Route path="*" element={
          user ? (
            <div className="app-container">
              <Sidebar />
              <main className="main-content">
                <Routes>
                  <Route path="/dashboard" element={<DashboardOverview />} />
                  <Route path="/agent-mesh" element={<AgentMeshPage />} />
                  <Route path="/threats" element={<ThreatsDetectedPage />} />
                  <Route path="/risk-analysis" element={<RiskAnalysisPage />} />
                  <Route path="/approvals" element={<ApprovalsPage />} />
                  <Route path="/audit-trail" element={<AuditTrailPage />} />
                  <Route path="/policies" element={<PolicyManagementPage />} />
                  <Route path="/reports" element={<ReportsAnalyticsPage />} />
                  <Route path="/integrations" element={<IntegrationsPage />} />
                  <Route path="/admin" element={<AdminPanelPage />} />
                  <Route path="/simulation" element={<SimulationPage />} />
                  <Route path="/profile" element={<ProfilePage />} />
                  <Route path="*" element={<Navigate to="/dashboard" replace />} />
                </Routes>
              </main>
            </div>
          ) : (
            <Navigate to="/" replace />
          )
        } />
      </Routes>
    </Router>
  );
};

export default App;
