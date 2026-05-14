import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, Bot, Activity, AlertTriangle,
  Shield, FileText, Users, Zap, Play, Cog, User
} from 'lucide-react';
import logo from '../assets/hero.png';

const Sidebar = () => {
  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Command Center' },
    { path: '/agent-mesh', icon: Users, label: 'Agent Mesh' },
    { path: '/live-monitoring', icon: Activity, label: 'Live Monitoring' },
    { path: '/risk-analysis', icon: Zap, label: 'Risk Analysis' },
    { path: '/approvals', icon: Shield, label: 'Consensus Queue' },
    { path: '/threats', icon: AlertTriangle, label: 'Threats' },
    { path: '/audit-trail', icon: FileText, label: 'Audit Trail' },
    { path: '/policies', icon: Shield, label: 'Policies' },
    { path: '/integrations', icon: Users, label: 'Integrations' },
    { path: '/simulation', icon: Play, label: 'Simulation' },
    { path: '/admin', icon: Cog, label: 'Admin' },
  ];

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <img src={logo} alt="OBSTRA" style={{ width: 32, height: 32, objectFit: 'contain' }} />
        <h1>OBSTRA</h1>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
        {navItems.map(item => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `sidebar-nav-item ${isActive ? 'active' : ''}`
            }
          >
            <item.icon size={18} />
            {item.label}
          </NavLink>
        ))}
      </div>
      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12, marginTop: 8 }}>
        <NavLink to="/profile" className="sidebar-nav-item">
          <User size={18} />
          Profile
        </NavLink>
      </div>
    </div>
  );
};

export default Sidebar;
