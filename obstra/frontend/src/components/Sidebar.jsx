import React from 'react';
import { 
  LayoutDashboard, Network, AlertTriangle, TrendingUp, 
  CheckSquare, Database, Settings, BarChart3, Puzzle, 
  UserCog, PlayCircle, LogOut, User, UserCircle 
} from 'lucide-react';
import logo from '../assets/hero.png';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { supabase } from '../supabase';

const Sidebar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      navigate('/');
    } catch (err) {
      console.error(err);
    }
  };

  const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/agent-mesh', icon: Network, label: 'Agent Mesh' },
    { path: '/threats', icon: AlertTriangle, label: 'Threats' },
    { path: '/risk-analysis', icon: TrendingUp, label: 'Risk Analysis' },
    { path: '/approvals', icon: CheckSquare, label: 'Approvals' },
    { path: '/audit-trail', icon: Database, label: 'Audit Trail' },
    { path: '/policies', icon: Settings, label: 'Policies' },
    { path: '/reports', icon: BarChart3, label: 'Reports & Analytics' },
    { path: '/integrations', icon: Puzzle, label: 'Integrations' },
    { path: '/admin', icon: UserCog, label: 'Admin' },
    { path: '/simulation', icon: PlayCircle, label: 'Simulation' },
  ];

  return (
    <aside className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '32px', padding: '0 24px' }}>
        <div style={{ width: '40px', height: '40px' }}>
          <img src={logo} alt="Obstra Logo" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        </div>
        <div>
          <h2 style={{ fontSize: '1.2rem', color: 'var(--text-primary)', margin: 0 }}>Obstra</h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Enterprise Governance</span>
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, padding: '0 12px' }}>
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                textDecoration: 'none',
                background: isActive ? '#E6E3DF' : 'transparent',
                borderLeft: isActive ? '4px solid var(--primary-red)' : '4px solid transparent',
                borderRadius: '4px',
                transition: 'all 0.2s',
                fontWeight: isActive ? 500 : 400
              }}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto', padding: '16px 24px', borderTop: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{ width: '36px', height: '36px', borderRadius: '50%', background: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 600 }}>
            <User size={18} />
          </div>
          <div>
            <p style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>User</p>
            <p style={{ margin: 0, fontSize: '0.8rem', color: 'var(--text-muted)' }}>user@obstra.com</p>
          </div>
        </div>

        <button 
          onClick={handleLogout}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            color: 'var(--text-secondary)',
            textDecoration: 'none',
            padding: '8px 0',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            width: '100%',
            fontSize: '0.95rem'
          }}
        >
          <LogOut size={16} />
          <span>Logout</span>
        </button>
        
        <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '12px', marginBottom: '8px' }}>System Status</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--success)' }}></div>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>Risk Engine Active</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
