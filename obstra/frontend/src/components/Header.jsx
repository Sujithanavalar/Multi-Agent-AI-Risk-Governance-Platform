import React, { useEffect, useState } from 'react';
import { Bell, Search, User, RefreshCw } from 'lucide-react';
import api from '../api/client';
import { Link } from 'react-router-dom';

const Header = () => {
  const [notifications, setNotifications] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);

  const fetchNotifs = async () => {
    try {
      const res = await api.get('/notifications');
      setNotifications(res.data);
    } catch (e) {
      console.error('Failed to fetch notifications:', e);
    }
  };

  useEffect(() => {
    fetchNotifs();
    const t = setInterval(fetchNotifs, 10000);
    return () => clearInterval(t);
  }, []);

  const unread = notifications.filter(n => !n.read).length;

  const markRead = async (id) => {
    try {
      await api.patch(`/notifications/${id}/read`);
      fetchNotifs();
    } catch (e) {
      console.error(e);
    }
  };

  const markAllRead = async () => {
    try {
      await api.patch('/notifications/read-all');
      fetchNotifs();
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <header className="main-header">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ position: 'relative' }}>
          <Search size={16} style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
          <input
            className="input"
            placeholder="Search agents, actions, rules…"
            style={{ paddingLeft: 38, width: 420, background: 'rgba(14, 27, 46, 0.6)' }}
          />
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <div className="live-indicator">
          <div className="live-dot" />
          LIVE
        </div>

        <div style={{ position: 'relative' }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setShowDropdown(!showDropdown)}
          >
            <Bell size={18} />
            {unread > 0 && (
              <span
                style={{
                  position: 'absolute',
                  top: -2,
                  right: -2,
                  background: 'var(--danger)',
                  color: 'white',
                  borderRadius: 999,
                  width: 18,
                  height: 18,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.7rem',
                  fontWeight: 700,
                }}
              >
                {unread}
              </span>
            )}
          </button>

          {showDropdown && (
            <div
              className="glass-card"
              style={{
                position: 'absolute',
                right: 0,
                top: '100%',
                marginTop: 10,
                width: 360,
                maxHeight: 420,
                overflow: 'hidden',
                zIndex: 50,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 18px', borderBottom: '1px solid var(--border)' }}>
                <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>Notifications</h3>
                <button className="btn btn-ghost btn-sm" onClick={markAllRead} style={{ fontSize: '0.8rem' }}>
                  Mark all read
                </button>
              </div>
              <div style={{ maxHeight: 320, overflow: 'auto' }}>
                {notifications.length === 0 ? (
                  <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No notifications yet
                  </div>
                ) : (
                  notifications.slice(0, 8).map(n => (
                    <div
                      key={n.id}
                      onClick={() => !n.read && markRead(n.id)}
                      style={{
                        display: 'flex',
                        gap: 12,
                        padding: '12px 18px',
                        borderBottom: '1px solid var(--border)',
                        cursor: 'pointer',
                        background: n.read ? 'transparent' : 'rgba(0, 183, 255, 0.05)',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '0.88rem', fontWeight: 600, marginBottom: 4 }}>{n.title}</div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 4 }}>{n.message}</div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          {new Date(n.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div style={{ borderTop: '1px solid var(--border)', padding: 12, textAlign: 'center' }}>
                <Link to="/notifications" className="btn btn-ghost btn-sm" style={{ width: '100%' }} onClick={() => setShowDropdown(false)}>
                  View all notifications
                </Link>
              </div>
            </div>
          )}
        </div>

        <button className="btn btn-ghost btn-sm">
          <User size={18} />
        </button>
      </div>
    </header>
  );
};

export default Header;
