import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { Bell, X } from 'lucide-react';
import Header from '../components/Header';

const Notifications = () => {
  const { notifications, fetchNotifications } = useStore();

  useEffect(() => fetchNotifications(), [fetchNotifications]);

  const getNotificationIconColor = (type) => {
    switch (type) {
      case 'anomaly_detected': return '#C0392B';
      case 'action_blocked': return '#C0392B';
      case 'approval_required': return '#B45309';
      case 'agent_offline': return '#9AA5B4';
      case 'export_ready': return '#1B8A4E';
      case 'integration_failed': return '#C0392B';
      default: return '#5F6B7A';
    }
  };

  const formatTimeAgo = (dateStr) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'just now';
  };

  const handleMarkRead = async (id) => {
    try {
      await api.patch(`/notifications/${id}/read`);
      await fetchNotifications();
    } catch (error) {
      console.error('Error marking read:', error);
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await api.patch('/notifications/read-all');
      await fetchNotifications();
    } catch (error) {
      console.error('Error marking all read:', error);
    }
  };

  return (
    <div className="main-container">
      <Header />
      <div className="page-content">
        <div className="page-header">
          <h1>Notifications</h1>
          <p>All system notifications</p>
        </div>

        <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'flex-end' }}>
          <button onClick={handleMarkAllRead} className="btn btn-secondary">
            Mark all read
          </button>
        </div>

        <div className="card" style={{ padding: 0 }}>
          {notifications.length > 0 ? (
            notifications.map(notif => (
              <div key={notif.id} className={`notification-item ${!notif.read ? 'unread' : ''}`} style={{ borderBottom: '1px solid var(--color-border)', cursor: 'default' }}>
                <div className="notification-icon">
                  <Bell size={20} color={getNotificationIconColor(notif.type)} />
                </div>
                <div className="notification-content">
                  <p className="notification-title">{notif.title}</p>
                  <p className="notification-message">{notif.message}</p>
                  <p className="notification-time">{formatTimeAgo(notif.created_at)}</p>
                </div>
                {!notif.read && (
                  <button
                    onClick={() => handleMarkRead(notif.id)}
                    className="notification-close"
                  >
                    <X size={18} />
                  </button>
                )}
              </div>
            ))
          ) : (
            <div style={{ padding: 48, textAlign: 'center', color: '#9AA5B4' }}>
              No notifications
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Notifications;
