import React, { useEffect } from 'react';
import useStore from '../store/useStore';
import api from '../api/client';
import { AlertTriangle, Shield, Clock, Download, Link2, Power, Bell } from 'lucide-react';

const NotificationsPage = () => {
  const { notifications, fetchNotifications } = useStore();

  useEffect(() => {
    fetchNotifications();
  }, [fetchNotifications]);

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'anomaly_detected':
        return <AlertTriangle className="text-obstra-danger" size={20} />;
      case 'action_blocked':
        return <Shield className="text-obstra-danger" size={20} />;
      case 'approval_required':
        return <Clock className="text-obstra-warning" size={20} />;
      case 'agent_offline':
        return <Power className="text-obstra-text-muted" size={20} />;
      case 'export_ready':
        return <Download className="text-obstra-success" size={20} />;
      case 'integration_failed':
        return <Link2 className="text-obstra-danger" size={20} />;
      default:
        return <Bell size={20} />;
    }
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
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-obstra-text mb-2">Notifications</h1>
          <p className="text-obstra-text-secondary">All your notifications</p>
        </div>
        <button
          onClick={handleMarkAllRead}
          className="px-4 py-2 bg-obstra-secondary text-white rounded-obstra hover:bg-obstra-secondary-hover transition-colors"
        >
          Mark All Read
        </button>
      </div>

      <div className="bg-white rounded-obstra shadow-card border border-obstra-border overflow-hidden">
        {notifications.length > 0 ? (
          <div className="divide-y divide-obstra-border">
            {notifications.map(notif => (
              <div key={notif.id} className={`p-6 flex items-start gap-4 ${!notif.read ? 'bg-obstra-table-header/30' : ''}`}>
                <div className="mt-1">{getNotificationIcon(notif.type)}</div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-semibold text-obstra-text">{notif.title}</h3>
                    <span className="text-xs text-obstra-text-muted">
                      {new Date(notif.created_at).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-obstra-text-secondary">{notif.message}</p>
                </div>
                {!notif.read && (
                  <button
                    onClick={() => handleMarkRead(notif.id)}
                    className="p-2 text-obstra-text-muted hover:text-obstra-secondary rounded-obstra"
                  >
                    Mark read
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="p-12 text-center">
            <p className="text-obstra-text-secondary">No notifications yet</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NotificationsPage;
