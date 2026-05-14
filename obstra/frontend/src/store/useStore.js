import { create } from 'zustand';
import api from '../api/client';

const useStore = create((set, get) => ({
  user: JSON.parse(localStorage.getItem('user') || 'null'),
  accessToken: localStorage.getItem('access_token'),
  refreshToken: localStorage.getItem('refresh_token'),
  
  setUser: (user) => {
    set({ user });
    localStorage.setItem('user', JSON.stringify(user));
  },
  
  setTokens: (access, refresh) => {
    set({ accessToken: access, refreshToken: refresh });
    localStorage.setItem('access_token', access);
    if (refresh) localStorage.setItem('refresh_token', refresh);
  },
  
  logout: () => {
    set({ user: null, accessToken: null, refreshToken: null });
    localStorage.removeItem('user');
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  },
  
  agents: [],
  fetchAgents: async () => {
    try {
      const response = await api.get('/agents');
      set({ agents: response.data.agents });
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  },
  
  rules: [],
  fetchRules: async () => {
    try {
      const response = await api.get('/rules');
      set({ rules: response.data });
    } catch (error) {
      console.error('Error fetching rules:', error);
    }
  },
  
  approvals: [],
  fetchApprovals: async () => {
    try {
      const response = await api.get('/approvals');
      set({ approvals: response.data });
    } catch (error) {
      console.error('Error fetching approvals:', error);
    }
  },
  
  anomalies: [],
  fetchAnomalies: async () => {
    try {
      const response = await api.get('/anomalies');
      set({ anomalies: response.data });
    } catch (error) {
      console.error('Error fetching anomalies:', error);
    }
  },
  
  notifications: [],
  fetchNotifications: async () => {
    try {
      const response = await api.get('/notifications');
      set({ notifications: response.data });
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  }
}));

export default useStore;
