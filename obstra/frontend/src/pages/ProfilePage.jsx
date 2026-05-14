import React, { useState, useEffect } from 'react';
import { User, Mail, Phone, Building, Save, Eye, Lock, Globe } from 'lucide-react';
import { supabase } from '../supabase';

const ProfilePage = () => {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [user, setUser] = useState(null);
  const [profile, setProfile] = useState({
    full_name: 'Demo User',
    email: 'demo@obstra.ai',
    phone: '',
    organization: 'Demo Organization',
    role: 'Security Officer',
    bio: ''
  });

  useEffect(() => {
    const USE_SUPABASE_AUTH = false;
    if (USE_SUPABASE_AUTH) {
      fetchUserAndProfile();
    }
  }, []);

  const fetchUserAndProfile = async () => {
    setLoading(true);
    try {
      const { data: { user: authUser } } = await supabase.auth.getUser();
      setUser(authUser);

      if (authUser) {
        setProfile(prev => ({ ...prev, email: authUser.email }));

        // Check if profile exists in Supabase
        const { data, error } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', authUser.id)
          .single();

        if (data) {
          setProfile(prev => ({ ...prev, ...data }));
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    const USE_SUPABASE_AUTH = false;
    
    if (USE_SUPABASE_AUTH) {
      setSaving(true);
      try {
        const { data: { user: authUser } } = await supabase.auth.getUser();
        
        // Check if profile exists
        const { data: existingProfile } = await supabase
          .from('profiles')
          .select('*')
          .eq('id', authUser.id)
          .single();

        if (existingProfile) {
          // Update existing profile
          await supabase
            .from('profiles')
            .update(profile)
            .eq('id', authUser.id);
        } else {
          // Insert new profile
          await supabase
            .from('profiles')
            .insert([
              { 
                id: authUser.id, 
                ...profile,
                created_at: new Date().toISOString()
              }
            ]);
        }

        alert('Profile saved successfully!');
      } catch (err) {
        console.error(err);
        alert('Error saving profile: ' + err.message);
      } finally {
        setSaving(false);
      }
    } else {
      // Demo mode - just save locally
      alert('Profile saved locally (Demo mode)!');
    }
  };

  return (
    <div>
      <header style={{ marginBottom: '32px' }}>
        <h1 style={{ color: 'var(--text-primary)', marginBottom: '8px' }}>User Profile</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Manage your account details and preferences</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
        {/* Profile Overview */}
        <div className="glass-card">
          <div style={{ textAlign: 'center', paddingBottom: '24px', borderBottom: '1px solid var(--border-subtle)' }}>
            <div style={{ 
              width: '100px', height: '100px', 
              borderRadius: '50%', 
              background: 'var(--accent-cyan)', 
              color: 'white', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              fontWeight: 700,
              fontSize: '2.5rem',
              margin: '0 auto 16px'
            }}>
              {profile.full_name?.charAt(0) || user?.email?.charAt(0)?.toUpperCase() || 'U'}
            </div>
            <h3 style={{ margin: '0 0 4px 0', color: 'var(--text-primary)' }}>
              {profile.full_name || 'User'}
            </h3>
            <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              {profile.role || 'User'}
            </p>
          </div>
          
          <div style={{ padding: '20px 0' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Mail size={18} color="var(--text-muted)" />
                <span style={{ color: 'var(--text-secondary)' }}>{profile.email || 'Not set'}</span>
              </div>
              {profile.phone && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Phone size={18} color="var(--text-muted)" />
                  <span style={{ color: 'var(--text-secondary)' }}>{profile.phone}</span>
                </div>
              )}
              {profile.organization && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <Building size={18} color="var(--text-muted)" />
                  <span style={{ color: 'var(--text-secondary)' }}>{profile.organization}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Profile Form */}
        <div className="glass-card">
          <h3 style={{ marginBottom: '24px', color: 'var(--text-primary)' }}>Edit Profile</h3>
          
          <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                  Full Name
                </label>
                <input 
                  type="text" 
                  value={profile.full_name}
                  onChange={(e) => setProfile({ ...profile, full_name: e.target.value })}
                  placeholder="John Doe"
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-primary)',
                    background: 'white'
                  }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                  Email Address
                </label>
                <input 
                  type="email" 
                  value={profile.email}
                  disabled
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-muted)',
                    background: '#FAFAFA'
                  }}
                />
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                  Phone Number
                </label>
                <input 
                  type="tel" 
                  value={profile.phone}
                  onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                  placeholder="+1 (555) 000-0000"
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-primary)',
                    background: 'white'
                  }}
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                  Role
                </label>
                <select 
                  value={profile.role}
                  onChange={(e) => setProfile({ ...profile, role: e.target.value })}
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-primary)',
                    background: 'white'
                  }}
                >
                  <option value="">Select role</option>
                  <option value="Admin">Admin</option>
                  <option value="Security Officer">Security Officer</option>
                  <option value="Developer">Developer</option>
                  <option value="Auditor">Auditor</option>
                  <option value="Viewer">Viewer</option>
                </select>
              </div>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                Organization
              </label>
              <input 
                type="text" 
                value={profile.organization}
                onChange={(e) => setProfile({ ...profile, organization: e.target.value })}
                placeholder="Your Company"
                style={{ 
                  width: '100%', 
                  padding: '12px 16px', 
                  borderRadius: '8px', 
                  border: '1px solid var(--border-subtle)', 
                  fontSize: '1rem',
                  color: 'var(--text-primary)',
                  background: 'white'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                Bio
              </label>
              <textarea 
                value={profile.bio}
                onChange={(e) => setProfile({ ...profile, bio: e.target.value })}
                placeholder="Tell us a little about yourself..."
                rows={4}
                style={{ 
                  width: '100%', 
                  padding: '12px 16px', 
                  borderRadius: '8px', 
                  border: '1px solid var(--border-subtle)', 
                  fontSize: '1rem',
                  color: 'var(--text-primary)',
                  background: 'white',
                  resize: 'vertical'
                }}
              />
            </div>

            <button 
              type="submit" 
              disabled={saving}
              className="btn btn-primary"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              <Save size={18} />
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
