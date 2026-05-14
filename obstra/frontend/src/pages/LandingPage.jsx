import React, { useState } from 'react';
import { Eye, Shield, Activity, Users, ArrowRight, FileText, Play, Lock, Globe, Mail, Key } from 'lucide-react';
import logo from '../assets/hero.png';
import { Link, useNavigate } from 'react-router-dom';
import { supabase } from '../supabase';

const LandingPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [orgId, setOrgId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [termsAccepted, setTermsAccepted] = useState(false);
  const navigate = useNavigate();

  const handleAuth = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (!isLogin && !termsAccepted) {
        setError('You must accept the Terms and Conditions');
        setLoading(false);
        return;
      }

      if (isLogin) {
        const { error: signInError } = await supabase.auth.signInWithPassword({
          email,
          password
        });
        if (signInError) throw signInError;
      } else {
        const { error: signUpError } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              organization_id: orgId
            }
          }
        });
        if (signUpError) throw signUpError;
      }
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #F9F9F9 0%, #F2F0ED 100%',
      padding: '0',
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      alignItems: 'stretch'
    }}>
      {/* Left Side: Hero */}
      <div style={{ 
        padding: '48px 64px', 
        display: 'flex', 
        flexDirection: 'column', 
        background: 'linear-gradient(135deg, #07111F 0%, #0E1B2E 100%)',
        color: 'white'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '64px' }}>
          <img src={logo} alt="Obstra Logo" style={{ width: '48px', height: '48px', objectFit: 'contain' }} />
          <h1 style={{ margin: 0, fontSize: '1.75rem', fontFamily: 'var(--font-display)', letterSpacing: '2px' }}>OBSTRA</h1>
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h2 style={{ 
            fontSize: '3rem', 
            marginBottom: '24px', 
            lineHeight: '1.2', 
            fontFamily: 'var(--font-display)'
          }}>
            Observe • Analyze • Detect • Protect
          </h2>
          <p style={{ fontSize: '1.1rem', lineHeight: '1.7', color: 'rgba(234, 246, 255, 0.85)', marginBottom: '32px' }}>
            Real-Time AI Governance &amp; Anomaly Detection Platform. Secure and monitor autonomous AI agents using anomaly detection, contextual risk analysis, consensus validation, and tamper-proof audit chains.
          </p>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', marginTop: '32px' }}>
            {[
              { icon: Activity, title: 'Real-Time Monitoring', desc: 'Live agent activity tracking' },
              { icon: Shield, title: 'Isolation Forest Detection', desc: 'ML-powered anomaly detection' },
              { icon: Lock, title: 'Hash-Chained Audit Logs', desc: 'Tamper-proof blockchain-style' }
            ].map((feature, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                <div style={{ 
                  width: '40px', height: '40px', borderRadius: '8px', 
                  background: 'rgba(0, 183, 255, 0.1)', 
                  display: 'flex', alignItems: 'center', justifyContent: 'center', 
                  marginTop: '4px'
                }}>
                  <feature.icon size={20} color="#00B7FF" />
                </div>
                <div>
                  <h4 style={{ margin: 0, fontSize: '1.1rem', marginBottom: '4px' }}>{feature.title}</h4>
                  <p style={{ margin: 0, fontSize: '0.9rem', color: 'rgba(234, 246, 255, 0.7)' }}>{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 'auto', color: 'rgba(234, 246, 255, 0.5)', fontSize: '0.85rem' }}>
          © 2025 Obstra. Securing Autonomous Intelligence.
        </div>
      </div>

      {/* Right Side: Login/Signup */}
      <div style={{ 
        padding: '48px 64px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center'
      }}>
        <div className="glass-card" style={{ 
          width: '100%', 
          maxWidth: '420px', 
          padding: '40px',
          borderTop: '4px solid var(--primary-red)'
        }}>
          <div style={{ marginBottom: '32px', textAlign: 'center' }}>
            <h2 style={{ fontSize: '1.75rem', marginBottom: '8px', color: 'var(--text-primary)' }}>
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h2>
            <p style={{ color: 'var(--text-secondary)' }}>
              {isLogin ? 'Sign in to your Obstra dashboard' : 'Get started with Obstra'}
            </p>
          </div>

          {error && (
            <div style={{ 
              padding: '12px', 
              background: 'rgba(220, 53, 69, 0.1)', 
              border: '1px solid var(--danger)', 
              color: 'var(--danger)',
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              {error}
            </div>
          )}

          <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                Email Address
              </label>
              <div style={{ position: 'relative' }}>
                <Mail size={18} color="var(--text-muted)" style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)' }} />
                <input 
                  type="email" 
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px 12px 44px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-primary)',
                    background: 'white'
                  }}
                />
              </div>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                Password
              </label>
              <div style={{ position: 'relative' }}>
                <Key size={18} color="var(--text-muted)" style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)' }} />
                <input 
                  type="password" 
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  style={{ 
                    width: '100%', 
                    padding: '12px 16px 12px 44px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-subtle)', 
                    fontSize: '1rem',
                    color: 'var(--text-primary)',
                    background: 'white'
                  }}
                />
              </div>
            </div>

            {!isLogin && (
              <>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)', fontSize: '0.9rem', fontWeight: 500 }}>
                    Organization ID
                  </label>
                  <input 
                    type="text" 
                    value={orgId}
                    onChange={(e) => setOrgId(e.target.value)}
                    placeholder="ORG-XXXX"
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
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px' }}>
                  <input 
                    type="checkbox" 
                    id="terms" 
                    checked={termsAccepted}
                    onChange={(e) => setTermsAccepted(e.target.checked)}
                    style={{ marginTop: '4px' }}
                  />
                  <label htmlFor="terms" style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                    I accept the <a href="#" style={{ color: 'var(--primary-red)', textDecoration: 'none' }}>Terms and Conditions</a> and <a href="#" style={{ color: 'var(--primary-red)', textDecoration: 'none' }}>Privacy Policy</a>
                  </label>
                </div>
              </>
            )}

            <button 
              type="submit" 
              disabled={loading}
              className="btn btn-primary"
              style={{ marginTop: '8px', fontSize: '1rem', padding: '14px' }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ width: '16px', height: '16px', border: '2px solid white', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></span>
                  {isLogin ? 'Signing in...' : 'Creating account...'}
                </span>
              ) : (
                <span>{isLogin ? 'Sign In' : 'Sign Up'}</span>
              )}
            </button>
          </form>

          <div style={{ marginTop: '28px', textAlign: 'center' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '0.95rem' }}>
              {isLogin ? "Don't have an account?" : 'Already have an account?'}
              <button 
                onClick={() => { setIsLogin(!isLogin); setError(''); }}
                style={{ 
                  background: 'none', 
                  border: 'none', 
                  color: 'var(--primary-red)', 
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  marginLeft: '6px'
                }}
              >
                {isLogin ? 'Create one' : 'Sign in'}
              </button>
            </p>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '20px', marginBottom: '20px' }}>
              <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }}></div>
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>OR</span>
              <div style={{ flex: 1, height: '1px', background: 'var(--border-subtle)' }}></div>
            </div>

            <button 
              onClick={() => navigate('/dashboard')}
              className="btn"
              style={{ width: '100%', background: 'white', border: '1px solid var(--border-highlight)' }}
            >
              <Play size={16} /> Demo Access
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default LandingPage;
