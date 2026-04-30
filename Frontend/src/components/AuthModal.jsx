/**
 * Name: Jayesh Pandey
 * Summary: Source file for AuthModal.jsx in the components module.
 */

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Mail, Lock, User } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const AuthModal = ({ isOpen, onClose }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, password);
      }
      onClose();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
          padding: '2rem'
        }}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.8)',
              backdropFilter: 'blur(10px)',
              zIndex: -1
            }}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            style={{
              width: '100%',
              maxWidth: '450px',
              backgroundColor: '#fff',
              borderRadius: '2.5rem',
              padding: '2.5rem',
              position: 'relative',
              boxShadow: '0 30px 60px rgba(0,0,0,0.3)',
              overflowY: 'auto',
              maxHeight: 'calc(100vh - 4rem)'
            }}
          >
            <button
              onClick={onClose}
              style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', color: '#666', padding: '0.5rem', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              className="hover-scale"
            >
              <X size={20} />
            </button>

            <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
              <div style={{ width: '48px', height: '48px', backgroundColor: 'var(--color-bg-dark)', borderRadius: '12px', position: 'relative', margin: '0 auto 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div style={{ width: 14, height: 14, backgroundColor: 'var(--color-accent)', borderRadius: '3px' }}></div>
              </div>
              <h2 className="text-h3" style={{ fontSize: '1.75rem', fontWeight: 700 }}>{isLogin ? 'Welcome back' : 'Join VLMJudge'}</h2>
              <p style={{ color: '#666', marginTop: '0.5rem', fontSize: '1rem' }}>{isLogin ? 'Login to your account' : 'Create a new account'}</p>
            </div>

            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div style={{ position: 'relative' }}>
                <div style={{ position: 'absolute', left: '1.25rem', top: '50%', transform: 'translateY(-50%)', color: '#999', display: 'flex', alignItems: 'center' }}>
                  <Mail size={18} />
                </div>
                <input
                  type="email"
                  placeholder="Email address"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  style={{ 
                    width: '100%', 
                    padding: '1rem 1.25rem 1rem 3.5rem', 
                    borderRadius: '1.25rem', 
                    border: '1px solid #eee', 
                    backgroundColor: '#f9f9f9', 
                    fontFamily: 'inherit',
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.3s ease'
                  }}
                  onFocus={(e) => e.target.style.borderColor = 'var(--color-accent)'}
                  onBlur={(e) => e.target.style.borderColor = '#eee'}
                />
              </div>

              <div style={{ position: 'relative' }}>
                <div style={{ position: 'absolute', left: '1.25rem', top: '50%', transform: 'translateY(-50%)', color: '#999', display: 'flex', alignItems: 'center' }}>
                  <Lock size={18} />
                </div>
                <input
                  type="password"
                  placeholder="Password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  style={{ 
                    width: '100%', 
                    padding: '1rem 1.25rem 1rem 3.5rem', 
                    borderRadius: '1.25rem', 
                    border: '1px solid #eee', 
                    backgroundColor: '#f9f9f9', 
                    fontFamily: 'inherit',
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.3s ease'
                  }}
                  onFocus={(e) => e.target.style.borderColor = 'var(--color-accent)'}
                  onBlur={(e) => e.target.style.borderColor = '#eee'}
                />
              </div>

              {error && <p style={{ color: '#ff4d4d', fontSize: '0.85rem', textAlign: 'center', margin: '0' }}>{error}</p>}

              <button className="btn btn-primary" style={{ padding: '1rem', borderRadius: '1.25rem', width: '100%', marginTop: '0.5rem', fontSize: '1rem' }}>
                {isLogin ? 'Login' : 'Register'}
              </button>
            </form>

            <div style={{ textAlign: 'center', marginTop: '2rem' }}>
              <button
                onClick={() => setIsLogin(!isLogin)}
                style={{ color: 'var(--color-accent)', fontWeight: 600, fontSize: '0.95rem' }}
                className="hover-scale"
              >
                {isLogin ? "Don't have an account? Register" : "Already have an account? Login"}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default AuthModal;
