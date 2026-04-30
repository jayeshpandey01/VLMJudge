/**
 * Name: Jayesh Pandey
 * Summary: React component for the global navigation bar, handling authentication state, navigation links, and scroll-responsive UI effects.
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import { NAV_LINKS } from '../constants/data';
import { useAuth } from '../context/AuthContext';
import AuthModal from './AuthModal';
import { User, LogOut } from 'lucide-react';

const Navbar = () => {
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const { user, logout } = useAuth();
  const location = useLocation();

  return (
    <>
      <motion.nav 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '1.5rem 0',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 100,
          backgroundColor: 'rgba(245, 245, 245, 0.8)',
          backdropFilter: 'blur(10px)',
          borderBottom: '1px solid rgba(0,0,0,0.05)'
        }}
      >
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 800, fontSize: '1.5rem' }}>
            <div style={{ width: '24px', height: '24px', backgroundColor: 'var(--color-bg-dark)', borderRadius: '4px', position: 'relative' }}>
                <div style={{position: 'absolute', top: 4, left: 4, width: 8, height: 8, backgroundColor: 'var(--color-accent)', borderRadius: '2px'}}></div>
            </div>
            VLMJudge
          </Link>

          <ul style={{ display: 'flex', gap: '2rem', listStyle: 'none', margin: 0, padding: 0, fontWeight: 500, fontSize: '0.9rem' }}>
            {NAV_LINKS.map((link) => (
              <li key={link.name}>
                <Link 
                  to={link.href} 
                  className="hover-scale" 
                  style={{
                    display: 'inline-block',
                    color: location.pathname === link.href ? 'var(--color-accent)' : 'inherit',
                    fontWeight: location.pathname === link.href ? 700 : 500
                  }}
                >
                  {link.name}
                </Link>
              </li>
            ))}
          </ul>

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            {user ? (
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: '#666' }}>
                  <User size={18} />
                  {user.email.split('@')[0]}
                </div>
                <button onClick={logout} className="btn btn-outline" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <LogOut size={16} />
                  Logout
                </button>
              </div>
            ) : (
              <button 
                onClick={() => setIsAuthModalOpen(true)}
                className="btn btn-primary" 
                style={{ borderRadius: '2rem', padding: '0.5rem 1.5rem' }}
              >
                Login
              </button>
            )}
          </div>
        </div>
      </motion.nav>

      <AuthModal 
        isOpen={isAuthModalOpen} 
        onClose={() => setIsAuthModalOpen(false)} 
      />
    </>
  );
};

export default Navbar;
