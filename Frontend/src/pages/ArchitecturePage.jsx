/**
 * Name: Jayesh Pandey
 * Summary: Source file for ArchitecturePage.jsx in the pages module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { BRANDING_CONTENT } from '../constants/data';

const ArchitecturePage = () => {
  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh', backgroundColor: '#fff' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-display" style={{ fontSize: '4rem', marginBottom: '4rem' }}>System Architecture</h1>
          
          <div className="glass-panel" style={{ 
            backgroundColor: 'var(--color-bg-dark)', 
            color: '#fff', 
            padding: '5rem', 
            borderRadius: '3rem',
            marginBottom: '5rem',
            position: 'relative',
            overflow: 'hidden'
          }}>
             <div style={{ position: 'relative', zIndex: 2 }}>
                <h2 style={{ fontSize: '3rem', marginBottom: '2rem' }}>{BRANDING_CONTENT.bannerTitle}</h2>
                <p style={{ maxWidth: '600px', fontSize: '1.25rem', opacity: 0.8, lineHeight: 1.6 }}>{BRANDING_CONTENT.bannerDesc}</p>
             </div>
             <img 
               src={BRANDING_CONTENT.bannerImage} 
               style={{ 
                 position: 'absolute', 
                 top: 0, 
                 right: 0, 
                 width: '50%', 
                 height: '100%', 
                 objectFit: 'cover', 
                 opacity: 0.4,
                 maskImage: 'linear-gradient(to left, black, transparent)'
               }} 
             />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '3rem' }}>
             {BRANDING_CONTENT.categories.map((cat, i) => (
               <div key={i} style={{ padding: '2rem', borderBottom: '1px solid #eee' }}>
                 <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>{cat}</h3>
                 <p style={{ color: '#666' }}>Comprehensive modular implementation for {cat.toLowerCase()}, optimized for low latency and high accuracy.</p>
               </div>
             ))}
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default ArchitecturePage;
