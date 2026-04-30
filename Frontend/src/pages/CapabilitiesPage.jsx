/**
 * Name: Jayesh Pandey
 * Summary: Source file for CapabilitiesPage.jsx in the pages module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { SERVICES } from '../constants/data';

const CapabilitiesPage = () => {
  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          style={{ textAlign: 'center', marginBottom: '5rem' }}
        >
          <h1 className="text-display" style={{ fontSize: '5rem', marginBottom: '1.5rem' }}>Capabilities</h1>
          <p className="text-body" style={{ maxWidth: '700px', margin: '0 auto' }}>
            VLMJudge provides a robust suite of tools designed for deep multimodal analysis and preference learning.
          </p>
        </motion.div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '2rem' }}>
          {SERVICES.map((service, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="glass-panel-light"
              style={{
                padding: '3rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '1.5rem',
                backgroundColor: service.dark ? 'var(--color-bg-dark)' : '#fff',
                color: service.dark ? '#fff' : 'inherit',
                border: '1px solid rgba(0,0,0,0.05)',
                borderRadius: '2rem'
              }}
            >
              <span style={{ 
                fontSize: '0.8rem', 
                textTransform: 'uppercase', 
                letterSpacing: '0.1em',
                opacity: 0.7
              }}>
                {service.tag}
              </span>
              <h3 style={{ fontSize: '2rem', fontWeight: 600 }}>{service.title}</h3>
              <p style={{ opacity: service.dark ? 0.8 : 1, lineHeight: 1.6 }}>{service.desc}</p>
              {service.image && (
                <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
                   <img src={service.image} alt={service.title} style={{ width: '100%', borderRadius: '1rem', height: '200px', objectFit: 'cover' }} />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default CapabilitiesPage;
