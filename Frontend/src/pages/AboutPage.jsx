/**
 * Name: Jayesh Pandey
 * Summary: Source file for AboutPage.jsx in the pages module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { ABOUT_CONTENT } from '../constants/data';

const AboutPage = () => {
  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <span style={{ 
            textTransform: 'uppercase', 
            letterSpacing: '0.2em', 
            fontSize: '0.85rem', 
            fontWeight: 600, 
            color: 'var(--color-accent)',
            display: 'block',
            marginBottom: '1rem'
          }}>
            {ABOUT_CONTENT.tag}
          </span>
          <h1 className="text-h1" style={{ marginBottom: '3rem', maxWidth: '800px' }}>
            {ABOUT_CONTENT.title}
          </h1>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '4rem' }}>
            <div>
              <p className="text-body" style={{ fontSize: '1.25rem', lineHeight: 1.6, color: '#333' }}>
                {ABOUT_CONTENT.description}
              </p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="glass-panel-light" style={{ padding: '2rem' }}>
                <h3 style={{ marginBottom: '1rem' }}>Our Mission</h3>
                <p style={{ color: '#666' }}>To bridge the gap between automated metrics and human perception in multimodal AI systems through rigorous, reasoning-based evaluation.</p>
              </div>
              <div className="glass-panel-light" style={{ padding: '2rem' }}>
                <h3 style={{ marginBottom: '1rem' }}>The Framework</h3>
                <p style={{ color: '#666' }}>VLMJudge leverages an ensemble of state-of-the-art vision-language models to provide not just scores, but explainable reasoning for every decision.</p>
              </div>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default AboutPage;
