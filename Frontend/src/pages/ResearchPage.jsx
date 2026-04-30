/**
 * Name: Jayesh Pandey
 * Summary: Source file for ResearchPage.jsx in the pages module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { BarChart2, TrendingUp, Target, Database } from 'lucide-react';

const ResearchPage = () => {
  const benchmarks = [
    { name: 'ImageReward-DB', score: '82.4%', delta: '+4.2%', desc: 'State-of-the-art alignment with human preferences.' },
    { name: 'HPS v2', score: '79.1%', delta: '+2.8%', desc: 'Superior performance in stylistic consistency.' },
    { name: 'Pick-a-Pic', score: '85.6%', delta: '+5.1%', desc: 'Highly calibrated scoring for prompt adherence.' },
  ];

  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div style={{ textAlign: 'center', marginBottom: '6rem' }}>
            <h1 className="text-display" style={{ fontSize: '5.5rem', marginBottom: '1.5rem' }}>Research</h1>
            <p className="text-body" style={{ maxWidth: '800px', margin: '0 auto', fontSize: '1.25rem' }}>
              Pushing the boundaries of multimodal evaluation through rigorous benchmarking, confidence calibration, and reasoning-driven insights.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '8rem' }}>
            {benchmarks.map((b, i) => (
              <motion.div
                key={i}
                whileHover={{ y: -10 }}
                className="glass-panel-light"
                style={{ padding: '3rem', borderRadius: '2.5rem', border: '1px solid #eee' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                  <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>{b.name}</span>
                  <TrendingUp size={20} color="var(--color-accent)" />
                </div>
                <div style={{ fontSize: '4rem', fontWeight: 800, marginBottom: '0.5rem', color: 'var(--color-bg-dark)' }}>
                  {b.score}
                </div>
                <div style={{ color: 'var(--color-accent)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                  {b.delta} improvement
                </div>
                <p style={{ color: '#666', fontSize: '1rem', lineHeight: 1.5 }}>{b.desc}</p>
              </motion.div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '4rem', alignItems: 'center' }}>
            <div>
              <h2 className="text-h1" style={{ marginBottom: '2rem' }}>Visualization & Metrics</h2>
              <p className="text-body" style={{ marginBottom: '2rem' }}>
                Our research focuses on the visualization of model disagreements and the calibration of confidence scores across diverse semantic domains.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--color-accent)' }}></div>
                  <span>Confidence Calibration Curves</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#000' }}></div>
                  <span>Semantic Disagreement Heatmaps</span>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                  <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#999' }}></div>
                  <span>Cross-Model Consensus Benchmarks</span>
                </div>
              </div>
            </div>
            <div className="glass-panel" style={{ 
              height: '400px', 
              backgroundColor: '#fff', 
              borderRadius: '3rem', 
              display: 'flex', 
              flexDirection: 'column',
              alignItems: 'center', 
              justifyContent: 'center',
              border: '1px solid #eee',
              padding: '3rem',
              gap: '2rem'
            }}>
               <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1.5rem', height: '200px', width: '100%' }}>
                  {[60, 85, 45, 90, 70, 95].map((h, i) => (
                    <motion.div 
                      key={i}
                      initial={{ height: 0 }}
                      whileInView={{ height: `${h}%` }}
                      transition={{ duration: 1, delay: i * 0.1 }}
                      style={{ 
                        flex: 1, 
                        backgroundColor: i === 3 ? 'var(--color-accent)' : '#000', 
                        borderRadius: '0.5rem 0.5rem 0 0' 
                      }}
                    />
                  ))}
               </div>
               <div style={{ textAlign: 'center', color: '#666', fontSize: '0.9rem' }}>
                 <p>Performance Delta across semantic categories (Calibrated)</p>
               </div>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default ResearchPage;
