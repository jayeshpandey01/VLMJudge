/**
 * Name: Jayesh Pandey
 * Summary: Source file for APIPage.jsx in the pages module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Code, Terminal, Cpu, ShieldCheck } from 'lucide-react';

const APIPage = () => {
  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <h1 className="text-display" style={{ fontSize: '5rem', marginBottom: '1.5rem' }}>API Reference</h1>
          <p className="text-body" style={{ maxWidth: '600px', marginBottom: '4rem' }}>
            Integrate VLMJudge into your evaluation pipelines with our high-performance REST API and Python SDK.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4rem', alignItems: 'start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
              <div style={{ display: 'flex', gap: '1.5rem' }}>
                <div style={{ width: '48px', height: '48px', backgroundColor: '#f0f0f0', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Terminal size={24} />
                </div>
                <div>
                  <h3 style={{ marginBottom: '0.5rem' }}>Python SDK</h3>
                  <p style={{ color: '#666' }}>Simple `pip install vlmjudge` to start evaluating images locally or in the cloud.</p>
                </div>
              </div>
              <div style={{ display: 'flex', gap: '1.5rem' }}>
                <div style={{ width: '48px', height: '48px', backgroundColor: '#f0f0f0', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Cpu size={24} />
                </div>
                <div>
                  <h3 style={{ marginBottom: '0.5rem' }}>Inference Engine</h3>
                  <p style={{ color: '#666' }}>Optimized for low-latency scoring using distilled student models.</p>
                </div>
              </div>
            </div>

            <div className="glass-panel" style={{ backgroundColor: 'var(--color-bg-dark)', padding: '2rem', borderRadius: '1.5rem', color: '#fff' }}>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#ff5f56' }}></div>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#ffbd2e' }}></div>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#27c93f' }}></div>
              </div>
              <pre style={{ margin: 0, fontSize: '0.9rem', color: '#999', overflowX: 'auto' }}>
                <code>{`import vlmjudge

# Initialize the judge
judge = vlmjudge.load("vlmjudge-v1")

# Compare two images
result = judge.compare(
    img1="path/to/img1.jpg",
    img2="path/to/img2.jpg",
    prompt="A high quality portrait"
)

print(f"Winner: {result.winner}")
print(f"Reasoning: {result.reasoning}")`}</code>
              </pre>
            </div>
          </div>
        </motion.div>
      </section>
    </div>
  );
};

export default APIPage;
