/**
 * Name: Jayesh Pandey
 * Summary: Source file for About.jsx in the components module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { ABOUT_CONTENT } from '../constants/data';

const About = () => {
  return (
    <section id="about" className="section-padding container" style={{ position: 'relative', zIndex: 10 }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto', textAlign: 'center' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8 }}
        >
          <p style={{ color: 'var(--color-accent)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '1.5rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            {ABOUT_CONTENT.tag}
          </p>
          
            {ABOUT_CONTENT.title.split('VLMJudge').map((part, index) => (
              <React.Fragment key={index}>
                {part}
                {index === 0 && <><span className="glowing-orb"></span> VLMJudge</>}
              </React.Fragment>
            ))}
          
          <p className="text-body" style={{ marginTop: '2.5rem', maxWidth: '500px', margin: '2.5rem auto 0', fontSize: '1rem' }}>
            {ABOUT_CONTENT.description}
          </p>
        </motion.div>
      </div>
    </section>
  );
};

export default About;
