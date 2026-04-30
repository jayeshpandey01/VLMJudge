/**
 * Name: Jayesh Pandey
 * Summary: Source file for Services.jsx in the components module.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { SERVICES } from '../constants/data';

const Services = () => {
  return (
    <section id="services" className="section-padding container">
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
        gap: '2rem',
        marginTop: '2rem'
      }}>
        {SERVICES.map((card, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: index * 0.1 }}
            className="hover-scale"
            style={{
              borderRadius: '2rem',
              overflow: 'hidden',
              position: 'relative',
              height: '450px',
              backgroundColor: card.dark ? 'var(--color-bg-dark)' : 'var(--color-bg-light)',
              color: card.dark ? 'var(--color-text-dark)' : 'var(--color-text-light)',
              border: card.dark ? 'none' : '1px solid rgba(0,0,0,0.05)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              padding: '2.5rem'
            }}
          >
            {card.image && (
              <img 
                src={card.image} 
                alt={card.title} 
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  zIndex: 0,
                  filter: 'brightness(0.6)'
                }}
              />
            )}
            
            <div style={{ position: 'relative', zIndex: 1 }}>
              <span style={{
                display: 'inline-block',
                padding: '0.4rem 1.2rem',
                borderRadius: '2rem',
                border: `1px solid ${card.dark || card.image ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.2)'}`,
                fontSize: '0.8rem',
                color: card.dark || card.image ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.6)',
                backdropFilter: 'blur(4px)'
              }}>
                {card.tag}
              </span>
            </div>

            <div style={{ position: 'relative', zIndex: 1, marginTop: 'auto' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.2rem' }}>
                <div style={{ width: '10px', height: '10px', backgroundColor: 'var(--color-accent)', borderRadius: '50%' }}></div>
                <h3 className="text-h3" style={{ fontSize: '1.75rem', color: card.image ? '#fff' : 'inherit', lineHeight: 1.2 }}>{card.title}</h3>
              </div>
              <p className="text-body" style={{ fontSize: '1rem', color: card.image ? 'rgba(255,255,255,0.7)' : (card.dark ? 'rgba(255,255,255,0.5)' : 'rgba(0,0,0,0.5)') }}>
                {card.desc}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
};

export default Services;
