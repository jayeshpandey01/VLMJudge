/**
 * Name: Jayesh Pandey
 * Summary: Source file for Branding.jsx in the components module.
 */

import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { BRANDING_CONTENT } from '../constants/data';

const Branding = () => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"]
  });

  const x1 = useTransform(scrollYProgress, [0, 1], [0, -100]);
  const x2 = useTransform(scrollYProgress, [0, 1], [0, 100]);

  return (
    <section ref={ref} className="section-padding container">
      <div style={{ textAlign: 'center', marginBottom: '6rem' }}>
        <div style={{ display: 'inline-block', marginBottom: '3rem' }}>
           <div style={{ width: '40px', height: '40px', backgroundColor: 'var(--color-bg-dark)', borderRadius: '6px', position: 'relative', margin: '0 auto' }}>
            <div style={{position: 'absolute', top: 8, left: 8, width: 12, height: 12, backgroundColor: 'var(--color-accent)', borderRadius: '2.5px'}}></div>
          </div>
        </div>
        
        <motion.h2 
          className="text-display"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          style={{ lineHeight: 0.95, letterSpacing: '-0.03em' }}
        >
          {BRANDING_CONTENT.title.split(' Capabilities').map((part, index) => (
            <React.Fragment key={index}>
              {part}
              {index === 0 && <><br/>Capabilities</>}
            </React.Fragment>
          ))}
        </motion.h2>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '4rem', marginBottom: '6rem', flexWrap: 'wrap' }}>
        <motion.p 
          className="text-body" 
          style={{ flex: '1 1 300px', fontSize: '1rem' }}
          initial={{ opacity: 0, x: -30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
        >
          {BRANDING_CONTENT.description1}
        </motion.p>
        <motion.p 
          className="text-body" 
          style={{ flex: '1 1 300px', fontSize: '1rem' }}
          initial={{ opacity: 0, x: 30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
        >
          {BRANDING_CONTENT.description2}
        </motion.p>
      </div>

      {/* Categories Scroller - Dynamic Marquee Feel */}
      <div style={{ 
        overflow: 'hidden',
        width: 'calc(100% + 8rem)',
        marginLeft: '-4rem',
        padding: '2rem 0'
      }}>
        <motion.div 
          style={{ 
            display: 'flex', 
            gap: '1.5rem', 
            x: x1,
            paddingLeft: '4rem'
          }}
        >
          {[...BRANDING_CONTENT.categories, ...BRANDING_CONTENT.categories].map((cat, index) => (
            <motion.div
              key={index}
              className="hover-scale"
              style={{
                flexShrink: 0,
                width: index % 3 === 0 ? '320px' : '260px',
                height: '320px',
                backgroundColor: index % 4 === 2 ? 'var(--color-bg-dark)' : 'rgba(0,0,0,0.05)',
                borderRadius: '2.5rem',
                padding: '2.5rem',
                position: 'relative',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                color: index % 4 === 2 ? '#fff' : 'inherit',
                border: index % 4 === 2 ? 'none' : '1px solid rgba(0,0,0,0.05)'
              }}
            >
              {index % 4 === 2 && (
                <img 
                  src={BRANDING_CONTENT.bannerImage} 
                  alt={cat} 
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    filter: 'brightness(0.5) contrast(1.2)',
                    zIndex: 0
                  }}
                />
              )}
              
              <div style={{ position: 'relative', zIndex: 1 }}>
                <span style={{
                  display: 'inline-block',
                  padding: '0.4rem 1.2rem',
                  borderRadius: '2rem',
                  border: `1px solid ${index % 4 === 2 ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.2)'}`,
                  fontSize: '0.8rem'
                }}>
                  {cat}
                </span>
              </div>
              
              <div style={{ position: 'relative', zIndex: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
                 <h3 style={{ fontSize: '1.75rem', fontWeight: 500, lineHeight: 1.2 }}>
                   {cat}
                 </h3>
                 {index % 4 === 2 && (
                   <button style={{ backgroundColor: '#fff', color: '#000', padding: '0.5rem 1.2rem', borderRadius: '2rem', fontSize: '0.85rem', fontWeight: 600 }}>Get started</button>
                 )}
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Banner */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1 }}
        style={{
          width: '100%',
          height: '600px',
          position: 'relative',
          overflow: 'hidden',
          marginTop: '4rem',
          display: 'flex',
          alignItems: 'flex-end',
          padding: '5rem',
          borderRadius: '4rem'
        }}
      >
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'linear-gradient(to right, #000 30%, transparent 100%)',
          zIndex: 1
        }}></div>
        <img 
          src={BRANDING_CONTENT.bannerImage} 
          alt="Banner background" 
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            filter: 'brightness(0.7) contrast(1.5)',
            zIndex: 0
          }}
        />
        <motion.div 
          style={{ position: 'relative', zIndex: 2, color: '#fff', x: x2 }}
        >
          <h2 className="text-display" style={{ fontSize: '6rem', color: '#fff', marginBottom: '1.5rem' }}>{BRANDING_CONTENT.bannerTitle}</h2>
          <p style={{ maxWidth: '450px', fontSize: '1.1rem', opacity: 0.8, lineHeight: 1.6 }}>{BRANDING_CONTENT.bannerDesc}</p>
        </motion.div>
      </motion.div>
    </section>
  );
};

export default Branding;
