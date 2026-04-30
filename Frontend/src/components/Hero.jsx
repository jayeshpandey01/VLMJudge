/**
 * Name: Jayesh Pandey
 * Summary: Source file for Hero.jsx in the components module.
 */

import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { HERO_CONTENT } from '../constants/data';

const Hero = () => {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"]
  });

  const y1 = useTransform(scrollYProgress, [0, 1], [0, 200]);
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -100]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [0.05, 0]);

  return (
    <section id="home" ref={ref} style={{ 
      position: 'relative', 
      minHeight: '100vh', 
      display: 'flex', 
      flexDirection: 'column', 
      justifyContent: 'center',
      paddingTop: '80px',
      overflow: 'hidden'
    }} className="container">
      
      <motion.div 
        style={{ position: 'relative', zIndex: 10, width: '100%', maxWidth: '900px', y: y2 }}
      >
        <motion.h1 
          className="text-display"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          style={{ marginBottom: '1.5rem', color: 'var(--color-text-light)' }}
        >
          {HERO_CONTENT.title}<br />
          <span style={{ color: 'rgba(0,0,0,0.3)' }}>{HERO_CONTENT.subtitle}</span>
        </motion.h1>

        <motion.p 
          className="text-body"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          style={{ maxWidth: '500px', marginBottom: '2.5rem' }}
        >
          {HERO_CONTENT.description}
        </motion.p>

        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
          style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}
        >
          <button className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.5rem 1.5rem 0.5rem 0.5rem' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'linear-gradient(135deg, #FF4D00, #992E00)' }}></div>
            {HERO_CONTENT.ctaText}
          </button>
        </motion.div>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.2, delay: 0.2 }}
        style={{
          position: 'absolute',
          top: '50%',
          right: '-5%',
          transform: 'translateY(-50%)',
          width: '65%',
          height: '80%',
          zIndex: 1,
          pointerEvents: 'none',
          y: y1
        }}
      >
        <img 
          src={HERO_CONTENT.bgImage} 
          alt="Hero background motion blur" 
          style={{ 
            width: '100%', 
            height: '100%', 
            objectFit: 'contain',
            filter: 'contrast(1.1) brightness(1.05)'
          }} 
        />
      </motion.div>

      {/* Floating service card */}
      <motion.div 
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, delay: 0.8 }}
        className="glass-panel"
        style={{
          position: 'absolute',
          top: '20%',
          right: '5%',
          padding: '2rem',
          zIndex: 20,
          width: 'min(320px, 90vw)',
          background: 'rgba(255,255,255,0.8)',
          y: y2
        }}
      >
        <h3 className="text-h3" style={{ fontSize: '1.4rem', marginBottom: '0.5rem' }}>{HERO_CONTENT.floatingCard.title}</h3>
        <p className="text-body" style={{ fontSize: '0.9rem' }}>{HERO_CONTENT.floatingCard.subtitle}</p>
      </motion.div>

      {/* Giant Background Text */}
      <motion.div
         style={{
           position: 'absolute',
           bottom: '-5%',
           left: '50%',
           x: '-50%',
           fontSize: '30vw',
           fontWeight: 900,
           lineHeight: 0.8,
           whiteSpace: 'nowrap',
           zIndex: 0,
           pointerEvents: 'none',
           color: 'var(--color-bg-dark)',
           opacity,
           y: y1
         }}
      >
        vlmjudge
      </motion.div>
    </section>
  );
};

export default Hero;
