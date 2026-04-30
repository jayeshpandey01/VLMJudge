/**
 * Name: Jayesh Pandey
 * Summary: React component for the footer section, featuring a contact form, project branding, and navigation links with framer-motion animations.
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { FOOTER_CONTENT } from '../constants/data';

const Footer = () => {
  return (
    <footer style={{ backgroundColor: 'var(--color-bg-light)', position: 'relative', overflow: 'hidden' }}>
      
      {/* Contact Form Section */}
      <div className="container section-padding" style={{ position: 'relative', zIndex: 10 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '6rem', alignItems: 'center' }}>
          
          <motion.div
             initial={{ opacity: 0, x: -50 }}
             whileInView={{ opacity: 1, x: 0 }}
             viewport={{ once: true }}
             transition={{ duration: 0.8 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', fontWeight: 800, fontSize: '1.75rem', marginBottom: '4rem' }}>
              <div style={{ width: '32px', height: '32px', backgroundColor: 'var(--color-bg-dark)', borderRadius: '6px', position: 'relative' }}>
                  <div style={{position: 'absolute', top: 6, left: 6, width: 10, height: 10, backgroundColor: 'var(--color-accent)', borderRadius: '2.5px'}}></div>
              </div>
            </div>
            
            <h2 className="text-h2" style={{ fontWeight: 400, marginBottom: '1.5rem', fontSize: '3.5rem' }}>
              {FOOTER_CONTENT.ctaTitle.split(' ').map((word, i) => (
                <React.Fragment key={i}>{word}{i === 2 ? <br/> : ' '}</React.Fragment>
              ))}
            </h2>
            <p className="text-body" style={{ maxWidth: '400px', fontSize: '1rem', marginBottom: '2.5rem' }}>
              {FOOTER_CONTENT.ctaDesc}
            </p>
            
            <button className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.6rem 2rem 0.6rem 0.6rem', borderRadius: '3rem' }}>
              <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'linear-gradient(135deg, #FF4D00, #992E00)' }}></div>
              Start Evaluating
            </button>
          </motion.div>
          
          <motion.div
             initial={{ opacity: 0, y: 50 }}
             whileInView={{ opacity: 1, y: 0 }}
             viewport={{ once: true }}
             transition={{ duration: 0.8, delay: 0.2 }}
             style={{
               backgroundColor: '#fff',
               borderRadius: '3rem',
               padding: '4rem',
               boxShadow: '0 30px 60px rgba(0,0,0,0.08)',
               width: '100%',
               maxWidth: '600px'
             }}
          >
             <h3 className="text-h3" style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '2rem' }}>{FOOTER_CONTENT.formTitle}</h3>
             
             <form style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
               <input 
                 type="text" 
                 placeholder="Your project name" 
                 style={{ padding: '1.25rem', borderRadius: '1.5rem', border: '1px solid rgba(0,0,0,0.1)', backgroundColor: 'var(--color-bg-light)', fontFamily: 'inherit', fontSize: '1rem' }}
               />
               <input 
                 type="email" 
                 placeholder="Contact email" 
                 style={{ padding: '1.25rem', borderRadius: '1.5rem', border: '1px solid rgba(0,0,0,0.1)', backgroundColor: 'var(--color-bg-light)', fontFamily: 'inherit', fontSize: '1rem' }}
               />
               <button 
                 type="button" 
                 className="btn btn-outline" 
                 style={{ width: '100%', marginTop: '1rem', borderRadius: '1.5rem', padding: '1.25rem', fontSize: '1rem', fontWeight: 600 }}
               >
                 Send Request
               </button>
             </form>
          </motion.div>
          
        </div>
      </div>

      {/* Giant Background Text */}
      <motion.div
         initial={{ opacity: 0 }}
         whileInView={{ opacity: 0.05 }}
         viewport={{ once: true }}
         transition={{ duration: 1.5 }}
         style={{
           position: 'absolute',
           bottom: '15%',
           left: '50%',
           transform: 'translateX(-50%)',
           fontSize: '35vw',
           fontWeight: 900,
           lineHeight: 0.8,
           whiteSpace: 'nowrap',
           zIndex: 0,
           pointerEvents: 'none',
           color: 'var(--color-bg-dark)'
         }}
      >
        vlmjudge
      </motion.div>
      
      {/* Footer Bottom */}
      <div className="container" style={{ position: 'relative', zIndex: 10, paddingBottom: '3rem', marginTop: '6rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '4rem' }}>
          
          <div style={{ gridColumn: 'span 1' }}>
            <h2 className="text-h2" style={{ fontWeight: 500, letterSpacing: '-0.02em', marginBottom: '2.5rem', fontSize: '2.5rem' }}>
              We'll be glad to<br/>collaborate with you.
            </h2>
            <button className="btn btn-outline" style={{ borderRadius: '2rem', padding: '0.6rem 2rem' }}>
              Get started
            </button>
            <p style={{ marginTop: '5rem', fontSize: '0.8rem', color: 'rgba(0,0,0,0.4)' }}>© VLMJudge Project 2025</p>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
            <div>
              <h4 style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '1.1rem' }}>Location</h4>
              <p className="text-body" style={{ fontSize: '1rem' }}>{FOOTER_CONTENT.location}</p>
            </div>
            <div>
               <h4 style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '1.1rem' }}>Social</h4>
               <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '1rem', color: 'rgba(0,0,0,0.6)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                 {FOOTER_CONTENT.socials.map(social => (
                   <li key={social}><a href="#" className="hover-scale">{social}</a></li>
                 ))}
               </ul>
            </div>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}>
             <div>
              <h4 style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '1.1rem' }}>Contact</h4>
              <p className="text-body" style={{ fontSize: '1rem', whiteSpace: 'pre-line' }}>{FOOTER_CONTENT.contact}</p>
            </div>
            <div>
               <h4 style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '1.1rem' }}>Helpful Links</h4>
               <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '1rem', color: 'rgba(0,0,0,0.6)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                 {FOOTER_CONTENT.links.map(link => {
                   const href = link.toLowerCase() === 'about' ? '/about' : 
                                link.toLowerCase() === 'capabilities' ? '/capabilities' : 
                                link.toLowerCase() === 'architecture' ? '/architecture' : '/';
                   return (
                     <li key={link}><Link to={href} className="hover-scale">{link}</Link></li>
                   );
                 })}
               </ul>
            </div>
          </div>
          
        </div>
      </div>
    </footer>
  );
};

export default Footer;
