/**
 * Name: Jayesh Pandey
 * Summary: Source file for Team.jsx in the components module.
 */

import React, { useRef, useState, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { TEAM_CONTENT, STORIES } from '../constants/data';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const TeamCard = ({ member, index }) => {
  const cardRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    const checkActive = () => {
      if (cardRef.current) {
        const rect = cardRef.current.getBoundingClientRect();
        const center = window.innerWidth / 2;
        const cardCenter = rect.left + rect.width / 2;
        if (Math.abs(cardCenter - center) < 200) {
          setIsActive(true);
        } else {
          setIsActive(false);
        }
      }
    };

    window.addEventListener('scroll', checkActive);
    const container = document.getElementById('team-container');
    if (container) {
      container.addEventListener('scroll', checkActive);
    }

    checkActive();
    return () => {
      window.removeEventListener('scroll', checkActive);
      if (container) {
        container.removeEventListener('scroll', checkActive);
      }
    };
  }, []);

  return (
    <motion.div
      ref={cardRef}
      animate={{
        scale: isActive ? 1.1 : 0.85,
        opacity: isActive ? 1 : 0.6,
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      style={{
        flexShrink: 0,
        width: '400px',
        height: '400px',
        borderRadius: '3rem',
        overflow: 'hidden',
        position: 'relative',
        border: isActive ? '1px solid rgba(255,255,255,0.4)' : '1px solid rgba(255,255,255,0.1)',
        cursor: 'pointer'
      }}
    >
      <img 
        src={member.image} 
        alt={member.name} 
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          filter: isActive ? 'none' : 'grayscale(100%)'
        }}
      />
      <motion.div 
        animate={{ opacity: isActive ? 1 : 0 }}
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          padding: '3rem',
          background: 'linear-gradient(to top, rgba(0,0,0,0.95), transparent)',
          pointerEvents: 'none'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem' }}>
          <div style={{ width: '12px', height: '12px', backgroundColor: 'var(--color-accent)', borderRadius: '50%' }}></div>
          <h3 className="text-h3" style={{ fontSize: '2rem', marginBottom: '0.4rem' }}>{member.name}</h3>
        </div>
        <p className="text-body-dark" style={{ fontSize: '1.1rem' }}>{member.role}</p>
      </motion.div>
    </motion.div>
  );
};

const StoryCard = ({ item, index }) => {
  const cardRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    const checkActive = () => {
      if (cardRef.current) {
        const rect = cardRef.current.getBoundingClientRect();
        const center = window.innerWidth / 2;
        const cardCenter = rect.left + rect.width / 2;
        if (Math.abs(cardCenter - center) < 150) {
          setIsActive(true);
        } else {
          setIsActive(false);
        }
      }
    };

    window.addEventListener('scroll', checkActive);
    const container = document.getElementById('stories-container');
    if (container) {
      container.addEventListener('scroll', checkActive);
    }

    checkActive();
    return () => {
      window.removeEventListener('scroll', checkActive);
      if (container) {
        container.removeEventListener('scroll', checkActive);
      }
    };
  }, []);

  return (
    <motion.div
      ref={cardRef}
      animate={{
        scale: isActive ? 1.1 : 0.9,
        backgroundColor: isActive ? '#fff' : 'rgba(255,255,255,0.05)',
        color: isActive ? '#000' : '#fff',
      }}
      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      style={{
        flexShrink: 0,
        width: '320px',
        height: '320px',
        borderRadius: '3rem',
        padding: '3rem',
        position: 'relative',
        textAlign: 'left',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-end',
        border: isActive ? 'none' : '1px solid rgba(255,255,255,0.1)',
        cursor: 'pointer'
      }}
    >
      <div style={{ position: 'relative', zIndex: 1 }}>
        <p style={{ color: 'var(--color-accent)', fontSize: '0.9rem', fontWeight: 600, marginBottom: '1rem' }}>{item.handle}</p>
        <h3 style={{ fontSize: '1.75rem', fontWeight: 600, marginBottom: '1.25rem' }}>{item.title}</h3>
        <p style={{ fontSize: '1rem', opacity: 0.7, lineHeight: 1.6 }}>{item.text}</p>
      </div>
    </motion.div>
  );
};

const Team = () => {
  const containerRef = useRef(null);
  const scrollRef1 = useRef(null);
  const scrollRef2 = useRef(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const x1 = useTransform(scrollYProgress, [0, 1], [-100, 100]);
  const x2 = useTransform(scrollYProgress, [0, 1], [100, -100]);
  const yGiant = useTransform(scrollYProgress, [0, 1], [200, -200]);

  const scroll = (ref, direction) => {
    if (ref.current) {
      const { scrollLeft, clientWidth } = ref.current;
      const scrollTo = direction === 'left' ? scrollLeft - clientWidth / 2 : scrollLeft + clientWidth / 2;
      ref.current.scrollTo({ left: scrollTo, behavior: 'smooth' });
    }
  };

  return (
    <section id="portfolio" className="dark-section section-padding" style={{ position: 'relative', overflow: 'hidden' }} ref={containerRef}>
      
      {/* Background Giant Text */}
      <motion.div
         style={{
           position: 'absolute',
           top: '20%',
           left: '50%',
           x: '-50%',
           fontSize: '30vw',
           fontWeight: 900,
           lineHeight: 0.8,
           whiteSpace: 'nowrap',
           zIndex: 0,
           pointerEvents: 'none',
           color: 'rgba(255, 255, 255, 0.03)',
           y: yGiant
         }}
      >
        team
      </motion.div>

      <div className="container" style={{ position: 'relative', zIndex: 10 }}>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '8rem' }}>
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            style={{ maxWidth: '900px', textAlign: 'right' }}
          >
            <p style={{ color: 'var(--color-accent)', fontWeight: 600, fontSize: '0.9rem', marginBottom: '2rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              {TEAM_CONTENT.tag}
            </p>
            <h2 className="text-h2" style={{ fontWeight: 400, lineHeight: 1.1, fontSize: '4rem' }}>
              {TEAM_CONTENT.title.split(' experts').map((part, index) => (
                <React.Fragment key={index}>
                  {part}
                  {index === 0 && <><span className="glowing-orb"></span> experts</>}
                </React.Fragment>
              ))}
            </h2>
          </motion.div>
        </div>

        {/* Team Carousel - Scroll Driven */}
        <div style={{ position: 'relative', overflow: 'hidden', width: 'calc(100% + 8rem)', marginLeft: '-4rem' }}>
          <motion.div 
            id="team-container"
            ref={scrollRef1}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '2.5rem', 
              overflowX: 'auto', 
              padding: '4rem 4rem', 
              scrollbarWidth: 'none', 
              msOverflowStyle: 'none',
              x: x1
            }}
          >
             {[...TEAM_CONTENT.members, ...TEAM_CONTENT.members].map((member, index) => (
               <TeamCard key={index} member={member} index={index} />
             ))}
          </motion.div>
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem' }}>
           <button 
             onClick={() => scroll(scrollRef1, 'left')}
             style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: '#fff', color: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }} 
             className="hover-scale"
           >
             <ChevronLeft size={28} />
           </button>
           <button 
             onClick={() => scroll(scrollRef1, 'right')}
             style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'var(--color-accent)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }} 
             className="hover-scale"
           >
             <ChevronRight size={28} />
           </button>
        </div>

        {/* Stories of Growth */}
        <div style={{ marginTop: '12rem', textAlign: 'center' }}>
          <motion.h2 
            className="text-h2"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            style={{ marginBottom: '2rem', fontSize: '5rem' }}
          >
            Stories of growth<br/>and impact
          </motion.h2>
          <motion.p 
            className="text-body-dark"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            style={{ maxWidth: '600px', margin: '0 auto 6rem', fontSize: '1.1rem', lineHeight: 1.6 }}
          >
            From startups to established brands, we transform ideas into results that inspire communities and drive business.
          </motion.p>

          <div style={{ position: 'relative', overflow: 'hidden', width: 'calc(100% + 8rem)', marginLeft: '-4rem' }}>
            <motion.div 
              id="stories-container"
              ref={scrollRef2}
              style={{ 
                display: 'flex', 
                gap: '2.5rem', 
                overflowX: 'auto', 
                padding: '4rem 4rem', 
                scrollbarWidth: 'none', 
                msOverflowStyle: 'none',
                x: x2
              }}
            >
              {[...STORIES, ...STORIES].map((item, index) => (
                 <StoryCard key={index} item={item} index={index} />
              ))}
            </motion.div>
          </div>
          
           <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '2rem' }}>
             <button 
               onClick={() => scroll(scrollRef2, 'left')}
               style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: '#fff', color: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center' }} 
               className="hover-scale"
             >
               <ChevronLeft size={28} />
             </button>
             <button 
               onClick={() => scroll(scrollRef2, 'right')}
               style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'var(--color-accent)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center' }} 
               className="hover-scale"
             >
               <ChevronRight size={28} />
             </button>
          </div>
        </div>

      </div>
    </section>
  );
};

export default Team;
