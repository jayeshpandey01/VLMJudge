/**
 * Name: Jayesh Pandey
 * Summary: Source file for EvaluationPage.jsx in the pages module.
 */

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Image as ImageIcon, CheckCircle, AlertCircle, RefreshCw, ChevronRight, MessageSquare, ShieldCheck, Zap } from 'lucide-react';

const EvaluationPage = () => {
  const [imageA, setImageA] = useState(null);
  const [imageB, setImageB] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fileInputARef = useRef(null);
  const fileInputBRef = useRef(null);

  const handleImageUpload = (e, side) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (side === 'A') setImageA(reader.result);
        else setImageB(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const runEvaluation = async () => {
    if (!imageA || !imageB || !prompt) {
      setError('Please provide a prompt and two images for comparison.');
      return;
    }

    setError(null);
    setIsEvaluating(true);
    setResult(null);

    const baseUrlRaw = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
    const baseUrl = String(baseUrlRaw).replace(/\/+$/, '');

    try {
      const resp = await fetch(`${baseUrl}/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, imageA, imageB }),
      });

      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        const msg = data?.detail || data?.error || `HTTP ${resp.status}`;
        throw new Error(msg);
      }

      let reasoning = data?.reasoning || '';
      if (!reasoning) {
        const explainResp = await fetch(`${baseUrl}/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, imageA, imageB }),
        });
        const explainData = await explainResp.json().catch(() => null);
        if (explainResp.ok) {
          reasoning = explainData?.reasoning_full || explainData?.reasoning_short || '';
        }
      }

      const latencyMs = Number(data?.timing_ms?.total ?? 0);
      const winnerRaw = String(data?.winner || 'tie');
      const winner = winnerRaw === 'A' || winnerRaw === 'B' ? winnerRaw : 'Tie';

      setResult({
        winner,
        confidence: Number(data?.confidence ?? 0),
        scores: {
          A: Number(data?.scoreA ?? 0.5).toFixed(2),
          B: Number(data?.scoreB ?? 0.5).toFixed(2),
        },
        reasoning: reasoning || 'No reasoning available for this result.',
        method: String(data?.method || 'unknown'),
        latency: `${(latencyMs / 1000).toFixed(2)}s`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Evaluation failed.');
    } finally {
      setIsEvaluating(false);
    }
  };

  const reset = () => {
    setImageA(null);
    setImageB(null);
    setPrompt('');
    setResult(null);
    setError(null);
  };

  return (
    <div style={{ paddingTop: '100px', minHeight: '100vh', backgroundColor: '#fcfcfc' }}>
      <section className="section-padding container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
            <h1 className="text-display" style={{ fontSize: '4.5rem', marginBottom: '1rem' }}>VLMJudge <span style={{ color: 'var(--color-accent)' }}>Live</span></h1>
            <p className="text-body" style={{ maxWidth: '600px', margin: '0 auto' }}>
              Compare two AI-generated images against a prompt to get a calibrated preference score and structured reasoning.
            </p>
          </div>

          {!result && !isEvaluating && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '3rem' }}
            >
              <div style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
                 <div style={{ position: 'relative', marginBottom: '1rem' }}>
                   <MessageSquare size={18} style={{ position: 'absolute', left: '1.5rem', top: '50%', transform: 'translateY(-50%)', color: '#999' }} />
                   <input 
                     type="text" 
                     placeholder="Enter the prompt used to generate these images..." 
                     value={prompt}
                     onChange={(e) => setPrompt(e.target.value)}
                     style={{ 
                       width: '100%', 
                       padding: '1.5rem 1.5rem 1.5rem 3.5rem', 
                       borderRadius: '1.5rem', 
                       border: '1px solid #eee', 
                       backgroundColor: '#fff', 
                       fontSize: '1.1rem',
                       outline: 'none',
                       boxShadow: '0 4px 20px rgba(0,0,0,0.02)'
                     }}
                   />
                 </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                {/* Image A */}
                <div 
                  onClick={() => fileInputARef.current.click()}
                  style={{ 
                    height: '400px', 
                    borderRadius: '2.5rem', 
                    border: '2px dashed #eee', 
                    backgroundColor: '#fff', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    cursor: 'pointer',
                    overflow: 'hidden',
                    position: 'relative',
                    transition: 'all 0.3s ease'
                  }}
                  className="hover-scale"
                >
                  {imageA ? (
                    <img src={imageA} alt="Option A" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <>
                      <div style={{ width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#f9f9f9', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                        <ImageIcon size={24} color="#ccc" />
                      </div>
                      <p style={{ fontWeight: 600, color: '#666' }}>Upload Image A</p>
                      <p style={{ fontSize: '0.8rem', color: '#999' }}>PNG, JPG or WebP</p>
                    </>
                  )}
                  <input type="file" ref={fileInputARef} hidden onChange={(e) => handleImageUpload(e, 'A')} accept="image/*" />
                  <div style={{ position: 'absolute', top: '1.5rem', left: '1.5rem', backgroundColor: 'var(--color-bg-dark)', color: '#fff', padding: '0.4rem 1rem', borderRadius: '1rem', fontSize: '0.8rem', fontWeight: 600 }}>Option A</div>
                </div>

                {/* Image B */}
                <div 
                  onClick={() => fileInputBRef.current.click()}
                  style={{ 
                    height: '400px', 
                    borderRadius: '2.5rem', 
                    border: '2px dashed #eee', 
                    backgroundColor: '#fff', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    cursor: 'pointer',
                    overflow: 'hidden',
                    position: 'relative',
                    transition: 'all 0.3s ease'
                  }}
                  className="hover-scale"
                >
                  {imageB ? (
                    <img src={imageB} alt="Option B" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <>
                      <div style={{ width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#f9f9f9', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                        <ImageIcon size={24} color="#ccc" />
                      </div>
                      <p style={{ fontWeight: 600, color: '#666' }}>Upload Image B</p>
                      <p style={{ fontSize: '0.8rem', color: '#999' }}>PNG, JPG or WebP</p>
                    </>
                  )}
                  <input type="file" ref={fileInputBRef} hidden onChange={(e) => handleImageUpload(e, 'B')} accept="image/*" />
                  <div style={{ position: 'absolute', top: '1.5rem', left: '1.5rem', backgroundColor: 'var(--color-bg-dark)', color: '#fff', padding: '0.4rem 1rem', borderRadius: '1rem', fontSize: '0.8rem', fontWeight: 600 }}>Option B</div>
                </div>
              </div>

              {error && (
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', justifyContent: 'center', color: '#ff4d4d', fontSize: '0.9rem' }}>
                  <AlertCircle size={16} />
                  {error}
                </div>
              )}

              <div style={{ textAlign: 'center' }}>
                <button 
                  onClick={runEvaluation}
                  className="btn btn-primary" 
                  style={{ padding: '1.25rem 4rem', borderRadius: '2rem', fontSize: '1.1rem' }}
                >
                  Judge Comparison
                </button>
              </div>
            </motion.div>
          )}

          {isEvaluating && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }}
              style={{ textAlign: 'center', padding: '6rem 0' }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                style={{ display: 'inline-block', marginBottom: '2rem' }}
              >
                <RefreshCw size={64} color="var(--color-accent)" />
              </motion.div>
              <h2 className="text-h2">Analyzing Visual Content...</h2>
              <p className="text-body" style={{ marginTop: '1rem' }}>Our VLM ensemble is comparing semantic details and scoring alignment.</p>
              
              <div style={{ maxWidth: '400px', margin: '3rem auto 0' }}>
                 <div style={{ height: '4px', width: '100%', backgroundColor: '#eee', borderRadius: '2px', overflow: 'hidden' }}>
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 2.5 }}
                      style={{ height: '100%', backgroundColor: 'var(--color-accent)' }}
                    />
                 </div>
              </div>
            </motion.div>
          )}

          {result && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }} 
              animate={{ opacity: 1, scale: 1 }}
              style={{ display: 'flex', flexDirection: 'column', gap: '4rem' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#fff', padding: '2rem 4rem', borderRadius: '2.5rem', boxShadow: '0 20px 40px rgba(0,0,0,0.03)' }}>
                <div>
                   <p style={{ color: '#666', fontSize: '0.9rem', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Preference Winner</p>
                   <h2 style={{ fontSize: '3rem', fontWeight: 800 }}>Option {result.winner}</h2>
                </div>
                <div style={{ textAlign: 'right' }}>
                   <p style={{ color: '#666', fontSize: '0.9rem', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Confidence Score</p>
                   <h2 style={{ fontSize: '3rem', fontWeight: 800, color: 'var(--color-accent)' }}>{(result.confidence * 100).toFixed(0)}%</h2>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                 <div style={{ position: 'relative', borderRadius: '2rem', overflow: 'hidden', border: result.winner === 'A' ? '6px solid var(--color-accent)' : 'none' }}>
                    <img src={imageA} alt="Option A" style={{ width: '100%', height: '400px', objectFit: 'cover', opacity: result.winner === 'A' ? 1 : 0.6 }} />
                    <div style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem', backgroundColor: '#fff', padding: '0.5rem 1.5rem', borderRadius: '1rem', fontWeight: 700 }}>Score: {result.scores.A}</div>
                    {result.winner === 'A' && <div style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', backgroundColor: 'var(--color-accent)', color: '#fff', padding: '0.5rem 1.5rem', borderRadius: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><CheckCircle size={18} /> Winner</div>}
                 </div>
                 <div style={{ position: 'relative', borderRadius: '2rem', overflow: 'hidden', border: result.winner === 'B' ? '6px solid var(--color-accent)' : 'none' }}>
                    <img src={imageB} alt="Option B" style={{ width: '100%', height: '400px', objectFit: 'cover', opacity: result.winner === 'B' ? 1 : 0.6 }} />
                    <div style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem', backgroundColor: '#fff', padding: '0.5rem 1.5rem', borderRadius: '1rem', fontWeight: 700 }}>Score: {result.scores.B}</div>
                    {result.winner === 'B' && <div style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', backgroundColor: 'var(--color-accent)', color: '#fff', padding: '0.5rem 1.5rem', borderRadius: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}><CheckCircle size={18} /> Winner</div>}
                 </div>
              </div>

              <div className="glass-panel" style={{ padding: '4rem', borderRadius: '3rem', backgroundColor: '#fff', border: '1px solid #eee' }}>
                 <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', marginBottom: '2rem' }}>
                    <ShieldCheck size={28} color="var(--color-accent)" />
                    <h3 className="text-h3">Reasoning & Alignment Summary</h3>
                 </div>
                 <p className="text-body" style={{ fontSize: '1.2rem', lineHeight: 1.6, color: '#333', marginBottom: '3rem' }}>
                   {result.reasoning}
                 </p>

                 <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', borderTop: '1px solid #eee', paddingTop: '3rem' }}>
                    <div>
                       <p style={{ color: '#999', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Engine</p>
                       <p style={{ fontWeight: 600 }}>{result.method}</p>
                    </div>
                    <div>
                       <p style={{ color: '#999', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Latency</p>
                       <p style={{ fontWeight: 600 }}>{result.latency}</p>
                    </div>
                    <div>
                       <p style={{ color: '#999', fontSize: '0.8rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Calibration</p>
                       <p style={{ fontWeight: 600 }}>Isotonic Regression</p>
                    </div>
                 </div>
              </div>

              <div style={{ textAlign: 'center' }}>
                <button 
                  onClick={reset}
                  className="btn btn-outline" 
                  style={{ padding: '1rem 3rem', borderRadius: '2rem' }}
                >
                  Evaluate New Pair
                </button>
              </div>
            </motion.div>
          )}
        </motion.div>
      </section>
    </div>
  );
};

export default EvaluationPage;
