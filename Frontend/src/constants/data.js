/**
 * Name: Jayesh Pandey
 * Summary: Source file for data.js in the constants module.
 */

import heroBg from '../assets/hero_bg.png';
import serviceBg from '../assets/service_bg.png';
import teamPortrait from '../assets/team_portrait.png';
import bannerImg from '../assets/banner.jpg';
import architectureImg from '../assets/vlmjudge/architecture.png';
import compareSampleImg from '../assets/vlmjudge/compare_sample.png';

export const NAV_LINKS = [
  { name: 'Home', href: '/' },
  { name: 'About', href: '/about' },
  { name: 'Capabilities', href: '/capabilities' },
  { name: 'Architecture', href: '/architecture' },
  { name: 'Research', href: '/research' },
  { name: 'Evaluation', href: '/evaluation' },
  { name: 'API', href: '/api' },
];

export const HERO_CONTENT = {
  title: 'VLMJudge',
  subtitle: 'Visual Preference Intelligence',
  description: 'A production-grade multimodal evaluation framework for image comparison, ranking, and preference learning. Combining fast scoring with deep vision-language reasoning.',
  ctaText: 'Explore API',
  bgImage: bannerImg,
  floatingCard: {
    title: 'Model Evaluation',
    subtitle: 'Accurate and Calibrated'
  }
};

export const ABOUT_CONTENT = {
  tag: 'Intelligent Evaluation',
  title: 'VLMJudge — A Multimodal Reasoning Framework',
  description: 'Designed for image comparison and preference learning, VLMJudge produces accurate, calibrated, and explainable judgments using a hybrid system of fast models and deep reasoning ensembles.'
};

export const SERVICES = [
  {
    title: "Hybrid Decision System",
    desc: "Combines fast similarity-based models with deep vision-language reasoning for optimal speed and accuracy.",
    dark: true,
    tag: "Intelligence"
  },
  {
    title: "Multi-VLM Ensemble",
    desc: "Leverages state-of-the-art VLMs like Qwen2.5-VL to provide robust, consensus-based evaluations.",
    image: compareSampleImg,
    tag: "Robustness"
  },
  {
    title: "Explainable Reasoning",
    desc: "Every preference decision comes with structured reasoning, explaining exactly why one image was chosen over another.",
    image: serviceBg,
    tag: "Transparency"
  },
  {
    title: "Continuous Learning",
    desc: "Automated pipelines for logging disagreements, filtering high-quality samples, and retraining student models.",
    dark: false,
    tag: "Learning"
  }
];

export const BRANDING_CONTENT = {
  title: 'Advanced Multimodal Evaluation Capabilities',
  description1: 'VLMJudge provides a comprehensive suite of tools for evaluating AI-generated images, ensuring alignment with human preferences and semantic accuracy across diverse domains.',
  description2: 'Our framework is built for scale, supporting batch processing, confidence calibration, and disagreement-aware scoring to deliver reliable metrics for research and production.',
  categories: [
    "Pairwise Comparison", 
    "Single Image Scoring", 
    "Reasoning Extraction", 
    "Confidence Calibration", 
    "Batch Processing", 
    "Human Alignment", 
    "Continuous Retraining"
  ],
  bannerTitle: 'Intelligence built for alignment',
  bannerDesc: 'Not just scores — we provide the reasoning that drives model improvement.',
  bannerImage: bannerImg
};

export const TEAM_CONTENT = {
  tag: 'Project Scope',
  title: "VLMJudge is designed for LLM post-training pipelines, image generation evaluation, and research in multimodal alignment.",
  members: [
    { name: 'Preference Learning', role: 'Dataset construction and distillation', image: teamPortrait },
    { name: 'Model Evaluation', role: 'Benchmarking and metric tracking', image: teamPortrait },
    { name: 'Reasoning Engine', role: 'Structured output and explainability', image: teamPortrait },
    { name: 'Hybrid Scaling', role: 'Low-latency production deployment', image: teamPortrait },
    { name: 'Safety & Guardrails', role: 'Alignment and preference filtering', image: teamPortrait },
  ]
};

export const STORIES = [
  { id: 'llm', title: 'LLM Training', handle: '@posttraining', text: 'VLMJudge accelerates the feedback loop for RLHF in multimodal LLMs by providing high-quality reward signals.' },
  { id: 'gen', title: 'Image Generation', handle: '@genai', text: 'Evaluate the fidelity and prompt alignment of diffusion models with human-calibrated preference scores.' },
  { id: 'data', title: 'Dataset Curation', handle: '@datanengine', text: 'Automatically filter and rank millions of images to construct high-quality preference datasets for fine-tuning.', active: true },
  { id: 'research', title: 'Alignment Research', handle: '@alignment', text: 'Explore the boundaries of vision-language understanding with our modular evaluation pipeline.' },
  { id: 'prod', title: 'Production API', handle: '@deployment', text: 'Deploy robust evaluation metrics at scale with low-latency student models and deep VLM fallbacks.' }
];

export const FOOTER_CONTENT = {
  ctaTitle: 'Ready to evaluate your models?',
  ctaDesc: "Join the research community in building better aligned multimodal systems with VLMJudge.",
  formTitle: "Connect with the Project",
  location: 'VLMJudge Research Project',
  contact: 'Documentation & API\ngithub.com/vlmjudge',
  socials: ['GitHub', 'ArXiv', 'Twitter', 'LinkedIn'],
  links: ['About', 'Capabilities', 'Architecture', 'License']
};

