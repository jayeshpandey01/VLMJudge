/**
 * Name: Jayesh Pandey
 * Summary: Source file for App.jsx in the src module.
 */

import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import CapabilitiesPage from './pages/CapabilitiesPage';
import ArchitecturePage from './pages/ArchitecturePage';
import APIPage from './pages/APIPage';
import ResearchPage from './pages/ResearchPage';
import EvaluationPage from './pages/EvaluationPage';

// Scroll to top on route change
const ScrollToTop = () => {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
};

function App() {
  return (
    <Router>
      <ScrollToTop />
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/capabilities" element={<CapabilitiesPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
          <Route path="/api" element={<APIPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
        </Routes>
      </main>
      <Footer />
    </Router>
  );
}

export default App;
