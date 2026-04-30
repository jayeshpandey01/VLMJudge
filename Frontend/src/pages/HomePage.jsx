/**
 * Name: Jayesh Pandey
 * Summary: Source file for HomePage.jsx in the pages module.
 */

import React from 'react';
import Hero from '../components/Hero';
import About from '../components/About';
import Services from '../components/Services';
import Branding from '../components/Branding';
import Team from '../components/Team';

const HomePage = () => {
  return (
    <>
      <Hero />
      <About />
      <Services />
      <Branding />
      <Team />
    </>
  );
};

export default HomePage;
