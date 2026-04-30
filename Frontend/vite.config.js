/**
 * Name: Jayesh Pandey
 * Summary: Source file for vite.config.js in the Frontend module.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
})
