---
name: Jayesh Pandey
summary: Project documentation and setup instructions.
---

# VLMJudge Frontend

This is the React-based frontend for the **VLMJudge** project, a multimodal evaluation and reasoning framework.

## Project Overview

VLMJudge provides a sophisticated UI to visualize image comparisons, ranking results, and deep vision-language reasoning. This frontend is designed to showcase the capabilities of the VLMJudge API and provide a user-friendly interface for researchers and developers.

## Features

- **Hero Dashboard**: Overview of VLMJudge capabilities.
- **Evaluation Showcase**: Visualization of pairwise image comparisons.
- **Architecture Insights**: Deep dive into the hybrid decision system.
- **API Portal**: Documentation and access to the evaluation endpoints.

## Tech Stack

- **React**: UI library.
- **Vite**: Build tool and dev server.
- **Framer Motion**: Smooth animations and transitions.
- **Vanilla CSS**: Clean and modular styling.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. (Optional) Point the frontend at the FastAPI server:
   - Create `Frontend/.env` with `VITE_API_BASE_URL=http://127.0.0.1:8000`

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Build for production:
   ```bash
   npm run build
   ```

## Development

The project structure is organized into:
- `src/components`: UI components.
- `src/constants/data.js`: Centralized content and configuration.
- `src/assets`: Project-specific image and icon assets.

For more information about the core VLMJudge logic, refer to the root `README.md`.
