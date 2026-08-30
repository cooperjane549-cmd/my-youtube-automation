# YouTube Automation Engine

An end-to-end 80/20 YouTube video creation and publishing platform powered by FastAPI, Edge-TTS, MoviePy, and an HTML5 Web Dashboard.

## Project Structure
- `.github/workflows/keep-alive.yml`: GitHub Action to prevent Render free-tier sleep.
- `frontend/index.html`: Dashboard UI (Hosted on GitHub Pages).
- `backend/main.py`: FastAPI backend engine handling script generation and background video rendering.
- `backend/Dockerfile`: Container configuration installing Python and FFmpeg binaries.
- `backend/render.yaml`: Render cloud deployment blueprint.

## Local Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
