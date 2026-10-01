import os
import json
import uuid
import asyncio
import shutil
import numpy as np
from typing import List
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import edge_tts

app = FastAPI(title="Hybrid Video Automation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OS_MEDIA_DIR = "static_media"
UPLOAD_DIR = "user_uploads"
os.makedirs(OS_MEDIA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OS_MEDIA_DIR), name="static")

jobs = {}

class CustomScriptRequest(BaseModel):
    niche: str
    aspect_ratio: str = "16:9"
    script_text: str

@app.get("/")
def root():
    return {"status": "online", "system": "Hybrid Manual/Automated Engine"}

# --- STEP 1: UPLOAD LOCAL B-ROLL CLIPS ---
@app.post("/api/upload-clips")
async def upload_clips(files: List[UploadFile] = File(...)):
    saved_files = []
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    for file in files:
        if file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            file_path = os.path.join(session_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)

    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid video files uploaded.")

    return {"status": "success", "session_id": session_id, "clip_paths": saved_files}


# --- STEP 2: SUBTITLE & OVERLAY GENERATOR ---
def get_fallback_font(size: int):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_subtitle_frame(t, duration, sentence, width=1280, height=720):
    words = sentence.split()
    total_words = len(words)
    if total_words == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)

    words_per_sec = total_words / max(duration, 0.1)
    active_word_idx = min(int(t * words_per_sec), total_words - 1)

    chunk_size = 3
    chunk_start = (active_word_idx // chunk_size) * chunk_size
    chunk_words = words[chunk_start:chunk_start + chunk_size]
    chunk_text = " ".join(chunk_words).upper()

    img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = get_fallback_font(36 if width == 1280 else 48)

    box_w, box_h = int(width * 0.85), 110
    box_x = (width - box_w) // 2
    box_y = height - 180 if height == 720 else height - 350

    draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=15, fill=(15, 23, 42, 220), outline=(16, 185, 129, 255), width=3)
    draw.text((width // 2, box_y + (box_h // 2)), chunk_text, fill=(251, 191, 36, 255), font=font, anchor="mm")

    return np.array(img)


# --- STEP 3: PIPELINE EXECUTION ---
def run_hybrid_pipeline(job_id: str, script_text: str, niche: str, aspect_ratio: str, session_id: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Generating AI Voiceover..."}
        
        width, height = (1280, 720) if aspect_ratio == "16:9" else (720, 1280)
        audio_filename = f"audio_{job_id}.mp3"
        video_filename = f"video_{job_id}.mp4"
        audio_path = os.path.join(OS_MEDIA_DIR, audio_filename)
        video_path = os.path.join(OS_MEDIA_DIR, video_filename)

        # 1. Voiceover Generation
        async def make_audio():
            voice = "en-US-ChristopherNeural" if niche == "Military" else "en-US-GuyNeural"
            communicate = edge_tts.Communicate(script_text, voice)
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. Gather Uploaded Clips
        jobs[job_id]["progress"] = "Processing Uploaded Video B-Roll..."
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        uploaded_clips = []
        if os.path.exists(session_dir):
            uploaded_clips = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

        try:
            from moviepy.editor import AudioFileClip, VideoClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip
        except ImportError:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import ColorClip, VideoClip, VideoFileClip
            from moviepy.video.compositing.concatenate import concatenate_videoclips
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration

        # 3. Align Video Clips to Audio Length
        jobs[job_id]["progress"] = "Stitching Video Clips & Burning Subtitles..."
        bg_clips = []

        if uploaded_clips:
            clip_dur = total_duration / len(uploaded_clips)
            for path in uploaded_clips:
                try:
                    c = VideoFileClip(path).resize(newsize=(width, height)).without_audio()
                    if c.duration < clip_dur:
                        c = c.loop(duration=clip_dur)
                    else:
                        c = c.subclip(0, clip_dur)
                    bg_clips.append(c)
                except Exception as e:
                    print(f"Error loading clip {path}: {e}")

        if bg_clips:
            base_bg = concatenate_videoclips(bg_clips, method="compose").subclip(0, total_duration)
        else:
            base_bg = ColorClip(size=(width, height), color=(15, 23, 42), duration=total_duration)

        # 4. Generate Subtitles Mask
        def subtitle_gen(t):
            return render_subtitle_frame(t, total_duration, script_text, width=width, height=height)

        sub_clip = VideoClip(subtitle_gen, duration=total_duration).set_ismask(False)
        final_video = CompositeVideoClip([base_bg, sub_clip]).set_audio(audio_clip)

        # 5. Export MP4
        final_video.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            bitrate="2000k",
            verbose=False,
            logger=None
        )

        audio_clip.close()
        base_bg.close()
        final_video.close()

        jobs[job_id] = {
            "status": "completed",
            "filename": video_filename,
            "video_url": f"/static/{video_filename}"
        }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}


@app.post("/api/render-hybrid")
def render_hybrid(
    background_tasks: BackgroundTasks,
    script_text: str = Form(...),
    niche: str = Form(...),
    aspect_ratio: str = Form(...),
    session_id: str = Form(...)
):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": "Task queued..."}
    background_tasks.add_task(
        run_hybrid_pipeline, job_id, script_text, niche, aspect_ratio, session_id
    )
    return {"status": "success", "job_id": job_id}


@app.get("/api/job-status/{job_id}")
def check_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job
