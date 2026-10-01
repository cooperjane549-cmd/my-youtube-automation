import os
import json
import uuid
import asyncio
import random
import requests
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import edge_tts

app = FastAPI(title="Fast Multi-Niche YouTube Automation Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OS_MEDIA_DIR = "static_media"
CACHE_DIR = "media_cache"
os.makedirs(OS_MEDIA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=OS_MEDIA_DIR), name="static")

jobs = {}
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")

class ScriptGenerateRequest(BaseModel):
    topic: str
    niche: str
    duration_minutes: int = 1
    aspect_ratio: str = "16:9"

class RenderRequest(BaseModel):
    script_json: str

@app.get("/")
def root():
    return {"status": "online", "system": "High-Speed Video Engine"}

# --- STEP 1: SCRIPT GENERATION ---
@app.post("/api/generate-script")
def generate_niche_script(data: ScriptGenerateRequest):
    try:
        topic = data.topic.strip()
        niche = data.niche.strip().title()
        
        # Kept lean (3 to 5 scenes max) for fast rendering on standard cloud servers
        scene_count = 4

        keyword_pools = {
            "Health": ["healthy human", "running athlete", "doctor medical", "clean water"],
            "Military": ["military drone", "soldier field", "radar screen", "fighter jet"],
            "Tech": ["artificial intelligence", "circuit board", "data center", "futuristic city"],
            "Crime": ["dark street", "police light night", "mysterious shadow", "interrogation"],
            "Finance": ["stock chart", "currency money", "skyscraper city", "digital trading"]
        }

        selected_pool = keyword_pools.get(niche, ["abstract motion", "cinematic landscape"])

        scenes = []
        for i in range(1, scene_count + 1):
            narration = f"Scene {i} on {topic}. Key insight into {niche.lower()} developments."
            keywords = [random.choice(selected_pool)]
            scenes.append({
                "scene_id": i,
                "narration": narration,
                "keywords": keywords
            })

        script_payload = {
            "topic": topic,
            "niche": niche,
            "aspect_ratio": data.aspect_ratio,
            "total_scenes": len(scenes),
            "scenes": scenes
        }

        return {"status": "success", "script_json": json.dumps(script_payload, indent=2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- STEP 2: PIXABAY MEDIA SCRAPER ---
def fetch_pixabay_video(keyword: str) -> str:
    if not PIXABAY_API_KEY:
        return ""

    safe_kw = "".join(c for c in keyword if c.isalnum() or c == ' ').strip().replace(" ", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe_kw}.mp4")

    if os.path.exists(cache_path):
        return cache_path

    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={requests.utils.quote(keyword)}&per_page=3"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            hits = resp.json().get("hits", [])
            if hits:
                videos = hits[0].get("videos", {})
                # Use tiny/small files to accelerate network downloading and video decoding speed
                selected = videos.get("small") or videos.get("tiny") or videos.get("medium")
                if selected:
                    v_resp = requests.get(selected.get("url"), stream=True)
                    with open(cache_path, "wb") as f:
                        for chunk in v_resp.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                    return cache_path
    except Exception as e:
        print(f"Pixabay fetch error: {e}")
    return ""


# --- STEP 3: HIGH-SPEED RENDERER ---
def run_full_pipeline(job_id: str, script_json_str: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Parsing Script..."}
        script_data = json.loads(script_json_str)
        scenes = script_data.get("scenes", [])
        aspect_ratio = script_data.get("aspect_ratio", "16:9")

        width, height = (1280, 720) if aspect_ratio == "16:9" else (720, 1280)

        full_narration = " ".join([s["narration"] for s in scenes])
        audio_filename = f"audio_{job_id}.mp3"
        video_filename = f"video_{job_id}.mp4"
        audio_path = os.path.join(OS_MEDIA_DIR, audio_filename)
        video_path = os.path.join(OS_MEDIA_DIR, video_filename)

        # 1. Voiceover Generation
        jobs[job_id]["progress"] = "Generating AI Voiceover..."
        async def make_audio():
            voice = "en-US-ChristopherNeural" if script_data.get("niche") == "Military" else "en-US-GuyNeural"
            communicate = edge_tts.Communicate(full_narration, voice)
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. Fetch B-roll
        jobs[job_id]["progress"] = "Downloading Clips..."
        clip_paths = []
        for sc in scenes:
            kw = sc.get("keywords", ["abstract"])[0]
            fetched = fetch_pixabay_video(kw)
            if fetched:
                clip_paths.append(fetched)

        # 3. FAST VIDEO STITCHING
        jobs[job_id]["progress"] = "Rendering Video..."
        
        try:
            from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips, ColorClip
        except ImportError:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import ColorClip, VideoFileClip
            from moviepy.video.compositing.concatenate import concatenate_videoclips

        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        scene_dur = total_duration / max(len(scenes), 1)

        processed_clips = []
        for path in clip_paths:
            try:
                c = VideoFileClip(path).resize(newsize=(width, height)).without_audio()
                if c.duration < scene_dur:
                    c = c.loop(duration=scene_dur)
                else:
                    c = c.subclip(0, scene_dur)
                processed_clips.append(c)
            except Exception:
                pass

        if processed_clips:
            final_clip = concatenate_videoclips(processed_clips, method="compose").subclip(0, total_duration)
        else:
            final_clip = ColorClip(size=(width, height), color=(15, 23, 42), duration=total_duration)

        final_clip = final_clip.set_audio(audio_clip)

        # High-Speed Encoding Options for Restricted Hardware
        final_clip.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            bitrate="1500k",
            verbose=False,
            logger=None
        )

        audio_clip.close()
        final_clip.close()

        jobs[job_id] = {
            "status": "completed",
            "filename": video_filename,
            "video_url": f"/static/{video_filename}"
        }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}


@app.post("/api/render-video")
def start_render(data: RenderRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": "Task queued..."}
    background_tasks.add_task(run_full_pipeline, job_id, data.script_json)
    return {"status": "success", "job_id": job_id}


@app.get("/api/job-status/{job_id}")
def check_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job
