import os
import io
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
from PIL import Image, ImageDraw, ImageFont
import edge_tts

app = FastAPI(title="Multi-Niche YouTube Automation Engine")

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

PIXABAY_API_KEY = os.getenv("57821575-2c0420b658c3f9b8165a50eab", "")

class ScriptGenerateRequest(BaseModel):
    topic: str
    niche: str  # Options: Health, Military, Tech, Crime, Finance
    duration_minutes: int = 5
    aspect_ratio: str = "16:9"  # "16:9" for YouTube Longform, "9:16" for Shorts

class RenderRequest(BaseModel):
    script_json: str

@app.get("/")
def root():
    return {"status": "online", "system": "General Multi-Niche Automation Engine"}

# --- STEP 1: MULTI-NICHE SCRIPT GENERATOR ---
@app.post("/api/generate-script")
def generate_niche_script(data: ScriptGenerateRequest):
    try:
        topic = data.topic.strip()
        niche = data.niche.strip().title()
        scene_count = max(4, data.duration_minutes * 12)

        keyword_pools = {
            "Health": ["healthy food", "running man", "human brain", "doctor medical", "clean water"],
            "Military": ["military drone", "soldier field", "radar screen", "fighter jet", "satellite earth"],
            "Tech": ["robot artificial intelligence", "circuit board", "data center", "futuristic city", "cybersecurity"],
            "Crime": ["foggy street dark", "police car night", "old police file", "shadowy alley", "investigation board"],
            "Finance": ["stock market chart", "currency exchange", "bank skyscraper", "gold coins", "digital trading"]
        }

        selected_pool = keyword_pools.get(niche, ["abstract motion background", "cinematic landscape"])

        scenes = []
        for i in range(1, scene_count + 1):
            narration = f"Section {i} detailing {topic}. Analyzing key factors regarding {niche.lower()} trends and structural developments."
            keywords = [random.choice(selected_pool), topic]
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


# --- STEP 2: PIXABAY STOCK MEDIA SCRAPER ---
def fetch_pixabay_video(keyword: str, aspect_ratio: str = "16:9") -> str:
    if not PIXABAY_API_KEY:
        return ""

    safe_kw = "".join(c for c in keyword if c.isalnum() or c == ' ').strip().replace(" ", "_")
    cache_path = os.path.join(CACHE_DIR, f"{safe_kw}.mp4")

    if os.path.exists(cache_path):
        return cache_path

    # Search Pixabay Videos API
    url = f"https://pixabay.com/api/videos/?key={PIXABAY_API_KEY}&q={requests.utils.quote(keyword)}&per_page=3"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", [])
            if hits:
                videos = hits[0].get("videos", {})
                # Select medium or large quality MP4
                selected_video = videos.get("medium") or videos.get("large") or videos.get("small")
                if selected_video:
                    download_url = selected_video.get("url")
                    v_resp = requests.get(download_url, stream=True)
                    with open(cache_path, "wb") as f:
                        for chunk in v_resp.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                    return cache_path
    except Exception as e:
        print(f"Pixabay API error for '{keyword}': {e}")
    return ""


# --- STEP 3 & 4: AUDIO GENERATION, SUBTITLES & COMPOSITOR ---
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


def run_full_pipeline(job_id: str, script_json_str: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Parsing Script JSON..."}
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
        jobs[job_id]["progress"] = "Generating AI Voiceover via edge-tts..."
        async def make_audio():
            voice = "en-US-ChristopherNeural" if script_data.get("niche") == "Military" else "en-US-GuyNeural"
            communicate = edge_tts.Communicate(full_narration, voice)
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. B-Roll Media Fetcher
        jobs[job_id]["progress"] = "Fetching B-Roll Clips via Pixabay API..."
        clip_paths = []
        for sc in scenes:
            kw = sc.get("keywords", ["abstract"])[0]
            fetched = fetch_pixabay_video(kw, aspect_ratio=aspect_ratio)
            if fetched:
                clip_paths.append(fetched)

        # 3. Stitch & Compositing
        jobs[job_id]["progress"] = "Stitching Clips & Subtitles..."
        try:
            from moviepy.editor import AudioFileClip, VideoClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip
        except ImportError:
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import VideoClip, VideoFileClip
            from moviepy.video.compositing.concatenate import concatenate_videoclips
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration

        bg_clips = []
        if clip_paths:
            scene_duration = total_duration / len(scenes)
            for path in clip_paths:
                try:
                    c = VideoFileClip(path).resize(newsize=(width, height)).without_audio()
                    if c.duration < scene_duration:
                        c = c.loop(duration=scene_duration)
                    else:
                        c = c.subclip(0, scene_duration)
                    bg_clips.append(c)
                except Exception:
                    pass

        if bg_clips:
            base_bg = concatenate_videoclips(bg_clips, method="compose").subclip(0, total_duration)
        else:
            def color_canvas(t):
                img = Image.new('RGB', (width, height), color=(15, 23, 42))
                return np.array(img)
            base_bg = VideoClip(color_canvas, duration=total_duration)

        def subtitle_gen(t):
            return render_subtitle_frame(t, total_duration, full_narration, width=width, height=height)

        sub_clip = VideoClip(subtitle_gen, duration=total_duration).set_ismask(False)
        final_video = CompositeVideoClip([base_bg, sub_clip]).set_audio(audio_clip)

        final_video.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
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
