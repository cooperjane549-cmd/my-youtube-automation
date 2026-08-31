import asyncio
import os
import uuid
import edge_tts
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Google API client imports for YouTube Publishing
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OS_MEDIA_DIR = "static_media"
os.makedirs(OS_MEDIA_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OS_MEDIA_DIR), name="static")

# In-memory job state tracker
jobs = {}

class ScriptRequest(BaseModel):
    ticker: str

class RenderRequest(BaseModel):
    script_text: str

class UploadRequest(BaseModel):
    video_filename: str
    title: str
    description: str


@app.get("/")
def health_check():
    return {"status": "online", "service": "YouTube Automation Engine"}


@app.post("/api/generate-script")
def generate_script(data: ScriptRequest):
    try:
        stock = yf.Ticker(data.ticker)
        history = stock.history(period="1y")
        if history.empty:
            raise HTTPException(status_code=400, detail="Invalid ticker symbol")

        close_price = round(history["Close"].iloc[-1], 2)
        prev_close = round(history["Open"].iloc[-1], 2)
        change = round(((close_price - prev_close) / prev_close) * 100, 2)
        high_price = round(history["High"].max(), 2)
        low_price = round(history["Low"].min(), 2)
        avg_vol = int(history["Volume"].mean())
        
        # Calculate Technical Indicators for extra narrative depth
        sma_50 = round(history["Close"].rolling(window=50).mean().iloc[-1], 2)
        sma_200 = round(history["Close"].rolling(window=200).mean().iloc[-1], 2)
        
        ticker_sym = data.ticker.upper()

        # Expanded Multi-Section Script (Designed to reach 3 to 5 minutes duration)
        sections = [
            f"Welcome back to the Channel! Today we are taking an exhaustive deep dive into {ticker_sym}.",
            f"Let's start by breaking down recent price activity. {ticker_sym} opened trading today at ${prev_close} and ended the session at ${close_price}, marking a net movement of {change} percent.",
            f"Looking across the broader fifty-two week performance, the stock established a strong support floor around ${low_price}, while peaking at a resistance high of ${high_price}.",
            f"Trading volume remains steady, recording an average daily volume of {avg_vol:,} shares moving across exchanges.",
            f"Analyzing technical trends, the fifty-day moving average sits at ${sma_50}, while the long-term two-hundred-day moving average is floating at ${sma_200}.",
            f"When evaluating technical indicators like moving average convergence, price action hovering near these critical trendlines signals strategic interest from institutional buyers.",
            f"Macroeconomic factors, including sector-wide earnings reports, interest rate policy shifts, and broader market liquidity, continue to influence price volatility for {ticker_sym}.",
            f"Short-term traders should keep a close eye on incoming resistance targets around ${high_price}. A clean breakthrough past this boundary could trigger momentum buying.",
            f"Conversely, if selling pressure intensifies, key downside support rests firmly near ${low_price}, where buyers historically stepped in to absorb supply.",
            f"Long-term investors should weigh company fundamentals, earnings consistency, and balance sheet health before executing positioning strategy.",
            f"That concludes our multi-minute technical and fundamental breakdown for {ticker_sym}.",
            f"If you enjoyed this detailed report, make sure to like the video, hit subscribe, and turn on notifications so you never miss daily stock updates! Drop your price targets in the comments below."
        ]

        script = " ".join(sections)
        return {"status": "success", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def render_scene_frame(t, duration, sentences, width=540, height=960):
    """Renders highly legible, large text clips at lightweight 540x960 resolution for perfect sync."""
    num_scenes = max(len(sentences), 1)
    scene_duration = duration / num_scenes
    scene_idx = min(int(t // scene_duration), num_scenes - 1)
    
    current_text = sentences[scene_idx]

    color_palettes = [
        {"bg": (15, 23, 42), "card": (30, 41, 59), "accent": (99, 102, 241), "text": (255, 255, 255)},
        {"bg": (10, 25, 47), "card": (23, 42, 69), "accent": (16, 185, 129), "text": (255, 255, 255)},
        {"bg": (24, 15, 38), "card": (45, 27, 68), "accent": (236, 72, 153), "text": (255, 255, 255)},
        {"bg": (30, 27, 75), "card": (49, 46, 129), "accent": (245, 158, 11), "text": (255, 255, 255)},
        {"bg": (17, 24, 39), "card": (31, 41, 55), "accent": (14, 165, 233), "text": (255, 255, 255)}
    ]
    palette = color_palettes[scene_idx % len(color_palettes)]

    img = Image.new('RGB', (width, height), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    # 1. Header Banner
    draw.rectangle([20, 40, width - 20, 100], fill=palette["card"], outline=palette["accent"], width=3)
    draw.text((width // 2, 70), f"PART #{scene_idx + 1} / {num_scenes}", fill=palette["accent"], anchor="mm")

    # 2. Main Content Box with Large Font Simulation
    draw.rectangle([25, 130, width - 25, 750], fill=palette["card"], outline=(71, 85, 105), width=3)

    words = current_text.split()
    # 3 words per line makes font sizing fill the phone screen comfortably
    lines = [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]

    y_offset = 200
    for line in lines[:8]:
        # Emphasized bold visual text blocks
        draw.text((width // 2, y_offset), line.upper(), fill=palette["text"], anchor="mm")
        y_offset += 65

    # 3. Dynamic Progress Bar
    progress = int((t / max(duration, 1)) * (width - 60))
    draw.rectangle([30, 780, 30 + progress, 795], fill=palette["accent"])

    # 4. Footer
    draw.line([(50, 880), (width - 50, 880)], fill=palette["accent"], width=3)
    draw.text((width // 2, 915), "FINANCIAL AUTOMATION ENGINE", fill=(148, 163, 184), anchor="mm")

    return np.array(img)


def run_video_pipeline(job_id: str, script_text: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Generating Audio Voiceover..."}

        audio_filename = f"audio_{job_id}.mp3"
        video_filename = f"video_{job_id}.mp4"
        audio_path = os.path.join(OS_MEDIA_DIR, audio_filename)
        video_path = os.path.join(OS_MEDIA_DIR, video_filename)

        # 1. Generate Audio
        async def make_audio():
            communicate = edge_tts.Communicate(script_text, "en-US-ChristopherNeural")
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. Render Synchronized Video
        jobs[job_id]["progress"] = "Rendering High-Legibility Video Clips..."

        try:
            from moviepy.editor import AudioFileClip, VideoClip
        except (ImportError, AttributeError):
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import VideoClip

        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        sentences = [s.strip() for s in script_text.split('.') if s.strip()]
        if not sentences:
            sentences = [script_text]

        def frame_generator(t):
            return render_scene_frame(t, duration, sentences)

        try:
            video_clip = VideoClip(frame_generator, duration=duration).set_audio(audio_clip)
        except AttributeError:
            video_clip = VideoClip(frame_generator, duration=duration).with_audio(audio_clip)

        # Rendering at 12 FPS guarantees smooth processing on Render free servers without lag
        video_clip.write_videofile(
            video_path, fps=12, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio_clip.close()
        video_clip.close()

        # 3. Mark Complete
        jobs[job_id] = {
            "status": "completed",
            "filename": video_filename,
            "video_url": f"/static/{video_filename}",
        }
    except Exception as e:
        jobs[job_id] = {"status": "failed", "error": str(e)}


@app.post("/api/render-video")
def start_render(data: RenderRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": "Task queued..."}
    background_tasks.add_task(run_video_pipeline, job_id, data.script_text)
    return {"status": "success", "job_id": job_id}


@app.get("/api/job-status/{job_id}")
def check_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job


@app.post("/api/approve-upload")
def approve_and_upload(data: UploadRequest):
    file_path = os.path.join(OS_MEDIA_DIR, data.video_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    # YouTube API Integration
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        return {
            "status": "success",
            "message": f"Video '{data.title}' approved locally! Set YOUTUBE_REFRESH_TOKEN in Render environment to upload live.",
        }

    try:
        creds = Credentials(
            None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret
        )

        youtube = build("youtube", "v3", credentials=creds)

        body = {
            "snippet": {
                "title": data.title,
                "description": data.description,
                "tags": ["stocks", "finance", "market", "automation"],
                "categoryId": "27"
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }

        media = MediaFileUpload(file_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()

        video_id = response.get("id", "")
        return {
            "status": "success",
            "message": f"Published directly to YouTube! Video ID: {video_id}",
            "youtube_url": f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube Upload Error: {str(e)}")
