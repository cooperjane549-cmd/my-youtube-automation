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
        history = stock.history(period="1mo")
        if history.empty:
            raise HTTPException(status_code=400, detail="Invalid ticker symbol")

        close_price = round(history["Close"].iloc[-1], 2)
        prev_close = round(history["Open"].iloc[-1], 2)
        change = round(((close_price - prev_close) / prev_close) * 100, 2)
        high_price = round(history["High"].max(), 2)
        low_price = round(history["Low"].min(), 2)
        avg_vol = int(history["Volume"].mean())

        ticker_sym = data.ticker.upper()

        # Multi-section script generator to scale video length
        sections = [
            f"Welcome back to the Channel! Today we are doing a deep dive into {ticker_sym}.",
            f"Looking at recent price movements, {ticker_sym} opened trading at ${prev_close} and closed at ${close_price}, showing a {change}% shift.",
            f"Over the past 30 days, the high touched ${high_price} while the low established technical support near ${low_price}.",
            f"Trading activity was substantial, recording an average daily volume of {avg_vol:,} shares.",
            f"Moving forward, watch for key macroeconomic reports and market sentiment around {ticker_sym}.",
            f"Hit subscribe for daily stock breakdowns, and drop your target price for {ticker_sym} in the comments below!"
        ]

        script = " ".join(sections)
        return {"status": "success", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def render_dynamic_frame(t, total_duration, script_text):
    """Generates a dynamic 720x1280 frame at time t with animated ticker metrics & captions."""
    width, height = 720, 1280
    
    # 1. Base image background
    img = Image.new('RGB', (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # 2. Header banner
    draw.rectangle([40, 60, 680, 140], fill=(30, 41, 59), outline=(99, 102, 241), width=2)
    draw.text((width // 2, 100), "MARKET AUTOMATION LAB", fill=(241, 245, 249), anchor="mm")

    # 3. Dynamic Visual Card (Pulsing graphic block based on time t)
    pulse = int(20 * np.sin(2 * np.pi * t / 2))
    card_y1 = 200
    card_y2 = 500
    draw.rectangle([60, card_y1, 660, card_y2], fill=(15, 30, 55), outline=(59, 130, 246), width=3)
    
    # Visual status bar
    progress_width = int((t / max(total_duration, 1)) * 560)
    draw.rectangle([80, 460, 80 + progress_width, 475], fill=(16, 185, 129))

    draw.text((width // 2, 280), "LIVE TICKER ANALYSIS", fill=(148, 163, 184), anchor="mm")
    draw.text((width // 2, 350), f"TIMECODE: {round(t, 1)}s / {round(total_duration, 1)}s", fill=(52, 211, 153), anchor="mm")

    # 4. Dynamic Word Captioning (Cycles through sentence chunks based on video playback time)
    words = script_text.split()
    words_per_sec = len(words) / max(total_duration, 1)
    current_word_idx = int(t * words_per_sec)

    chunk_size = 6
    start_idx = (current_word_idx // chunk_size) * chunk_size
    caption_chunk = " ".join(words[start_idx : start_idx + chunk_size])

    # Draw Caption Background Box
    draw.rectangle([40, 800, 680, 1050], fill=(2, 6, 23), outline=(71, 85, 105), width=2)
    draw.text((width // 2, 840), "VOICEOVER CAPTIONS", fill=(99, 102, 241), anchor="mm")
    
    # Render active words on screen
    if caption_chunk:
        draw.text((width // 2, 930), caption_chunk, fill=(255, 255, 255), anchor="mm")

    return np.array(img)


def run_video_pipeline(job_id: str, script_text: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Generating Audio Voiceover..."}

        audio_filename = f"audio_{job_id}.mp3"
        video_filename = f"video_{job_id}.mp4"
        audio_path = os.path.join(OS_MEDIA_DIR, audio_filename)
        video_path = os.path.join(OS_MEDIA_DIR, video_filename)

        # 1. Generate Voiceover
        async def make_audio():
            communicate = edge_tts.Communicate(script_text, "en-US-ChristopherNeural")
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. Render Dynamic Animated Video
        jobs[job_id]["progress"] = "Rendering Dynamic Frame Sequence..."

        try:
            from moviepy.editor import AudioFileClip, VideoClip
        except (ImportError, AttributeError):
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import VideoClip

        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        # MoviePy function frame generator driven by time parameter t
        def make_frame(t):
            return render_dynamic_frame(t, duration, script_text)

        try:
            video_clip = VideoClip(make_frame, duration=duration).set_audio(audio_clip)
        except AttributeError:
            video_clip = VideoClip(make_frame, duration=duration).with_audio(audio_clip)

        video_clip.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
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
            "message": f"Video '{data.title}' approved locally! Add YouTube API keys to Render to enable auto-publishing.",
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
