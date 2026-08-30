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

        # Multi-section long form script generator for higher retention & duration
        sections = [
            f"Welcome back to the Channel! Today we are looking at an in-depth market breakdown for {ticker_sym}.",
            f"Looking at today's price movements, {ticker_sym} opened at ${prev_close} and closed at ${close_price}, showing a net change of {change} percent.",
            f"Over the last month, the stock traded within a range between a low of ${low_price} and a high of ${high_price}.",
            f"Trading volume has averaged {avg_vol:,} shares per day, highlighting active institutional interest.",
            f"Analyzing technical indicators, key support sits near ${low_price} while resistance remains established near ${high_price}.",
            f"Investors should keep a close eye on incoming earnings announcements and market macro factors moving forward.",
            f"If you found this analysis helpful, please hit the like button and subscribe for daily stock updates! Leave your thoughts in the comments."
        ]

        script = " ".join(sections)
        return {"status": "success", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def make_text_image(text_lines, width=720, height=1280):
    """Generates an RGB numpy frame with custom rendered text overlays without requiring ImageMagick."""
    img = Image.new('RGB', (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    
    # Render header
    draw.text((width // 2, 200), "DAILY MARKET REPORT", fill=(99, 102, 241), anchor="mm")
    draw.line([(100, 240), (620, 240)], fill=(51, 65, 85), width=3)
    
    # Render main content
    y_offset = 350
    for line in text_lines:
        draw.text((width // 2, y_offset), line, fill=(241, 245, 249), anchor="mm")
        y_offset += 60
        
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

        # 2. Render Video with Visual Cards
        jobs[job_id]["progress"] = "Rendering Video Visuals & Audio..."

        try:
            from moviepy.editor import AudioFileClip, ImageClip
        except (ImportError, AttributeError):
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import ImageClip

        audio_clip = AudioFileClip(audio_path)

        # Format script snippet for screen frame
        words = script_text.split()
        line_1 = " ".join(words[:6]) if len(words) >= 6 else script_text
        line_2 = " ".join(words[6:12]) if len(words) >= 12 else ""
        line_3 = " ".join(words[12:18]) if len(words) >= 18 else ""
        
        frame_array = make_text_image([line_1, line_2, line_3])

        # Create video clip with visual text matching audio length
        try:
            background_clip = ImageClip(frame_array).set_duration(audio_clip.duration)
            final_clip = background_clip.set_audio(audio_clip)
        except AttributeError:
            background_clip = ImageClip(frame_array).with_duration(audio_clip.duration)
            final_clip = background_clip.with_audio(audio_clip)

        final_clip.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio_clip.close()
        final_clip.close()

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
        # Fallback if env vars aren't set yet in Render
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
                "categoryId": "27"  # Education
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
