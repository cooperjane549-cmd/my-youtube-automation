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
from PIL import Image, ImageDraw

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


def make_scene_image(section_title, text, scene_index, width=720, height=1280):
    """Generates a dynamic visual frame for each specific section clip with unique colors & layouts."""
    # Palette themes for scene clips to rotate visually
    color_palettes = [
        {"bg": (15, 23, 42), "card": (30, 41, 59), "accent": (99, 102, 241), "text": (241, 245, 249)},
        {"bg": (10, 25, 47), "card": (23, 42, 69), "accent": (16, 185, 129), "text": (255, 255, 255)},
        {"bg": (24, 15, 38), "card": (45, 27, 68), "accent": (236, 72, 153), "text": (243, 244, 246)},
        {"bg": (30, 27, 75), "card": (49, 46, 129), "accent": (245, 158, 11), "text": (254, 243, 199)},
        {"bg": (17, 24, 39), "card": (31, 41, 55), "accent": (14, 165, 233), "text": (243, 244, 246)}
    ]
    
    palette = color_palettes[scene_index % len(color_palettes)]

    img = Image.new('RGB', (width, height), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    # 1. Top Header Banner
    draw.rectangle([40, 60, 680, 140], fill=palette["card"], outline=palette["accent"], width=2)
    draw.text((width // 2, 100), f"SCENE {scene_index + 1}: {section_title.upper()}", fill=palette["accent"], anchor="mm")

    # 2. Main Content Card
    draw.rectangle([50, 200, 670, 600], fill=palette["card"], outline=(71, 85, 105), width=2)

    # Wrap sentence text into multi-line blocks for visual display
    words = text.split()
    lines = []
    line_length = 5
    for i in range(0, len(words), line_length):
        lines.append(" ".join(words[i:i + line_length]))

    y_offset = 300
    for line in lines[:5]:
        draw.text((width // 2, y_offset), line, fill=palette["text"], anchor="mm")
        y_offset += 55

    # 3. Footer Branding
    draw.line([(100, 1150), (620, 1150)], fill=palette["accent"], width=3)
    draw.text((width // 2, 1190), "FINANCIAL AUTOMATION ENGINE", fill=(148, 163, 184), anchor="mm")

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

        # 2. Render Multiple Scene Clips & Concatenate
        jobs[job_id]["progress"] = "Rendering Multiple Visual Clips..."

        try:
            from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips
        except (ImportError, AttributeError):
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            from moviepy.video.VideoClip import ImageClip
            from moviepy.video.compositing.concatenate import concatenate_videoclips

        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration

        # Split script into individual sentences for scene segments
        sentences = [s.strip() for s in script_text.split('.') if s.strip()]
        if not sentences:
            sentences = [script_text]

        scene_duration = total_duration / len(sentences)
        scene_titles = ["Introduction", "Market Data", "30-Day Range", "Volume Analysis", "Technical Support", "Future Outlook", "Summary"]

        clips = []
        for idx, sentence in enumerate(sentences):
            title = scene_titles[idx % len(scene_titles)]
            frame_array = make_scene_image(title, sentence, idx)

            try:
                clip = ImageClip(frame_array).set_duration(scene_duration)
            except AttributeError:
                clip = ImageClip(frame_array).with_duration(scene_duration)

            clips.append(clip)

        # Combine all individual scene clips into one sequence
        concat_video = concatenate_videoclips(clips, method="compose")
        
        try:
            final_clip = concat_video.set_audio(audio_clip)
        except AttributeError:
            final_clip = concat_video.with_audio(audio_clip)

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
