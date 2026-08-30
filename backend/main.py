import asyncio
import os
import uuid
import edge_tts
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf

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
        ticker_str = data.ticker.upper()
        stock = yf.Ticker(ticker_str)
        history = stock.history(period="1mo")
        if history.empty:
            raise HTTPException(status_code=400, detail="Invalid ticker symbol")

        close_price = round(history["Close"].iloc[-1], 2)
        open_price = round(history["Open"].iloc[-1], 2)
        change = round(((close_price - open_price) / open_price) * 100, 2)
        high_price = round(history["High"].max(), 2)
        low_price = round(history["Low"].min(), 2)
        avg_volume = int(history["Volume"].mean())

        # Expanded narrative structure to dramatically increase duration
        script_sections = [
            f"Welcome back to the Daily Market Briefing. Today, we are taking an in-depth deep dive into {ticker_str}.",
            f"Looking at today's price action, {ticker_str} opened trading at ${open_price} and closed at ${close_price}, representing a net change of {change} percent.",
            f"Over the last 30 trading sessions, {ticker_str} reached a high of ${high_price} and hit a low of ${low_price}.",
            f"Trading activity remains strong, with average monthly volume sitting near {avg_volume:,} shares traded per session.",
            f"When examining broader market trends, institutional investors continue to monitor {ticker_str} closely due to shifting macroeconomic conditions, interest rate expectations, and sector momentum.",
            f"For long-term investors, key price levels to watch include the technical support zone near ${low_price} and resistance near ${high_price}.",
            f"Make sure to hit the subscribe button, leave a like, and comment below with your price target for {ticker_str}. Thank you for watching today's comprehensive analysis."
        ]
        
        full_script = " ".join(script_sections)
        return {"status": "success", "script": full_script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        # 2. Render Video with Visual Text Overlay
        jobs[job_id]["progress"] = "Rendering Video Media & Visuals..."

        from moviepy.editor import AudioFileClip, ColorClip, TextClip, CompositeVideoClip

        audio_clip = AudioFileClip(audio_path)
        
        # Base Background (Dark Theme)
        bg_clip = ColorClip(
            size=(720, 1280), color=(15, 23, 42), duration=audio_clip.duration
        )

        # Visual Text Overlay on Screen
        txt_clip = TextClip(
            "DAILY MARKET UPDATE\n\nAutomated Financial Report",
            fontsize=40,
            color='white',
            size=(600, 400),
            method='caption'
        ).set_position('center').set_duration(audio_clip.duration)

        # Combine Background and Text Overlay
        final_video = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio_clip)

        final_video.write_videofile(
            video_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio_clip.close()
        final_video.close()

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

    return {
        "status": "success",
        "message": f"Video '{data.title}' approved and queued for YouTube upload!",
    }
