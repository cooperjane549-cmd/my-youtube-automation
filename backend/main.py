import asyncio
import io
import os
import uuid
import edge_tts
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

        sma_50 = round(history["Close"].rolling(window=50).mean().iloc[-1], 2)
        sma_200 = round(history["Close"].rolling(window=200).mean().iloc[-1], 2)

        ticker_sym = data.ticker.upper()

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
            f"That concludes our technical and fundamental breakdown for {ticker_sym}.",
            f"If you enjoyed this detailed report, hit subscribe, like the video, and drop your price targets in the comments below."
        ]

        script = " ".join(sections)
        return {"status": "success", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_chart_image(ticker_sym, history, width=480, height=280):
    """Generates a high-contrast line chart for visual technical representation."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')

    prices = history["Close"].tail(60).values
    ax.plot(prices, color='#10B981', linewidth=3.0)
    ax.fill_between(range(len(prices)), prices, min(prices), color='#10B981', alpha=0.25)

    ax.set_title(f"{ticker_sym} - 60-Day Technical Trajectory", color='#FFFFFF', fontsize=12, pad=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#64748B')
    ax.spines['bottom'].set_color('#64748B')
    ax.tick_params(colors='#CBD5E1', labelsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def get_fallback_font(size):
    """Safely loads default TTF fonts to ensure high-visibility custom text rendering."""
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


def render_rich_scene_frame(t, duration, sentences, stock_data, width=540, height=960):
    """Renders visual scenes synced with narration timing and active word chunks."""
    num_scenes = max(len(sentences), 1)
    scene_duration = duration / num_scenes
    scene_idx = min(int(t // scene_duration), num_scenes - 1)

    # Calculate intra-scene time progress to sync words with voice audio
    scene_progress_time = t % scene_duration if scene_duration > 0 else 0

    current_sentence = sentences[scene_idx]
    words = current_sentence.split()
    total_words = len(words)

    # Estimate active spoken words dynamic frame by frame
    words_per_sec = total_words / max(scene_duration, 0.1)
    active_word_idx = min(int(scene_progress_time * words_per_sec), total_words - 1)

    palette = {"bg": (15, 23, 42), "card": (30, 41, 59), "accent": (16, 185, 129), "text": (255, 255, 255)}

    img = Image.new('RGB', (width, height), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    font_title = get_fallback_font(20)
    font_body = get_fallback_font(24)
    font_small = get_fallback_font(14)

    # 1. Header Card
    draw.rectangle([20, 25, width - 20, 85], fill=palette["card"], outline=palette["accent"], width=2)
    draw.text((width // 2, 55), f"ANALYSIS SCENE {scene_idx + 1} / {num_scenes}", fill=(16, 185, 129), font=font_title, anchor="mm")

    # 2. Section 1: Visual Chart or Key Metrics Table
    if scene_idx % 2 == 0 and stock_data.get("chart_img") is not None:
        chart_img = stock_data["chart_img"]
        img.paste(chart_img, (30, 105))
        draw.rectangle([30, 105, 510, 385], outline=(100, 116, 139), width=2)
    else:
        draw.rectangle([30, 105, 510, 385], fill=palette["card"], outline=palette["accent"], width=2)
        draw.text((width // 2, 135), "KEY METRICS SUMMARY", fill=(251, 191, 36), font=font_title, anchor="mm")
        draw.line([(50, 155), (490, 155)], fill=(71, 85, 105), width=2)

        metrics = [
            ("Ticker Symbol:", stock_data.get("ticker", "STK")),
            ("Close Price:", f"${stock_data.get('close', 0)}"),
            ("Daily Change:", f"{stock_data.get('change', 0)}%"),
            ("52-Wk Range:", f"${stock_data.get('low', 0)} - ${stock_data.get('high', 0)}"),
            ("50-Day Moving Avg:", f"${stock_data.get('sma50', 0)}")
        ]

        y_table = 185
        for label, val in metrics:
            draw.text((60, y_table), label, fill=(203, 213, 225), font=font_small, anchor="lm")
            draw.text((450, y_table), str(val), fill=(255, 255, 255), font=font_small, anchor="rm")
            y_table += 38

    # 3. Section 2: Text Card - Audio Synchronized Subtitle Display (3-4 words centered)
    draw.rectangle([30, 405, width - 30, 820], fill=palette["card"], outline=(100, 116, 139), width=2)

    chunk_size = 3
    chunk_start = (active_word_idx // chunk_size) * chunk_size
    chunk_words = words[chunk_start:chunk_start + chunk_size]
    chunk_text = " ".join(chunk_words).upper()

    # Highlight active synchronized chunk in bright accent color
    draw.text((width // 2, 610), chunk_text, fill=(251, 191, 36), font=font_body, anchor="mm")

    # 4. Progress Bar
    progress = int((t / max(duration, 1)) * (width - 60))
    draw.rectangle([30, 840, 30 + progress, 850], fill=palette["accent"])

    # 5. Footer Branding
    draw.line([(50, 885), (width - 50, 885)], fill=palette["accent"], width=2)
    draw.text((width // 2, 915), "MAGIC TORTOISE FINANCIAL ENGINE", fill=(148, 163, 184), font=font_small, anchor="mm")

    return np.array(img)


def run_video_pipeline(job_id: str, script_text: str, ticker: str):
    try:
        jobs[job_id] = {"status": "processing", "progress": "Fetching Market Data & Generating Audio..."}

        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")

        stock_data = {
            "ticker": ticker.upper(),
            "close": round(history["Close"].iloc[-1], 2) if not history.empty else 0,
            "change": round(((history["Close"].iloc[-1] - history["Open"].iloc[-1]) / history["Open"].iloc[-1]) * 100, 2) if not history.empty else 0,
            "low": round(history["Low"].min(), 2) if not history.empty else 0,
            "high": round(history["High"].max(), 2) if not history.empty else 0,
            "sma50": round(history["Close"].rolling(window=50).mean().iloc[-1], 2) if not history.empty else 0,
            "chart_img": generate_chart_image(ticker.upper(), history) if not history.empty else None
        }

        audio_filename = f"audio_{job_id}.mp3"
        video_filename = f"video_{job_id}.mp4"
        audio_path = os.path.join(OS_MEDIA_DIR, audio_filename)
        video_path = os.path.join(OS_MEDIA_DIR, video_filename)

        # 1. Voiceover Audio Generation
        async def make_audio():
            communicate = edge_tts.Communicate(script_text, "en-US-ChristopherNeural")
            await communicate.save(audio_path)

        asyncio.run(make_audio())

        # 2. Render Video with Exact Rhythm Matching
        jobs[job_id]["progress"] = "Rendering Animated Charts & Dynamic Text..."

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
            return render_rich_scene_frame(t, duration, sentences, stock_data)

        try:
            video_clip = VideoClip(frame_generator, duration=duration).set_audio(audio_clip)
        except AttributeError:
            video_clip = VideoClip(frame_generator, duration=duration).with_audio(audio_clip)

        video_clip.write_videofile(
            video_path, fps=12, codec="libx264", audio_codec="aac", verbose=False, logger=None
        )

        audio_clip.close()
        video_clip.close()

        # 3. Mark Completed
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

    words = data.script_text.split()
    ticker = "AAPL"
    for w in words:
        if w.isupper() and len(w) <= 5 and w.isalpha():
            ticker = w
            break

    background_tasks.add_task(run_video_pipeline, job_id, data.script_text, ticker)
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
