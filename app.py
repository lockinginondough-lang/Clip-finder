import os, re, subprocess, tempfile, json
import streamlit as st
import cv2
import imageio_ffmpeg
from faster_whisper import WhisperModel

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

st.set_page_config(page_title="AI Clip Finder", page_icon="🎬")

HOOK_WORDS = [
    "secret", "never", "always", "because", "but", "truth", "actually", "crazy", "insane",
    "worst", "best", "biggest", "mistake", "nobody", "everyone", "stop", "wait", "honestly",
    "literally", "imagine", "turns out", "the reason", "here's why", "no one tells you",
]


@st.cache_resource
def get_model(size):
    return WhisperModel(size, device="cpu", compute_type="int8")


def score_text(text):
    t = text.lower()
    score = sum(2 for w in HOOK_WORDS if w in t)
    score += t.count("?") * 1.5
    score += t.count("!") * 1.0
    score += len(re.findall(r"\b\d+\b", t)) * 1.0
    return score


def srt_timestamp(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(words, start, end, path, chunk_size=6):
    lines, idx, chunk, chunk_start = [], 1, [], None
    for w in words:
        if start <= w.start <= end:
            if chunk_start is None:
                chunk_start = w.start
            chunk.append(w.word)
            if len(chunk) >= chunk_size:
                lines.append(
                    f"{idx}\n{srt_timestamp(chunk_start)} --> {srt_timestamp(w.end)}\n{' '.join(chunk).strip()}\n"
                )
                idx += 1
                chunk = []
                chunk_start = None
    if chunk:
        lines.append(f"{idx}\n{srt_timestamp(chunk_start)} --> {srt_timestamp(end)}\n{' '.join(chunk).strip()}\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def get_face_center_x(video_path, start, end, sample_fps=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frame_start = int(start * fps)
    frame_end = int(end * fps)
    interval = max(int(fps / sample_fps), 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    centers = []
    idx = frame_start
    while idx < frame_end:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - frame_start) % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                centers.append(x + w / 2)
        idx += 1
    cap.release()
    return sum(centers) / len(centers) if centers else None


st.title("🎬 AI Clip Finder")
st.write(
    "Paste a YouTube link, get vertical clips with burned-in captions, picked automatically "
    "from the highest-scoring moments in the video."
)

youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    num_clips = st.slider("Number of clips", 1, 8, 5)
with col2:
    clip_length = st.slider("Clip length (seconds)", 15, 90, 40, step=5)

col3, col4 = st.columns(2)
with col3:
    face_tracking = st.checkbox("Face tracking crop", value=True)
with col4:
    model_size = st.selectbox("Transcription quality (bigger = slower)", ["tiny", "base", "small"], index=0)

if st.button("Find & Cut Clips", type="primary"):
    if not youtube_url or "http" not in youtube_url:
        st.error("Please paste a valid YouTube URL.")
        st.stop()

    workdir = tempfile.mkdtemp(prefix="clipjob_")
    outdir = os.path.join(workdir, "clips")
    os.makedirs(outdir, exist_ok=True)
    video_path = os.path.join(workdir, "source.mp4")

    with st.status("Working on it...", expanded=True) as status:
        st.write("Downloading video...")
        cmd = [
            "yt-dlp",
            "--ffmpeg-location", FFMPEG,
            "--extractor-args", "youtube:player_client=android,web",
            "-f", "bv*[height<=1080]+ba/b[height<=1080]/best",
            "--merge-output-format", "mp4",
            "-o", video_path,
            youtube_url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        except subprocess.CalledProcessError as e:
            status.update(label="Download failed", state="error")
            st.error(e.stderr.decode(errors="ignore")[-500:])
            st.stop()
        except subprocess.TimeoutExpired:
            status.update(label="Timed out", state="error")
            st.error("Download timed out — try a shorter video.")
            st.stop()

        st.write("Transcribing audio (slowest step on free CPU hosting)...")
        audio_path = os.path.join(workdir, "audio.wav")
        subprocess.run(
            [FFMPEG, "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
            check=True, capture_output=True,
        )
        model = get_model(model_size)
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        words = []
        for seg in segments:
            if seg.words:
                words.extend(seg.words)

        if not words:
            status.update(label="No speech detected", state="error")
            st.error("Couldn't detect any speech in this video.")
            st.stop()

        st.write(f"Scoring {info.duration:.0f}s of transcript for highlight moments...")
        total_duration = words[-1].end
        candidates = []
        t = 0.0
        step = 8.0
        while t + clip_length <= total_duration:
            text = " ".join(w.word for w in words if t <= w.start <= t + clip_length)
            candidates.append((t, t + clip_length, score_text(text), text))
            t += step

        candidates.sort(key=lambda c: c[2], reverse=True)
        chosen = []
        for c in candidates:
            overlaps = any(c[1] > o[0] and c[0] < o[1] for o in chosen)
            if not overlaps:
                chosen.append(c)
            if len(chosen) >= num_clips:
                break
        chosen.sort(key=lambda c: c[0])

        if not chosen:
            status.update(label="Video too short", state="error")
            st.error("Video too short for the requested clip length.")
            st.stop()

        clip_paths = []
        for i, (start, end, sc, text) in enumerate(chosen):
            st.write(f"Cutting clip {i + 1}/{len(chosen)}...")
            raw_clip = os.path.join(outdir, f"clip_{i + 1}_raw.mp4")
            subprocess.run(
                [FFMPEG, "-y", "-ss", str(start), "-to", str(end), "-i", video_path, "-c", "copy", raw_clip],
                check=True, capture_output=True,
            )

            probe_cap = cv2.VideoCapture(raw_clip)
            w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe_cap.release()
            target_w = int(h * 9 / 16)

            center_x = get_face_center_x(video_path, start, end) if face_tracking else None
            if center_x is None:
                center_x = w / 2
            crop_x = int(max(0, min(w - target_w, center_x - target_w / 2)))

            srt_path = os.path.join(outdir, f"clip_{i + 1}.srt")
            write_srt(words, start, end, srt_path)

            final_clip = os.path.join(outdir, f"clip_{i + 1}.mp4")
            vf = (
                f"crop={target_w}:{h}:{crop_x}:0,"
                f"subtitles={srt_path}:force_style='FontSize=16,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3'"
            )
            subprocess.run(
                [FFMPEG, "-y", "-i", raw_clip, "-vf", vf, "-c:a", "copy", final_clip],
                check=True, capture_output=True,
            )
            os.remove(raw_clip)
            clip_paths.append((final_clip, sc, text))

        status.update(label=f"Done! {len(clip_paths)} clip(s) ready.", state="complete")

    st.success(f"{len(clip_paths)} clip(s) ready:")
    for i, (path, sc, text) in enumerate(clip_paths):
        st.write(f"**Clip {i + 1}** — score {sc:.1f} — \"{text.strip()[:100]}...\"")
        st.video(path)
        with open(path, "rb") as f:
            st.download_button(
                f"Download clip {i + 1}", f, file_name=os.path.basename(path),
                mime="video/mp4", key=f"dl_{i}",
            )
