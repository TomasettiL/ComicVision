import json
import os
import shutil
import uuid
import threading
import subprocess
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from filters import FILTERS
from filters.base import overlay_layers

# ALLOW_VIDEO=1 python3 app.py
# ALLOW_VIDEO=0 python3 app.py
# http://127.0.0.1:5050

app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

jobs: dict = {}


def _make_video_writer(path: str, fps: float, width: int, height: int):
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError("No working video codec found — install ffmpeg or try a different format.")


def _find_ffmpeg() -> str | None:
    """Locate the ffmpeg binary, checking PATH and common install locations."""
    # 1. Check Python's own PATH (works when launched from a configured env)
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Common install locations (Homebrew, system)
    for candidate in [
        "/opt/homebrew/bin/ffmpeg",   # macOS Homebrew (Apple Silicon)
        "/usr/local/bin/ffmpeg",       # macOS Homebrew (Intel) / Linux
        "/usr/bin/ffmpeg",             # Linux system install
    ]:
        if os.path.isfile(candidate):
            return candidate

    # 3. Ask the login shell — picks up .zshrc / .bash_profile Homebrew PATH
    try:
        shell = os.environ.get("SHELL", "/bin/zsh")
        r = subprocess.run(
            [shell, "-l", "-c", "which ffmpeg"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            p = r.stdout.decode().strip()
            if p and os.path.isfile(p):
                return p
    except Exception:
        pass

    return None


def _mux_audio(video_path: str, audio_source: str) -> bool:
    """Copy the audio track from audio_source into video_path in-place."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        print("[audio] ffmpeg not found — skipping audio mux")
        return False

    print(f"[audio] ffmpeg: {ffmpeg}")
    print(f"[audio] video : {video_path}  ({os.path.getsize(video_path)} bytes)")
    print(f"[audio] source: {audio_source}")

    tmp = video_path + ".noaudio.mp4"
    try:
        os.rename(video_path, tmp)
        cmd = [
            ffmpeg, "-y", "-loglevel", "warning",
            "-i", tmp,
            "-i", audio_source,
            "-c:v", "copy",     # copy video stream — no re-encode
            "-c:a", "copy",     # copy audio stream — no re-encode
            "-map", "0:v:0",
            "-map", "1:a:0?",   # ? = silently skip if source has no audio
            "-shortest",
            video_path,
        ]
        print(f"[audio] cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        stderr = result.stderr.decode(errors="replace").strip()

        if result.returncode == 0:
            print(f"[audio] mux OK{f' — {stderr}' if stderr else ''}")
            os.remove(tmp)
            return True

        # Audio copy failed — try again with AAC re-encode (codec compatibility)
        print(f"[audio] copy failed (exit {result.returncode}): {stderr} — retrying with AAC encode")
        cmd[cmd.index("-c:a") + 1] = "aac"
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        stderr = result.stderr.decode(errors="replace").strip()

        if result.returncode == 0:
            print(f"[audio] mux OK (AAC){f' — {stderr}' if stderr else ''}")
            os.remove(tmp)
            return True

        print(f"[audio] mux failed (exit {result.returncode}): {stderr}")
        os.rename(tmp, video_path)
        return False

    except Exception as exc:
        print(f"[audio] mux error: {exc}")
        if os.path.exists(tmp) and not os.path.exists(video_path):
            os.rename(tmp, video_path)
        return False


def _coerce_params(filt, raw: dict) -> dict:
    out = {}
    for p in filt.params:
        val = raw.get(p["name"])
        if val is None:
            out[p["name"]] = p["default"]
        elif p["type"] == "range":
            out[p["name"]] = float(val)
        elif p["type"] == "select":
            out[p["name"]] = int(val)
        elif p["type"] == "checkbox":
            out[p["name"]] = val if isinstance(val, bool) else str(val).lower() in ("true", "1", "yes")
        else:
            out[p["name"]] = val
    return out


class _Compositor:
    """Two-stage compositor (colour chain → edge mask).

    Maintains per-layer EMA state so colour layers can be temporally smoothed
    across video frames.  For single images the state is never used.

    Stage 1 — Colour chain:
        Colour layers run sequentially; each receives the BGR output of the
        previous one.  After each layer, if temporal_smoothing > 0 the result
        is blended with the previous frame's output for that layer:
            out = (1-α) × current  +  α × previous
        This suppresses frame-to-frame flicker in mean shift region boundaries
        and k-means palette jumps without blurring the edges.

    Stage 2 — Edge mask:
        Edge layers all run on the *original* grayscale frame and are
        composited with overlay_layers (darken / cv2.min).  The mask is then
        multiplied onto the colour base.

    Returns BGR uint8.
    """

    def __init__(self, layer_configs: list):
        self.colour_cfgs = [lc for lc in layer_configs if FILTERS[lc["filter"]].output_type == "color"]
        self.edge_cfgs   = [lc for lc in layer_configs if FILTERS[lc["filter"]].output_type == "edge"]
        self._prev: dict = {}   # colour layer index → previous BGR frame

    def process(self, gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
        # ── Stage 1: colour base ──────────────────────────────────────────────
        if self.colour_cfgs:
            colour_base = bgr.copy()
            for i, lc in enumerate(self.colour_cfgs):
                current = FILTERS[lc["filter"]].process_frame(gray, bgr=colour_base, **lc["params"])
                alpha = float(lc["params"].get("temporal_smoothing", 0.0))
                if alpha > 0 and i in self._prev:
                    current = cv2.addWeighted(current, 1.0 - alpha, self._prev[i], alpha, 0)
                self._prev[i] = current
                colour_base = current
        else:
            colour_base = np.full((*gray.shape, 3), 255, dtype=np.uint8)

        # ── Stage 2: edge mask ────────────────────────────────────────────────
        if not self.edge_cfgs:
            return colour_base

        edge_frames = [
            FILTERS[lc["filter"]].process_frame(gray, bgr=bgr, **lc["params"])
            for lc in self.edge_cfgs
        ]
        edge_mask = overlay_layers(*edge_frames)
        mask_f = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        return (colour_base.astype(np.float32) * mask_f).clip(0, 255).astype(np.uint8)


def _apply_layers(gray: np.ndarray, bgr: np.ndarray, layer_configs: list) -> np.ndarray:
    """Stateless single-frame wrapper around _Compositor (used for images)."""
    return _Compositor(layer_configs).process(gray, bgr)


def _run_job(job_id: str, input_path: str, output_path: str,
             layer_configs: list, is_video: bool, convert_fps: bool = False):
    try:
        jobs[job_id]["status"] = "processing"

        if not is_video:
            bgr = cv2.imread(input_path)
            if bgr is None:
                raise ValueError("Could not read image file.")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            result = _apply_layers(gray, bgr, layer_configs)
            cv2.imwrite(output_path, result)
            jobs[job_id]["progress"] = 100

        else:
            TARGET_FPS = 24.0

            cap = cv2.VideoCapture(input_path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Only drop frames when source is meaningfully faster than target
            output_fps = TARGET_FPS if (convert_fps and fps > TARGET_FPS + 0.5) else fps

            writer = _make_video_writer(output_path, output_fps, width, height)
            compositor = _Compositor(layer_configs)

            source_frame  = 0
            next_out_time = 0.0   # seconds — when the next output frame is due

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = source_frame / fps
                if current_time >= next_out_time - 1e-9:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    writer.write(compositor.process(gray, frame))
                    next_out_time += 1.0 / output_fps

                source_frame += 1
                if total > 0:
                    jobs[job_id]["progress"] = int(source_frame / total * 100)

            cap.release()
            writer.release()
            _mux_audio(output_path, input_path)
            jobs[job_id]["progress"] = 100

        jobs[job_id]["status"] = "done"

    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/filters")
def get_filters():
    return jsonify({name: f.get_metadata() for name, f in FILTERS.items()})


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    ext = os.path.splitext(secure_filename(file.filename or ""))[1].lower()

    # Block video on public deployment unless ALLOW_VIDEO is set
    if ext in VIDEO_EXTS and not os.environ.get("ALLOW_VIDEO"):
        return jsonify({"error": "Video processing is not available in the public version."}), 403

    try:
        layer_configs = json.loads(request.form.get("layers", "[]"))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid layers data"}), 400

    if not layer_configs:
        return jsonify({"error": "No layers selected"}), 400

    for lc in layer_configs:
        fname = lc.get("filter", "")
        if fname not in FILTERS:
            return jsonify({"error": f"Unknown filter: '{fname}'"}), 400
        lc["params"] = _coerce_params(FILTERS[fname], lc.get("params", {}))

    filename  = secure_filename(file.filename or "upload")
    ext       = os.path.splitext(filename)[1].lower()
    is_video  = ext in VIDEO_EXTS
    job_id    = str(uuid.uuid4())
    input_path  = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    output_path = os.path.join(OUTPUT_FOLDER, f"{job_id}{'.mp4' if is_video else '.png'}")

    convert_fps = request.form.get("convert_fps", "false").lower() == "true"

    file.save(input_path)
    jobs[job_id] = {"status": "queued", "progress": 0, "is_video": is_video, "output_path": output_path}

    threading.Thread(
        target=_run_job,
        args=(job_id, input_path, output_path, layer_configs, is_video, convert_fps),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/result/<job_id>")
def get_result(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Not ready yet"}), 202
    return send_file(job["output_path"])


if __name__ == "__main__":
    print("ComicVision running at http://localhost:5050")
    app.run(debug=False, port=5050, threaded=True)
