import asyncio
import json
import logging
import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from streaming import ParakeetStreamer, STRIDE, SR as STREAM_SR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

SR = 16_000

# ── Model registry ─────────────────────────────────────────────────────────────
# Streamers are instantiated lazily on first WebSocket connection for each model.
# NemotronStreamer downloads ~1.3 GB on first use.

def _make_streamer(model_id: str):
    if model_id == "nemotron":
        from nemotron_streamer import NemotronStreamer
        return NemotronStreamer()
    if model_id == "nemo80":
        from nemotron_streamer import Nemo80Streamer
        return Nemo80Streamer()
    if model_id == "zipformer":
        from nemotron_streamer import ZipformerStreamer
        return ZipformerStreamer()
    return ParakeetStreamer()   # default: parakeet-tdt


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Parakeet-TDT-0.6B-v3 (streaming mode)…")
    warmup = ParakeetStreamer()
    warmup.add_audio(np.zeros(SR * 2, np.float32))
    warmup.infer()
    del warmup
    log.info("Parakeet model ready.  Nemotron will load on first use.")
    yield


app = FastAPI(lifespan=lifespan)


# ── Noise reduction (optional, per-connection) ────────────────────────────────

def _bandpass(audio: np.ndarray, lo: int, hi: int, state: dict) -> np.ndarray:
    """Stateful bandpass: carries filter state (zi) across chunks so 80 ms
    chunk boundaries don't produce filter transients (clicks)."""
    from scipy.signal import butter, sosfilt, sosfilt_zi
    if state.get("key") != (lo, hi):
        sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        state.update(key=(lo, hi), sos=sos, zi=sosfilt_zi(sos) * 0.0)
    out, state["zi"] = sosfilt(state["sos"], audio, zi=state["zi"])
    return out.astype(np.float32)


def _noise_reduce(audio: np.ndarray, strength: float) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=SR, stationary=True,
                           prop_decrease=float(np.clip(strength, 0.0, 1.0))).astype(np.float32)


def _preprocess(audio, cfg):
    if cfg["bp_enabled"]:
        audio = _bandpass(audio, cfg["bp_lo"], cfg["bp_hi"], cfg["_bp_state"])
    if cfg["nr_enabled"]:
        audio = _noise_reduce(audio, cfg["nr_strength"])
    return audio


# ── Per-connection config ─────────────────────────────────────────────────────

def _default_cfg():
    return {
        "nr_enabled":    False,
        "nr_strength":   0.75,
        "bp_enabled":    False,
        "bp_lo":         100,
        "bp_hi":         3800,
        "commit_margin": 1.2,   # seconds of right-context before committing
        "_bp_state":     {},    # bandpass filter state (zi) carried across chunks
    }


# ── YouTube helper ────────────────────────────────────────────────────────────

async def _transcribe_youtube(url: str, ws: WebSocket, streamer: ParakeetStreamer, cfg: dict):
    loop = asyncio.get_running_loop()
    log.info("YouTube: %s", url)

    ydl = await asyncio.create_subprocess_exec(
        "yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "--get-url", "--no-playlist", url,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await ydl.communicate()
    if ydl.returncode != 0:
        await ws.send_json({"type": "error", "message": f"yt-dlp: {stderr.decode()[:200]}"})
        return

    audio_url = stdout.decode().strip().splitlines()[0]
    ffmpeg = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", audio_url,
        "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )

    acc = np.empty(0, np.float32)
    READ_BYTES = STRIDE * 4   # read in stride-sized chunks (float32 = 4 bytes)
    leftover   = b""          # pipe reads aren't guaranteed multiples of 4 bytes
    stride_count = 0
    INFER_EVERY  = getattr(streamer, "infer_every", 4)   # per-model cadence

    try:
        while True:
            data = await ffmpeg.stdout.read(READ_BYTES)
            if not data:
                break
            raw = leftover + data
            cut = len(raw) - (len(raw) % 4)
            leftover = raw[cut:]
            if cut == 0:
                continue
            chunk = np.frombuffer(raw[:cut], np.float32).copy()
            if cfg["bp_enabled"] or cfg["nr_enabled"]:
                chunk = await loop.run_in_executor(None, _preprocess, chunk, cfg)
            acc = np.concatenate([acc, chunk])

            while len(acc) >= STRIDE:
                to_push, acc = acc[:STRIDE], acc[STRIDE:]
                streamer.add_audio(to_push)
                stride_count += 1

            if stride_count >= INFER_EVERY:
                stride_count = 0
                t0 = time.monotonic()
                committed, partial = await loop.run_in_executor(None, streamer.infer)
                inf_ms = int((time.monotonic() - t0) * 1000)
                await ws.send_json({
                        "type": "stream",
                        "committed": committed,
                        "partial":   partial,
                        "inference_ms": inf_ms,
                    })

        # Flush remainder
        if len(acc) > 0:
            streamer.add_audio(acc)
        committed, partial = await loop.run_in_executor(None, streamer.flush)
        if committed or partial:
            await ws.send_json({
                "type": "stream",
                "committed": committed,
                "partial":   "",
                "inference_ms": 0,
            })

    except asyncio.CancelledError:
        raise
    finally:
        if ffmpeg.returncode is None:
            ffmpeg.kill()
        await ffmpeg.wait()


# ── WebSocket handler ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    model: str = Query(default="parakeet", alias="model"),
):
    await ws.accept()

    model_id  = model if model in ("parakeet", "nemotron", "nemo80", "zipformer") else "parakeet"
    log.info("Client connected  model=%s", model_id)
    streamer  = _make_streamer(model_id)
    cfg       = _default_cfg()
    loop      = asyncio.get_running_loop()
    yt_task   = None

    # Accumulate audio between inference triggers
    audio_acc    = np.empty(0, np.float32)
    stride_count = 0  # strides added since last infer call

    # Infer every N strides instead of every stride.
    # Parakeet full-window inference (~120-200 ms on CPU) is slower than the
    # 80 ms stride interval, so it needs batching (N=4 → 320 ms cadence) or the
    # server falls behind real-time.  The sherpa streaming models decode a
    # 320 ms batch in ~30 ms, so they infer every stride (N=1 → 80 ms cadence,
    # ~110 ms word-to-screen latency).
    INFER_EVERY = getattr(streamer, "infer_every", 4)

    # (log already emitted above)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # ── control messages ──────────────────────────────────────────────
            if msg.get("text"):
                data = json.loads(msg["text"])

                if data.get("type") == "youtube":
                    if yt_task and not yt_task.done():
                        yt_task.cancel()
                    streamer.reset()
                    yt_task = asyncio.create_task(
                        _transcribe_youtube(data["url"], ws, streamer, cfg)
                    )

                elif data.get("type") == "stop":
                    if yt_task and not yt_task.done():
                        yt_task.cancel()
                        yt_task = None
                    # Push any sub-stride remainder before flushing so the
                    # last few words aren't dropped.
                    if len(audio_acc) > 0:
                        streamer.add_audio(audio_acc)
                    audio_acc    = np.empty(0, np.float32)
                    stride_count = 0
                    # Flush any buffered audio, always send to clear partial
                    committed, _ = await loop.run_in_executor(None, streamer.flush)
                    await ws.send_json({
                        "type": "stream", "committed": committed,
                        "partial": "", "inference_ms": 0,
                    })

                elif data.get("type") == "reset":
                    streamer.reset()
                    audio_acc    = np.empty(0, np.float32)
                    stride_count = 0

                elif data.get("type") == "config":
                    d = data
                    if "nr_enabled"     in d: cfg["nr_enabled"]     = bool(d["nr_enabled"])
                    if "nr_strength"    in d: cfg["nr_strength"]     = float(np.clip(d["nr_strength"], 0, 1))
                    if "bp_enabled"     in d: cfg["bp_enabled"]      = bool(d["bp_enabled"])
                    if "bp_lo"          in d: cfg["bp_lo"]           = int(max(20,  min(1000, d["bp_lo"])))
                    if "bp_hi"          in d: cfg["bp_hi"]           = int(max(500, min(7900, d["bp_hi"])))
                    if "commit_margin"  in d:
                        cfg["commit_margin"] = float(max(0.4, min(3.0, d["commit_margin"])))
                        streamer.set_commit_margin(cfg["commit_margin"])
                    await ws.send_json({
                        "type": "config_ack",
                        **{k: cfg[k] for k in cfg if not k.startswith("_")},
                    })

            # ── audio (mic / screen / file) ───────────────────────────────────
            elif msg.get("bytes"):
                chunk = np.frombuffer(msg["bytes"], np.float32).copy()

                preproc_ms = 0
                if cfg["bp_enabled"] or cfg["nr_enabled"]:
                    t0 = time.monotonic()
                    chunk = await loop.run_in_executor(None, _preprocess, chunk, cfg)
                    preproc_ms = int((time.monotonic() - t0) * 1000)

                audio_acc = np.concatenate([audio_acc, chunk])

                while len(audio_acc) >= STRIDE:
                    to_push, audio_acc = audio_acc[:STRIDE], audio_acc[STRIDE:]
                    streamer.add_audio(to_push)
                    stride_count += 1

                if stride_count >= INFER_EVERY:
                    stride_count = 0
                    t0 = time.monotonic()
                    committed, partial = await loop.run_in_executor(None, streamer.infer)
                    inf_ms = int((time.monotonic() - t0) * 1000)
                    await ws.send_json({
                        "type": "stream",
                        "committed": committed,
                        "partial":   partial,
                        "inference_ms": inf_ms,
                        "preproc_ms":   preproc_ms,
                    })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.exception("WebSocket error: %s", exc)
    finally:
        if yt_task and not yt_task.done():
            yt_task.cancel()
        log.info("Client disconnected")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
