"""
Streaming ASR via NVIDIA Nemotron-Speech-Streaming-0.6B.

Uses csukuangfj's sherpa-onnx INT8 ONNX export.  The encoder has built-in
cache support (Cache-Aware FastConformer), so each 80-ms chunk is decoded
with its previous hidden state — enabling true frame-by-frame streaming at
~80-300 ms latency.

First run downloads ~1.3 GB to ~/.cache/huggingface/hub/…
"""

import numpy as np
import sherpa_onnx
from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO  = "csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14"
SR       = 16_000
STRIDE_S = 0.08
STRIDE   = int(STRIDE_S * SR)   # 1280 samples per push


def _model_dir(repo: str = HF_REPO, allow_patterns=None) -> Path:
    local = snapshot_download(
        repo,
        cache_dir=str(Path.home() / ".cache/huggingface/hub"),
        allow_patterns=allow_patterns,
    )
    return Path(local)


def _build_recognizer(
    d: Path,
    encoder: str = "encoder.int8.onnx",
    decoder: str = "decoder.int8.onnx",
    joiner:  str = "joiner.int8.onnx",
    model_type: str = "",
) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens  = str(d / "tokens.txt"),
        encoder = str(d / encoder),
        decoder = str(d / decoder),
        joiner  = str(d / joiner),
        num_threads              = 4,
        sample_rate              = SR,
        feature_dim              = 80,
        # Endpoint detection OFF: the stream is never reset (see NemotronStreamer),
        # so a fired endpoint would latch permanently — is_endpoint() then returns
        # True on every call, force-committing word fragments and killing partials.
        enable_endpoint_detection= False,
        decoding_method          = "greedy_search",
        provider                 = "cpu",
        model_type               = model_type,
    )


class NemotronStreamer:
    """
    Streaming decoder wrapping sherpa-onnx's cache-aware RNNT engine.
    Interface matches ParakeetStreamer so server.py can use either.
    """

    # Cache-aware encoder decodes incrementally (~30 ms per call), so the
    # server can afford to infer on every 80 ms stride.
    infer_every = 1

    # Commit the trailing word once the decoder output has been unchanged for
    # this many consecutive infer() calls (~80 ms of audio each → ~1 s).
    _STALL_COMMIT_CALLS = 12

    def __init__(self):
        model_dir          = _model_dir()
        self._recognizer   = _build_recognizer(model_dir)
        self._stream       = self._recognizer.create_stream()
        self._committed_len   = 0   # chars of get_result() already committed
        self._stall_calls     = 0   # infer() calls with no new decoder output
        self._prev_result_len = 0

    # ── public API ────────────────────────────────────────────────────────────
    #
    # The stream is NEVER reset mid-session.  Greedy streaming RNNT never
    # revises already-emitted tokens, so committed text can simply be a
    # growing prefix of get_result().  Resetting (the old endpoint+reset
    # cycle) zeroed the encoder cache, which mangled the first words of every
    # new utterance and dropped lagging words at each endpoint.
    #
    # Each infer() commits everything up to the last complete word boundary;
    # the trailing (possibly still-growing) word is held back as the live
    # partial.  An endpoint (trailing silence) means the trailing word is
    # complete, so everything is committed.

    def add_audio(self, samples: np.ndarray) -> None:
        self._stream.accept_waveform(SR, samples.astype(np.float32))

    def infer(self) -> tuple[str, str]:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        current = self._recognizer.get_result(self._stream)

        if len(current) > self._prev_result_len:
            self._stall_calls = 0           # decoder is still producing
        else:
            self._stall_calls += 1
        self._prev_result_len = len(current)

        if self._stall_calls >= self._STALL_COMMIT_CALLS:
            cut = len(current)              # output stalled → trailing word is final
        else:
            cut = current.rfind(" ")        # hold back the in-progress word
            if cut < self._committed_len:
                cut = self._committed_len

        delta = ""
        if cut > self._committed_len:
            delta = current[self._committed_len:cut]
            self._committed_len = cut

        partial = current[self._committed_len:].strip()
        return delta, partial

    def flush(self) -> tuple[str, str]:
        # Tail padding: the cache-aware encoder needs ~0.5-0.8 s of right
        # context to decode the final words.  Without this the last words are
        # truncated (e.g. "incredible" -> "incred").
        self._stream.accept_waveform(SR, np.zeros(int(SR * 0.8), np.float32))
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        current = self._recognizer.get_result(self._stream)
        delta   = current[self._committed_len:]
        self._stream          = self._recognizer.create_stream()
        self._committed_len   = 0
        self._stall_calls     = 0
        self._prev_result_len = 0
        return delta, ""

    def reset(self) -> None:
        self._stream          = self._recognizer.create_stream()
        self._committed_len   = 0
        self._stall_calls     = 0
        self._prev_result_len = 0

    def set_commit_margin(self, seconds: float) -> None:
        pass   # commit is continuous word-by-word; margin not applicable


class Nemo80Streamer(NemotronStreamer):
    """
    NeMo cache-aware streaming FastConformer (80 ms lookahead), fp32.
    Smaller (~114M) but unquantized and noise-robust; ~200 ms/word latency.
    Output is lowercase without punctuation.
    """
    def __init__(self):
        d = _model_dir("csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms")
        self._recognizer    = _build_recognizer(
            d, encoder="encoder.onnx", decoder="decoder.onnx",
            joiner="joiner.onnx", model_type="nemo_transducer",
        )
        self._stream          = self._recognizer.create_stream()
        self._committed_len   = 0
        self._stall_calls     = 0
        self._prev_result_len = 0

    def set_commit_margin(self, seconds: float) -> None:
        pass   # endpoint-based; not applicable


class ZipformerStreamer(NemotronStreamer):
    """
    k2/icefall streaming Zipformer (int8, chunk-16 / left-64) — ~68 MB total.
    The smallest usable streaming transducer; runs in real time on a
    Raspberry Pi 4/5 (sherpa-onnx ships aarch64 wheels).  Lowercase, no
    punctuation, English only.
    """
    def __init__(self):
        d = _model_dir(
            "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26",
            allow_patterns=["*chunk-16-left-64.int8.onnx", "tokens.txt"],
        )
        self._recognizer = _build_recognizer(
            d,
            encoder="encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
            decoder="decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
            joiner="joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        )
        self._stream          = self._recognizer.create_stream()
        self._committed_len   = 0
        self._stall_calls     = 0
        self._prev_result_len = 0
