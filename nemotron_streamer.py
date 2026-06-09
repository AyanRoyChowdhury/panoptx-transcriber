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


def _model_dir() -> Path:
    local = snapshot_download(
        HF_REPO,
        cache_dir=str(Path.home() / ".cache/huggingface/hub"),
    )
    return Path(local)


def _build_recognizer(d: Path) -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens  = str(d / "tokens.txt"),
        encoder = str(d / "encoder.int8.onnx"),
        decoder = str(d / "decoder.int8.onnx"),
        joiner  = str(d / "joiner.int8.onnx"),
        num_threads              = 4,
        sample_rate              = SR,
        feature_dim              = 80,
        enable_endpoint_detection= True,
        rule1_min_trailing_silence = 2.4,
        rule2_min_trailing_silence = 1.2,
        rule3_min_utterance_length = 300.0,
        decoding_method          = "greedy_search",
        provider                 = "cpu",
    )


class NemotronStreamer:
    """
    Streaming decoder wrapping sherpa-onnx's cache-aware RNNT engine.
    Interface matches ParakeetStreamer so server.py can use either.
    """

    def __init__(self):
        model_dir          = _model_dir()
        self._recognizer   = _build_recognizer(model_dir)
        self._stream       = self._recognizer.create_stream()
        self._committed_tx = ""

    # ── public API ────────────────────────────────────────────────────────────

    def add_audio(self, samples: np.ndarray) -> None:
        self._stream.accept_waveform(SR, samples.astype(np.float32))

    def infer(self) -> tuple[str, str]:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        current = self._recognizer.get_result(self._stream).strip()

        if self._recognizer.is_endpoint(self._stream):
            if current:
                self._committed_tx += current + " "
                self._recognizer.reset(self._stream)
                return current + " ", ""
            self._recognizer.reset(self._stream)
            return "", ""

        # Suppress single-token noise (e.g. "F", "Th") until at least one space
        # appears, indicating a complete word has been decoded.
        if current and " " not in current and len(current) < 4:
            return "", ""
        return "", current

    def flush(self) -> tuple[str, str]:
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        current = self._recognizer.get_result(self._stream).strip()
        if current:
            self._committed_tx += current
            return current, ""
        return "", ""

    def reset(self) -> None:
        self._stream       = self._recognizer.create_stream()
        self._committed_tx = ""

    def set_commit_margin(self, seconds: float) -> None:
        pass   # endpoint-based; not applicable
