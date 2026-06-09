"""
Streaming TDT decoder for Parakeet-TDT-0.6B-v3.

Architecture
------------
The encoder uses full self-attention (no cache), so true frame-by-frame streaming
isn't possible.  Instead we use a sliding-window approach:

  • Audio is accumulated in a rolling buffer (up to MAX_WIN_S seconds).
  • Every STRIDE_S seconds we run a full greedy decode on the current window.
  • Only tokens whose encoder frame is at least COMMIT_MARGIN_S before the
    end of the window are committed (final).  Tokens after that boundary are
    shown as live "partial" text and replaced on the next pass.

The margin ensures that when a token is committed, the encoder has already
seen enough right-context to settle on a stable prediction.  Without a
margin, tokens near the right edge of the window are decoded with incomplete
context and frequently change on the next pass, causing word fragments.

Latency
-------
  committed_latency ≈ COMMIT_MARGIN_S + inference_time
  With COMMIT_MARGIN_S=1.2 s and inference ~120 ms on Apple Silicon:
    ≈ 1.2 s + 0.12 s  ≈ 1.32 s  above the end of the last committed word.

  This is a ~2× improvement over the original 3-second chunk approach, with
  significantly better accuracy than the cross-pass-stability approach
  (which committed tokens before the encoder had enough right-context).
"""

import re
import numpy as np
import onnxruntime as ort
from pathlib import Path
from importlib.resources import as_file, files

import onnx_asr.preprocessors  # only for the bundled fbanks.npz

# ── Model paths ────────────────────────────────────────────────────────────────
_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--istupakov--parakeet-tdt-0.6b-v3-onnx"
    / "snapshots/8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
)

# ── Feature-extraction constants ───────────────────────────────────────────────
SR           = 16_000
HOP          = 160          # 10 ms  (samples)
WIN_LEN      = 400          # 25 ms  (samples)
N_FFT        = 512
PREEMPH      = 0.97
LOG_GUARD    = float(2 ** -24)
N_MEL        = 128

# ── Model constants ────────────────────────────────────────────────────────────
BLANK_ID       = 8192
VOCAB_SIZE     = 8193       # token IDs 0..8192 (8192 = blank)
N_DUR          = 5          # TDT duration classes
MAX_TOKS_STEP  = 10         # greedy: max non-blank tokens per encoder frame

# ── Encoder subsampling (measured: enc_frames = (mel_frames-1)//8 + 1) ────────
# Conveniently: 1 encoder frame ≈ 8 mel frames = 8×HOP = 1280 samples = 80 ms
# so ENCODER_STRIDE_SAMPLES == STRIDE.
ENCODER_MEL_STRIDE = 8

# ── Streaming hyper-parameters ─────────────────────────────────────────────────
STRIDE_S        = 0.08        # push new audio every 80 ms
STRIDE          = int(STRIDE_S * SR)    # 1280 samples

MAX_WIN_S       = 6.0         # rolling audio buffer (seconds)
MAX_WIN         = int(MAX_WIN_S * SR)   # 96000 samples

# Tokens within COMMIT_MARGIN_S of the window edge lack right-context and may
# flip on the next pass.  Only commit tokens further back than this boundary.
# Higher margin → more accurate, higher latency.
# Lower  margin → faster, more word-fragments.
# Default 1.2 s; configurable at runtime via ParakeetStreamer.set_commit_margin().
DEFAULT_COMMIT_MARGIN_S = 1.2

# Don't commit until the buffer holds COMMIT_MARGIN_S + this much extra audio.
_MIN_EXTRA_S = 0.4

# ── Text-cleaning regex ────────────────────────────────────────────────────────
_SPACE_RE = re.compile(r"\A\s|\s\B|(\s)\b")

def _toks_to_text(vocab: dict[int, str], token_ids: list[int]) -> str:
    raw = "".join(vocab.get(i, "") for i in token_ids)
    return _SPACE_RE.sub(lambda m: " " if m.group(1) else "", raw).strip()


class ParakeetStreamer:
    """Stateful streaming decoder.  One instance per WebSocket connection."""

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(self):
        # Mel filterbank
        with (
            as_file(files(onnx_asr.preprocessors).joinpath("data/fbanks.npz")) as fp,
            np.load(fp) as data,
        ):
            self._mel_fb = data["nemo128"].astype(np.float32)  # [257, 128]

        # ONNX sessions
        opts = ["CPUExecutionProvider"]
        self._enc  = ort.InferenceSession(str(_MODEL_DIR / "encoder-model.onnx"),       providers=opts)
        self._decj = ort.InferenceSession(str(_MODEL_DIR / "decoder_joint-model.onnx"), providers=opts)

        # Vocabulary  (▁ → space)
        self._vocab: dict[int, str] = {}
        with (_MODEL_DIR / "vocab.txt").open() as f:
            for line in f:
                tok, idx = line.rstrip("\n").rsplit(" ", 1)
                self._vocab[int(idx)] = tok.replace("▁", " ")

        # Initial LSTM state [2, 1, 640]
        sh = {x.name: x.shape for x in self._decj.get_inputs()}
        self._state0 = (
            np.zeros((sh["input_states_1"][0], 1, sh["input_states_1"][2]), np.float32),
            np.zeros((sh["input_states_2"][0], 1, sh["input_states_2"][2]), np.float32),
        )

        # Streaming state
        self._win:                np.ndarray = np.empty(0, np.float32)
        self._committed_enc_frame: int       = -1  # window-relative enc frame of last committed token
        self._committed_txt:      str        = ""
        self._commit_margin_s:    float      = DEFAULT_COMMIT_MARGIN_S

    # ── public API ────────────────────────────────────────────────────────────

    def set_commit_margin(self, seconds: float) -> None:
        """Adjust the right-context margin (0.4–3.0 s).  Higher = more accurate."""
        self._commit_margin_s = float(max(0.4, min(3.0, seconds)))

    def add_audio(self, samples: np.ndarray) -> None:
        self._win = np.concatenate([self._win, samples])
        if len(self._win) > MAX_WIN:
            excess = len(self._win) - MAX_WIN
            self._win = self._win[excess:]
            # Shift the committed-frame pointer left to account for removed audio.
            # 1 encoder frame = STRIDE samples, so trim by the same number of frames.
            self._committed_enc_frame = max(-1, self._committed_enc_frame - excess // STRIDE)

    def infer(self) -> tuple[str, str]:
        """
        Run one inference pass on the current window.
        Returns (committed_delta, partial_text).
        committed_delta: newly committed text (append to transcript).
        partial_text:    live preview of uncommitted tokens (replace each call).
        """
        if len(self._win) < HOP * 8:
            return "", ""

        feats, feat_len = self._extract_features(self._win)
        enc_out         = self._run_encoder(feats, feat_len)
        tokens, frames  = self._greedy_decode(enc_out)

        min_win = int((self._commit_margin_s + _MIN_EXTRA_S) * SR)
        if len(self._win) < min_win:
            partial = _toks_to_text(self._vocab, tokens)
            return "", partial

        return self._update_commitment(tokens, frames, enc_out.shape[0])

    def flush(self) -> tuple[str, str]:
        """Commit everything remaining (call when audio source stops)."""
        if len(self._win) < HOP * 8:
            return "", ""
        feats, feat_len = self._extract_features(self._win)
        enc_out         = self._run_encoder(feats, feat_len)
        tokens, _       = self._greedy_decode(enc_out)

        full_text = _toks_to_text(self._vocab, tokens)
        delta = full_text[len(self._committed_txt):]
        self._committed_txt       = full_text
        self._committed_enc_frame = enc_out.shape[0] - 1
        return delta, ""

    def reset(self) -> None:
        self._win                 = np.empty(0, np.float32)
        self._committed_enc_frame = -1
        self._committed_txt       = ""

    # ── feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """128-bin log-mel matching NemoPreprocessorNumpy.  Returns ([1,128,T], feat_len)."""
        a = audio - PREEMPH * np.pad(audio, (1, 0))[:-1]
        a = np.pad(a.astype(np.float32), (N_FFT // 2, N_FFT // 2))
        frames = np.lib.stride_tricks.sliding_window_view(a, N_FFT)[::HOP]

        win = np.pad(
            np.hanning(WIN_LEN).astype(np.float32),
            ((N_FFT - WIN_LEN) // 2, (N_FFT - WIN_LEN) // 2),
        )
        spec = np.abs(
            np.fft.rfft((frames * win).astype(np.float64), N_FFT)
        ).astype(np.float32) ** 2   # [T, 257]

        mel     = np.maximum(np.matmul(spec, self._mel_fb), 0.0)
        log_mel = np.log(mel + LOG_GUARD)  # [T, 128]

        # Normalise over the full window (consistent with NeMo batch mode).
        # For streaming, normalization is slightly non-stationary as audio grows,
        # but the commit-margin ensures early tokens have already been committed
        # before they'd be affected by large normalization shifts.
        N    = log_mel.shape[0]
        mean = log_mel.sum(0) / N
        var  = ((log_mel - mean) ** 2).sum(0) / max(N - 1, 1)
        log_mel = ((log_mel - mean) / (np.sqrt(var) + 1e-5)).astype(np.float32)

        feat_len = len(audio) // HOP
        return log_mel.T[np.newaxis], feat_len   # [1, 128, T]

    # ── encoder ───────────────────────────────────────────────────────────────

    def _run_encoder(self, feats: np.ndarray, feat_len: int) -> np.ndarray:
        """Returns [T_enc, 1024]."""
        enc, _ = self._enc.run(
            ["outputs", "encoded_lengths"],
            {"audio_signal": feats, "length": np.array([feat_len], np.int64)},
        )
        return enc[0].T   # [T_enc, 1024]

    # ── greedy TDT decoder ────────────────────────────────────────────────────

    def _decode_step(self, frame, last_tok, state):
        out, s1, s2 = self._decj.run(
            ["outputs", "output_states_1", "output_states_2"],
            {
                "encoder_outputs": frame[np.newaxis, :, np.newaxis],
                "targets":         np.array([[last_tok]], np.int32),
                "target_length":   np.array([1],         np.int32),
                "input_states_1":  state[0],
                "input_states_2":  state[1],
            },
        )
        logits   = out.squeeze()
        token_id = int(logits[:VOCAB_SIZE].argmax())
        duration = int(logits[VOCAB_SIZE:].argmax())
        return token_id, duration, (s1, s2)

    def _greedy_decode(self, enc_out: np.ndarray) -> tuple[list[int], list[int]]:
        """
        Full greedy TDT decode.
        Returns (token_ids, frame_idxs) — frame_idxs[i] is the encoder frame
        that emitted token_ids[i].  Used by _update_commitment to determine
        which tokens are safe to commit (far enough from the right edge).
        """
        T      = enc_out.shape[0]
        tokens: list[int] = []
        frames: list[int] = []   # encoder frame index per token
        state  = self._state0
        t      = 0
        emitted = 0
        last_tok = BLANK_ID

        while t < T:
            token_id, duration, new_state = self._decode_step(enc_out[t], last_tok, state)

            if token_id != BLANK_ID:
                state    = new_state
                last_tok = token_id
                tokens.append(token_id)
                frames.append(t)   # record which encoder frame produced this token
                emitted += 1

            if duration > 0:
                t       += duration
                emitted  = 0
            elif token_id == BLANK_ID or emitted >= MAX_TOKS_STEP:
                t       += 1
                emitted  = 0

        return tokens, frames

    # ── commitment logic ──────────────────────────────────────────────────────

    def _update_commitment(
        self,
        curr_tokens: list[int],
        curr_frames: list[int],
        total_enc_frames: int,
    ) -> tuple[str, str]:
        """
        Commit tokens that are safely before the right-context margin.

        Uses encoder-frame position as the watermark (not token count) so that
        trimming the rolling audio buffer never causes the watermark to exceed
        the number of tokens the trimmed window can produce.

        A token at encoder frame F is committed when:
          F < total_enc_frames - COMMIT_MARGIN_ENC
          F > _committed_enc_frame  (not already committed)

        Commits only advance to the last COMPLETE WORD boundary.
        """
        commit_margin_enc = int(self._commit_margin_s / STRIDE_S)
        commit_boundary   = total_enc_frames - commit_margin_enc

        safe_end         = 0
        last_safe_frame  = self._committed_enc_frame

        for i in range(len(curr_tokens)):
            if curr_frames[i] >= commit_boundary:
                break                               # too close to right edge
            if curr_frames[i] <= self._committed_enc_frame:
                continue                            # already committed in a prior pass
            next_i = i + 1
            at_boundary = (
                next_i >= len(curr_tokens)
                or self._vocab.get(curr_tokens[next_i], "").startswith(" ")
            )
            if at_boundary:
                safe_end        = next_i
                last_safe_frame = curr_frames[i]

        if safe_end > 0:
            full_committed = _toks_to_text(self._vocab, curr_tokens[:safe_end])
            if len(full_committed) > len(self._committed_txt):
                delta = full_committed[len(self._committed_txt):]
                self._committed_txt = full_committed
            else:
                delta = ""
            self._committed_enc_frame = last_safe_frame
        else:
            delta = ""

        # Partial: tokens whose encoder frame is past the committed frontier
        partial_toks = [
            t for t, f in zip(curr_tokens, curr_frames)
            if f > self._committed_enc_frame
        ]
        partial = _toks_to_text(self._vocab, partial_toks) if partial_toks else ""

        return delta, partial
