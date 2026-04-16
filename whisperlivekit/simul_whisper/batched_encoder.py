"""Batched encoder scheduler for concurrent streaming sessions.

Instead of N sequential encoder passes (one per session), collects pending
requests and runs one batched forward pass through the encoder. This dramatically
improves GPU utilization and concurrent session throughput.

Usage:
    # At startup (once):
    scheduler = BatchedEncoderScheduler(fw_encoder, feature_extractor, device)
    scheduler.start()

    # Per session (in _encode_full):
    encoder_output, content_mel_len = await scheduler.encode(audio_segments)

    # At shutdown:
    scheduler.stop()
"""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EncodeRequest:
    """A pending encoder request from a session."""
    audio: torch.Tensor  # Raw audio waveform
    result: Optional[Tuple[torch.Tensor, int]] = None  # (encoder_output, content_mel_len)
    event: threading.Event = None  # Signaled when result is ready
    error: Optional[Exception] = None

    def __post_init__(self):
        if self.event is None:
            self.event = threading.Event()


class BatchedEncoderScheduler:
    """Collects encoder requests from sessions and runs batched forward passes.

    Sessions call encode() which blocks until the batch is processed.
    The scheduler collects requests for up to `batch_wait_ms` then runs
    one batched encoder pass for all pending requests.
    """

    def __init__(
        self,
        fw_encoder=None,
        feature_extractor=None,
        model_encoder=None,
        device: str = "cuda",
        n_mels: int = 128,
        batch_wait_ms: float = 50,
        max_batch_size: int = 8,
    ):
        self.fw_encoder = fw_encoder
        self.feature_extractor = feature_extractor
        self.model_encoder = model_encoder  # PyTorch encoder (fallback)
        self.device = device
        self.n_mels = n_mels
        self.batch_wait_ms = batch_wait_ms
        self.max_batch_size = max_batch_size

        self._pending: list[EncodeRequest] = []
        self._lock = threading.Lock()
        self._has_requests = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"BatchedEncoderScheduler started "
            f"(batch_wait={self.batch_wait_ms}ms, max_batch={self.max_batch_size})"
        )

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self._has_requests.set()  # Wake up the thread
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("BatchedEncoderScheduler stopped")

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Submit an encode request and block until result is ready.

        This is called from session threads (via asyncio.to_thread).
        Thread-safe.

        Args:
            audio: Raw audio waveform tensor

        Returns:
            (encoder_output, content_mel_len)
        """
        request = EncodeRequest(audio=audio)

        with self._lock:
            self._pending.append(request)
            self._has_requests.set()

        # Block until our result is ready
        request.event.wait()

        if request.error:
            raise request.error
        return request.result

    def _scheduler_loop(self):
        """Background thread that batches and processes encode requests."""
        from whisperlivekit.whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        if self.fw_encoder:
            from faster_whisper.audio import pad_or_trim as fw_pad_or_trim

        while self._running:
            # Wait for at least one request
            self._has_requests.wait(timeout=1.0)
            if not self._running:
                break

            # Short wait to accumulate more requests for batching
            time.sleep(self.batch_wait_ms / 1000.0)

            # Grab all pending requests
            with self._lock:
                if not self._pending:
                    self._has_requests.clear()
                    continue
                batch = self._pending[:self.max_batch_size]
                self._pending = self._pending[self.max_batch_size:]
                if not self._pending:
                    self._has_requests.clear()

            batch_size = len(batch)
            t0 = time.monotonic()

            try:
                if self.fw_encoder and self.feature_extractor:
                    self._process_batch_fw(batch, N_FRAMES, N_SAMPLES, fw_pad_or_trim)
                elif self.model_encoder:
                    self._process_batch_pytorch(
                        batch, N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim,
                    )
                else:
                    # Fallback: process sequentially
                    for req in batch:
                        self._process_single_fw(req, N_FRAMES, N_SAMPLES, fw_pad_or_trim)

                elapsed = (time.monotonic() - t0) * 1000
                logger.debug(
                    f"Batched encode: {batch_size} sessions in {elapsed:.0f}ms "
                    f"({elapsed / batch_size:.0f}ms/session)"
                )

            except Exception as e:
                logger.error(f"Batch encode failed: {e}")
                for req in batch:
                    req.error = e
                    req.event.set()

    def _process_batch_fw(self, batch, N_FRAMES, N_SAMPLES, fw_pad_or_trim):
        """Process a batch using faster-whisper encoder.

        CTranslate2's StorageView cannot be sliced back into per-sample
        results, so we encode each sample individually.  The scheduler's
        50ms collection window still provides fair round-robin scheduling
        across sessions and back-to-back GPU submission benefits from
        CTranslate2's internal worker queue (num_workers >= 2).
        """
        for req in batch:
            self._process_single_fw(req, N_FRAMES, N_SAMPLES, fw_pad_or_trim)

    def _process_batch_pytorch(self, batch, N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim):
        """Process a batch using PyTorch encoder."""
        mels = []
        content_mel_lens = []

        for req in batch:
            mel_padded = log_mel_spectrogram(
                req.audio, n_mels=self.n_mels,
                padding=N_SAMPLES, device=self.device,
            ).unsqueeze(0)
            mel = pad_or_trim(mel_padded, N_FRAMES)
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)
            content_mel_lens.append(content_mel_len)
            mels.append(mel)

        # Stack into batch
        mel_batch = torch.cat(mels, dim=0)  # (batch, n_mels, N_FRAMES)

        # Single batched encode call
        with torch.inference_mode():
            encoder_batch = self.model_encoder(mel_batch)

        # Distribute results
        for i, req in enumerate(batch):
            req.result = (encoder_batch[i:i + 1], content_mel_lens[i])
            req.event.set()

    def _process_single_fw(self, req, N_FRAMES, N_SAMPLES, fw_pad_or_trim):
        """Fallback: process a single request."""
        audio_length_seconds = len(req.audio) / 16000
        content_mel_len = int(audio_length_seconds * 100) // 2
        mel_padded = self.feature_extractor(
            waveform=req.audio.numpy(), padding=N_SAMPLES,
        )[None, :]
        mel = fw_pad_or_trim(mel_padded, N_FRAMES, axis=-1)
        encoder_feature = self.fw_encoder.encode(mel)
        try:
            feat = torch.as_tensor(encoder_feature, device=self.device)
        except TypeError:
            arr = np.asarray(encoder_feature, dtype=np.float32)
            feat = torch.as_tensor(arr, device=self.device)
        req.result = (feat, content_mel_len)
        req.event.set()
