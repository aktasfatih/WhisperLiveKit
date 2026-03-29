"""ONNX Runtime encoder with optional TensorRT acceleration.

Exports the Whisper encoder to ONNX on first use, then runs inference
via ONNX Runtime with the best available execution provider (TensorRT > CUDA > CPU).
"""
import logging
import os
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ort_available = False
try:
    import onnxruntime as ort
    _ort_available = True
except ImportError:
    pass


def ort_available() -> bool:
    return _ort_available


class OnnxEncoder:
    """Whisper encoder accelerated via ONNX Runtime (TensorRT/CUDA/CPU)."""

    def __init__(self, model_cache_dir: str = "/models/onnx"):
        self._session: Optional["ort.InferenceSession"] = None
        self._cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)

    def export_encoder(self, whisper_model, model_name: str) -> str:
        """Export Whisper encoder to ONNX if not already cached."""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        onnx_path = os.path.join(self._cache_dir, f"{safe_name}_encoder.onnx")

        if os.path.exists(onnx_path):
            logger.info(f"ONNX encoder already cached at {onnx_path}")
            return onnx_path

        logger.info(f"Exporting encoder to ONNX: {onnx_path}")
        encoder = whisper_model.encoder
        device = whisper_model.device
        n_mels = whisper_model.dims.n_mels
        n_audio_ctx = whisper_model.dims.n_audio_ctx  # 1500

        # Dummy input matching encoder's expected shape
        dummy_mel = torch.randn(1, n_mels, n_audio_ctx * 2, device=device, dtype=next(encoder.parameters()).dtype)

        encoder.eval()
        with torch.no_grad():
            torch.onnx.export(
                encoder,
                dummy_mel,
                onnx_path,
                opset_version=17,
                input_names=["mel"],
                output_names=["encoder_output"],
                dynamic_axes={
                    "mel": {0: "batch"},
                    "encoder_output": {0: "batch"},
                },
            )

        logger.info(f"Encoder exported to ONNX: {onnx_path}")
        return onnx_path

    def load(self, onnx_path: str) -> None:
        """Load ONNX encoder with best available execution provider."""
        if not _ort_available:
            raise ImportError("onnxruntime is required for ONNX encoder")

        providers = []
        provider_options = []

        # Try TensorRT first (best performance on NVIDIA GPUs)
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            trt_cache_dir = os.path.join(self._cache_dir, "trt_cache")
            os.makedirs(trt_cache_dir, exist_ok=True)
            providers.append("TensorrtExecutionProvider")
            provider_options.append({
                "trt_max_workspace_size": str(4 * 1024 * 1024 * 1024),  # 4GB
                "trt_fp16_enable": "1",
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": trt_cache_dir,
                "trt_timing_cache_enable": "1",
            })
            logger.info("Using TensorRT Execution Provider")

        # CUDA EP as fallback
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
            provider_options.append({
                "device_id": "0",
                "arena_extend_strategy": "kSameAsRequested",
            })
            if not providers or providers[0] != "TensorrtExecutionProvider":
                logger.info("Using CUDA Execution Provider")

        # CPU fallback
        providers.append("CPUExecutionProvider")
        provider_options.append({})

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        actual_provider = self._session.get_providers()[0]
        logger.info(f"ONNX encoder loaded with provider: {actual_provider}")

    def encode(self, mel: np.ndarray) -> np.ndarray:
        """Run encoder inference. Input/output are numpy arrays."""
        if self._session is None:
            raise RuntimeError("ONNX encoder not loaded. Call load() first.")

        result = self._session.run(
            ["encoder_output"],
            {"mel": mel},
        )
        return result[0]

    @property
    def is_loaded(self) -> bool:
        return self._session is not None
