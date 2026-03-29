import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from whisperlivekit.backend_support import faster_backend_available, mlx_backend_available
from whisperlivekit.whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES, TOKENS_PER_SECOND, log_mel_spectrogram, pad_or_trim
from whisperlivekit.whisper.decoding import BeamSearchDecoder, GreedyDecoder, SuppressTokens
from whisperlivekit.whisper.timing import median_filter

from .align_att_base import DEC_PAD, AlignAttBase
from .beam import BeamPyTorchInference
from .config import AlignAttConfig
from .decoder_state import DecoderState
from .eow_detection import fire_at_boundary, load_cif
from .token_buffer import TokenBuffer

logger = logging.getLogger(__name__)

if mlx_backend_available():
    from mlx_whisper.audio import log_mel_spectrogram as mlx_log_mel_spectrogram
    from mlx_whisper.transcribe import pad_or_trim as mlx_pad_or_trim

if faster_backend_available():
    from faster_whisper.audio import pad_or_trim as fw_pad_or_trim
    from faster_whisper.feature_extractor import FeatureExtractor

USE_MLCORE = False


def load_coreml_encoder():
    try:
        from coremltools.models import MLModel
    except ImportError:
        logger.warning("coremltools is not installed")
        return None
    COREML_ENCODER_PATH = os.environ.get(
        "MLCORE_ENCODER_PATH",
        "whisperlivekit/whisper/whisper_encoder.mlpackage",
    )
    _coreml_encoder = MLModel(COREML_ENCODER_PATH)
    spec = _coreml_encoder.get_spec()
    _coreml_input_name = spec.description.input[0].name if spec.description.input else "mel"
    _coreml_output_name = spec.description.output[0].name if spec.description.output else None
    return _coreml_encoder, _coreml_input_name, _coreml_output_name


class AlignAtt(AlignAttBase):
    """
    PyTorch Alignment-based Attention decoder for SimulStreaming.

    Hookless — the model can be shared across multiple sessions,
    with each session maintaining its own DecoderState.
    """

    def __init__(
        self,
        cfg: AlignAttConfig,
        loaded_model=None,
        mlx_encoder=None,
        fw_encoder=None,
    ) -> None:
        self.mlx_encoder = mlx_encoder
        self.fw_encoder = fw_encoder
        if fw_encoder:
            self.fw_feature_extractor = FeatureExtractor(
                feature_size=loaded_model.dims.n_mels,
            )
        self.coreml_encoder_tuple = None
        if USE_MLCORE:
            self.coreml_encoder_tuple = load_coreml_encoder()
        self.use_mlcore = self.coreml_encoder_tuple is not None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Common init (sets self.model, self.cfg, decode_options, etc.)
        self._base_init(cfg, loaded_model)
        logger.info(f"Model dimensions: {self.model.dims}")

        # Per-session state
        self.state = DecoderState()
        self._init_state(cfg)

    def _init_state(self, cfg: AlignAttConfig):
        self._init_state_common(cfg)

        # CIF helpers for end-of-word boundary detection
        self.state.CIFLinear, self.state.always_fire, self.state.never_fire = load_cif(
            cfg, n_audio_state=self.model.dims.n_audio_state, device=self.model.device,
        )

        # Build alignment source mapping
        self.state.align_source = {}
        self.state.num_align_heads = 0
        for layer_rank, head_id in self.model.alignment_heads.indices().T:
            layer_rank = layer_rank.item()
            heads = self.state.align_source.get(layer_rank, [])
            heads.append((self.state.num_align_heads, head_id.item()))
            self.state.align_source[layer_rank] = heads
            self.state.num_align_heads += 1

        # Build suppress tokens function
        suppress_tokens = [
            self.tokenizer.transcribe, self.tokenizer.translate,
            self.tokenizer.sot, self.tokenizer.sot_prev,
            self.tokenizer.sot_lm, self.tokenizer.no_timestamps,
        ] + list(self.tokenizer.all_language_tokens)
        if self.tokenizer.no_speech is not None:
            suppress_tokens.append(self.tokenizer.no_speech)
        suppress_tokens = tuple(sorted(set(suppress_tokens)))
        logger.debug(f"Suppress tokens: {suppress_tokens}")
        sup_tokens = SuppressTokens(suppress_tokens)
        self.state.suppress_tokens_fn = lambda logits: sup_tokens.apply(logits, None)

        self.init_tokens()
        self.init_context()

        # Decoder type
        self.state.decoder_type = cfg.decoder_type
        if cfg.decoder_type == "greedy":
            logger.info("Using greedy decoder")
            self.state.token_decoder = GreedyDecoder(0.0, self.tokenizer.eot)
        elif cfg.decoder_type == "beam":
            logger.info("Using beam decoder")
            self.state.inference = BeamPyTorchInference(
                self.model, self.state.initial_token_length,
            )
            self.state.inference.kv_cache = self.state.kv_cache
            self.state.token_decoder = BeamSearchDecoder(
                inference=self.state.inference,
                eot=self.tokenizer.eot,
                beam_size=cfg.beam_size,
            )

    # === Abstract method implementations ===

    def init_tokens(self):
        logger.debug(f"init tokens, {len(self.state.segments)}")
        self.state.initial_tokens = torch.tensor(
            self.tokenizer.sot_sequence_including_notimestamps,
            dtype=torch.long, device=self.model.device,
        ).unsqueeze(0)
        self.state.initial_token_length = self.state.initial_tokens.shape[1]
        self.state.sot_index = self.tokenizer.sot_sequence.index(self.tokenizer.sot)
        logger.debug(f"init tokens after, {len(self.state.segments)}")
        self.state.tokens = [self.state.initial_tokens]

    def init_context(self):
        kw = {
            'tokenizer': self.tokenizer,
            'device': self.model.device,
            'prefix_token_ids': [self.tokenizer.sot_prev],
        }
        self.state.context = TokenBuffer.empty(**kw)
        if self.cfg.static_init_prompt is not None:
            self.state.context = TokenBuffer.from_text(self.cfg.static_init_prompt, **kw)
        if self.cfg.init_prompt is not None:
            self.state.context.text += self.cfg.init_prompt

    def insert_audio(self, segment=None):
        if segment is not None:
            self.state.segments.append(segment)
        removed_len = 0
        segments_len = self.segments_len()
        while len(self.state.segments) > 1 and segments_len > self.cfg.audio_max_len:
            removed_len = self.state.segments[0].shape[0] / 16000
            segments_len -= removed_len
            self.state.last_attend_frame -= int(TOKENS_PER_SECOND * removed_len)
            self.state.cumulative_time_offset += removed_len
            self.state.segments = self.state.segments[1:]
            # Segments removed — encoder cache is stale
            self.state.invalidate_encoder_cache()
            logger.debug(
                f"remove segments: {len(self.state.segments)} {len(self.state.tokens)}, "
                f"cumulative offset: {self.state.cumulative_time_offset:.2f}s"
            )
            if len(self.state.tokens) > 1:
                self.state.context.append_token_ids(self.state.tokens[1][0, :].tolist())
                self.state.tokens = [self.state.initial_tokens] + self.state.tokens[2:]
        return removed_len

    def _current_tokens(self):
        toks = self.state.tokens
        if toks[0].shape[0] == 1:
            toks[0] = toks[0].repeat_interleave(self.cfg.beam_size, dim=0)
        if not self.state.context.is_empty():
            context_toks = self.state.context.as_tensor_beam(
                self.cfg.beam_size, device=self.model.device,
            )
            toks = [context_toks] + toks
        if len(toks) > 1:
            current_tokens = torch.cat(toks, dim=1)
        else:
            current_tokens = toks[0]
        logger.debug("debug print current_tokens:")
        self.debug_print_tokens(current_tokens)
        return current_tokens

    def fire_at_boundary(self, chunked_encoder_feature: torch.Tensor):
        if self.state.always_fire:
            return True
        if self.state.never_fire:
            return False
        return fire_at_boundary(chunked_encoder_feature, self.state.CIFLinear)

    @torch.no_grad()
    def lang_id(self, encoder_features):
        n_audio = encoder_features.shape[0]
        x = torch.tensor([[self.tokenizer.sot]] * n_audio).to(self.model.device)
        logits = self.model.logits(x, encoder_features)[:, 0]

        mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        mask[list(self.tokenizer.all_language_tokens)] = False
        logits[:, mask] = -np.inf
        language_tokens = logits.argmax(dim=-1)
        language_token_probs = logits.softmax(dim=-1).cpu()
        language_probs = [
            {
                c: language_token_probs[i, j].item()
                for j, c in zip(
                    self.tokenizer.all_language_tokens,
                    self.tokenizer.all_language_codes,
                )
            }
            for i in range(n_audio)
        ]
        single = encoder_features.ndim == 2
        if single:
            language_tokens = language_tokens[0]
            language_probs = language_probs[0]
        self._clean_cache()
        return language_tokens, language_probs

    def _concat_segments(self):
        if len(self.state.segments) > 1:
            return torch.cat(self.state.segments, dim=0)
        return self.state.segments[0]

    def _encode(self, input_segments):
        # Check encoder output cache: if audio hasn't grown enough since last
        # full encode, reuse the cached encoder output. The decoder cross-attention
        # sees slightly stale features for the newest ~2s of audio, but alignment
        # heads handle this gracefully via the rewind mechanism.
        current_audio_len = len(input_segments) / 16000.0
        cache_threshold = self.cfg.encoder_cache_threshold_s
        num_segments = len(self.state.segments)

        if (
            cache_threshold > 0
            and self.state.encoder_cache_output is not None
            and num_segments == self.state.encoder_cache_num_segments
            and current_audio_len - self.state.encoder_cache_audio_len < cache_threshold
        ):
            logger.debug(
                f"Encoder cache hit: {current_audio_len:.2f}s audio, "
                f"cached at {self.state.encoder_cache_audio_len:.2f}s "
                f"(threshold={cache_threshold}s)"
            )
            return self.state.encoder_cache_output, self.state.encoder_cache_mel_len

        # Cache miss — run full encode
        encoder_feature, content_mel_len = self._encode_full(input_segments)

        # Update cache
        if cache_threshold > 0:
            self.state.encoder_cache_output = encoder_feature
            self.state.encoder_cache_audio_len = current_audio_len
            self.state.encoder_cache_mel_len = content_mel_len
            self.state.encoder_cache_num_segments = num_segments
            logger.debug(
                f"Encoder cache updated: {current_audio_len:.2f}s audio, "
                f"mel_len={content_mel_len}"
            )

        return encoder_feature, content_mel_len

    def _encode_full(self, input_segments):
        """Full encoder pass — called only on cache miss."""
        if self.use_mlcore:
            coreml_encoder, coreml_input_name, coreml_output_name = self.coreml_encoder_tuple
            mel_padded = log_mel_spectrogram(
                input_segments, n_mels=self.model.dims.n_mels,
                padding=N_SAMPLES, device="cpu",
            ).unsqueeze(0)
            mel = pad_or_trim(mel_padded, N_FRAMES)
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)
            mel_np = np.ascontiguousarray(mel.numpy())
            ml_inputs = {coreml_input_name or "mel": mel_np}
            coreml_outputs = coreml_encoder.predict(ml_inputs)
            if coreml_output_name and coreml_output_name in coreml_outputs:
                encoder_feature_np = coreml_outputs[coreml_output_name]
            else:
                encoder_feature_np = next(iter(coreml_outputs.values()))
            encoder_feature = torch.as_tensor(
                np.array(encoder_feature_np), device=self.device,
            )
        if self.mlx_encoder:
            mlx_mel_padded = mlx_log_mel_spectrogram(
                audio=input_segments.detach(),
                n_mels=self.model.dims.n_mels, padding=N_SAMPLES,
            )
            mlx_mel = mlx_pad_or_trim(mlx_mel_padded, N_FRAMES, axis=-2)
            mlx_encoder_feature = self.mlx_encoder.encoder(mlx_mel[None])
            encoder_feature = torch.as_tensor(mlx_encoder_feature)
            content_mel_len = int((mlx_mel_padded.shape[0] - mlx_mel.shape[0]) / 2)
        elif self.fw_encoder:
            audio_length_seconds = len(input_segments) / 16000
            content_mel_len = int(audio_length_seconds * 100) // 2
            mel_padded_2 = self.fw_feature_extractor(
                waveform=input_segments.numpy(), padding=N_SAMPLES,
            )[None, :]
            mel = fw_pad_or_trim(mel_padded_2, N_FRAMES, axis=-1)
            encoder_feature_ctranslate = self.fw_encoder.encode(mel)
            if self.device == 'cpu':
                encoder_feature_ctranslate = np.array(encoder_feature_ctranslate)
            try:
                encoder_feature = torch.as_tensor(encoder_feature_ctranslate, device=self.device)
            except TypeError:
                try:
                    arr = np.asarray(encoder_feature_ctranslate, dtype=np.float32)
                except (TypeError, ValueError):
                    arr = np.array(encoder_feature_ctranslate)
                    if arr.dtype == np.object_:
                        try:
                            arr = np.stack([
                                np.asarray(item, dtype=np.float32) for item in arr.flat
                            ])
                        except (TypeError, ValueError):
                            arr = np.array(
                                [[float(x) for x in row] for row in arr.flat],
                                dtype=np.float32,
                            )
                encoder_feature = torch.as_tensor(arr, device=self.device)
        else:
            current_len = len(input_segments)
            if (
                self.state.mel_cache_spec is not None
                and current_len >= self.state.mel_cache_waveform_len
            ):
                # Incremental mel: only compute STFT for new audio, append to cache
                cached_mel = self.state.mel_cache_spec
                new_audio = input_segments[self.state.mel_cache_waveform_len:]
                if len(new_audio) > 0:
                    new_mel = log_mel_spectrogram(
                        new_audio, n_mels=self.model.dims.n_mels,
                        padding=0, device=self.device,
                    )
                    full_mel = torch.cat([cached_mel, new_mel], dim=-1)
                else:
                    full_mel = cached_mel
            else:
                # Full recompute (first call or segments were removed)
                full_mel = log_mel_spectrogram(
                    input_segments, n_mels=self.model.dims.n_mels,
                    padding=0, device=self.device,
                )

            # Cache the unpadded mel
            self.state.mel_cache_spec = full_mel
            self.state.mel_cache_waveform_len = current_len

            # Pad for encoder (N_SAMPLES padding, then trim to N_FRAMES)
            mel_padded = F.pad(full_mel, (0, N_SAMPLES // HOP_LENGTH)).unsqueeze(0)
            mel = pad_or_trim(mel_padded, N_FRAMES)
            content_mel_len = int((mel_padded.shape[2] - mel.shape[2]) / 2)
            encoder_feature = self.model.encoder(mel)
        return encoder_feature, content_mel_len

    def _init_sum_logprobs(self):
        return torch.zeros(self.cfg.beam_size, device=self.device)

    def _get_logits_and_cross_attn(self, tokens, encoder_feature):
        if self.state.decoder_type == "greedy":
            result = self.model.decoder(
                tokens, encoder_feature,
                kv_cache=self.state.kv_cache,
                return_cross_attn=True,
            )
            logits, cross_attns, hidden_state = result
            # Two-pass: extract alignment attention from hidden state
            # This is cheap — only computes Q·K^T for alignment heads
            align_attns = self.model.extract_alignment_attn(
                hidden_state, encoder_feature, kv_cache=self.state.kv_cache
            )
            return logits, align_attns
        else:
            logger.debug(f"Logits shape: {tokens.shape}")
            return self.state.inference.logits(
                tokens, encoder_feature, return_cross_attn=True,
            )

    def _check_no_speech(self, logits):
        if self.tokenizer.no_speech is not None:
            probs_at_sot = logits[:, self.state.sot_index, :].float().softmax(dim=-1)
            no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
            if no_speech_probs[0] > self.cfg.nonspeech_prob:
                logger.info("no speech, stop")
                return True
        return False

    def _suppress_blank_tokens(self, logits):
        logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
        return logits

    def _apply_token_suppression(self, logits):
        self.state.suppress_tokens_fn(logits)
        return logits

    def _update_tokens(self, current_tokens, logits, sum_logprobs):
        return self.state.token_decoder.update(current_tokens, logits, sum_logprobs)

    def _process_cross_attention(
        self, cross_attns: List, content_mel_len: int,
    ) -> torch.Tensor:
        """Process alignment attention weights from two-pass extraction.

        The input cross_attns can be:
        - From two-pass: list of tensors per layer, each (batch, num_align_heads, seq, audio_ctx)
          where non-alignment layers are None
        - From legacy single-pass: list of full QK tensors per token step
        """
        num_decoder_layers = len(self.model.decoder.blocks)

        if cross_attns and isinstance(cross_attns[0], list):
            flattened_attns = [attn for layer_list in cross_attns for attn in layer_list]
        else:
            flattened_attns = cross_attns

        # Check if this is two-pass format (per-layer, alignment-heads-only tensors)
        # Two-pass format: each entry is per-layer, only alignment heads present
        is_two_pass = (
            len(flattened_attns) == num_decoder_layers
            and hasattr(self.model, '_alignment_layer_heads')
        )

        if is_two_pass:
            # Two-pass: tensors already contain only alignment heads
            all_head_attns = []
            for layer_idx, attn_mat in enumerate(flattened_attns):
                if attn_mat is None:
                    continue
                attn_mat = F.softmax(attn_mat, dim=-1)
                # Each head in the tensor IS an alignment head
                for local_idx in range(attn_mat.shape[1]):
                    if self.cfg.beam_size == 1:
                        a = attn_mat[0, local_idx, :, :].unsqueeze(0)
                    else:
                        a = attn_mat[:, local_idx, :, :]
                    all_head_attns.append(a)

            if not all_head_attns:
                return torch.zeros(self.cfg.beam_size, 1, content_mel_len, device=self.device)

            # Stack: (num_align_heads, batch, seq, audio_ctx)
            attn_stack = torch.stack(all_head_attns, dim=1)
        else:
            # Legacy single-pass format
            attn_of_alignment_heads = [[] for _ in range(self.state.num_align_heads)]
            for idx, attn_mat in enumerate(flattened_attns):
                if attn_mat is None:
                    continue
                layer_rank = idx % num_decoder_layers
                align_heads_in_layer = self.state.align_source.get(layer_rank, [])
                if not align_heads_in_layer:
                    continue
                attn_mat = F.softmax(attn_mat, dim=-1)
                for align_head_rank, head_id in align_heads_in_layer:
                    if self.cfg.beam_size == 1:
                        if attn_mat.dim() == 4:
                            a = attn_mat[0, head_id, :, :]
                        else:
                            a = attn_mat[head_id, :, :]
                        a = a.unsqueeze(0)
                    else:
                        a = attn_mat[:, head_id, :, :]
                    attn_of_alignment_heads[align_head_rank].append(a)

            tmp = []
            for mat in attn_of_alignment_heads:
                if mat:
                    tmp.append(torch.cat(mat, dim=1))
            if not tmp:
                return torch.zeros(self.cfg.beam_size, 1, content_mel_len, device=self.device)
            attn_stack = torch.stack(tmp, dim=1)

        std, mean = torch.std_mean(
            attn_stack, dim=-2, keepdim=True, unbiased=False,
        )
        attn_stack = (attn_stack - mean) / (std + 1e-8)
        attn_stack = median_filter(attn_stack, 7)
        attn_stack = attn_stack.mean(dim=1)
        attn_stack = attn_stack[:, :, :content_mel_len]
        return attn_stack

    def _get_attended_frames(self, attn):
        most_attended_frames = torch.argmax(attn[:, -1, :], dim=-1)
        return most_attended_frames.tolist(), most_attended_frames[0].item()

    def _is_special_token(self, current_tokens):
        return current_tokens[0, -2].item() >= DEC_PAD

    def _rewind_tokens(self):
        if len(self.state.tokens) > 0:
            return torch.cat(self.state.tokens, dim=1)
        return self.state.tokens[0]

    def _tokens_to_list(self, current_tokens, start_col):
        return current_tokens[0, start_col:].flatten().tolist()

    def _make_new_tokens_tensor(self, hypothesis):
        return (
            torch.tensor([hypothesis], dtype=torch.long)
            .repeat_interleave(self.cfg.beam_size, dim=0)
            .to(device=self.device)
        )

    def _evaluate(self, tensor):
        pass  # No-op for PyTorch

    @torch.inference_mode()
    def infer(self, is_last=False):
        return super().infer(is_last)
