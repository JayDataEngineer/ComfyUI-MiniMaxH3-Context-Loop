"""Exact MiniMax H3 picture/audio timeline helpers.

H3 picture runs live on a 24 fps clock while generated audio latents live on
a 40 Hz clock. A decoded audio stream can consequently land a few
milliseconds either side of its exact frame-derived duration. Small grid
mismatches are time-conformed here instead of being repaired with silence.
"""

from __future__ import annotations

from fractions import Fraction
import logging

import torch

try:
    import torchaudio
except ImportError:
    torchaudio = None


_LOG = logging.getLogger("minimax_h3_context_loop.av_timing")

# Private AUDIO mapping fields passed only between Loop Trim and Segment Save.
# Keeping this transport inside the existing AUDIO value avoids another public
# socket and another copy of the same decoded stream in user workflows.
AUDIO_WITH_OVERLAP_WAVEFORM_KEY = "_h3_audio_with_overlap_waveform"
AUDIO_WITH_OVERLAP_FRAMES_KEY = "_h3_audio_with_overlap_frames"
AUDIO_TRIM_FRAMES_KEY = "_h3_audio_trim_frames"
AUDIO_TRIM_MODE_KEY = "_h3_audio_trim_mode"


def sample_boundary_from_seconds(seconds: float, sample_rate: int) -> int:
    """Return the nearest PCM sample on one absolute time boundary."""
    return int(round(float(seconds) * int(sample_rate)))


def sample_boundary_from_frames(
        frame_position: int, sample_rate: int, fps: int = 24) -> int:
    """Return the nearest PCM sample on one absolute video-frame boundary."""
    return int(round(int(frame_position) / float(fps) * int(sample_rate)))


def conform_waveform_length(
        waveform, samples: int, label: str,
        max_fractional_change: float = 0.005,
        grid_step_samples: int = 0):
    """Time-conform a small decoder/grid mismatch to an exact sample span.

    A larger discrepancy is likely incorrect wiring or timing metadata and is
    rejected instead of being hidden by a large resample.

    ``grid_step_samples`` widens the acceptance to ONE step of the audio
    encoder's own latent grid (H3: 40 Hz → 0.025 s per step, ~800 samples
    at 32 kHz). The step is the grid's resolution: ComfyUI rounds the
    required audio steps to the nearest integer, so a window shorter than
    1.67 s (40 frames) can land a full step — 0.91% on a 22-frame window —
    off the frame-exact span with nothing wrong wired. The fractional cap
    alone cannot scale to short windows because the error is absolute;
    two steps or more is still wiring/metadata and refuses.
    """
    target = int(samples)
    current = int(waveform.shape[-1])
    if current == target:
        return waveform
    if current <= 0 or target <= 0:
        raise ValueError(
            "%s has an invalid audio sample length %d -> %d." %
            (label, current, target))

    fractional_change = abs(target - current) / float(target)
    within_grid_step = (
        grid_step_samples > 0 and
        abs(target - current) <= int(grid_step_samples))
    if fractional_change > float(max_fractional_change) and not within_grid_step:
        raise ValueError(
            "%s differs from its exact frame timeline by %.3f%% "
            "(%d -> %d samples), too large for H3 audio time-conformance." %
            (label, fractional_change * 100.0, current, target))

    ratio = Fraction(target, current).limit_denominator(10000)
    if torchaudio is not None:
        conformed = torchaudio.functional.resample(
            waveform, int(ratio.denominator), int(ratio.numerator))
    else:
        shape = tuple(waveform.shape)
        conformed = torch.nn.functional.interpolate(
            waveform.reshape(-1, 1, current),
            size=target,
            mode="linear",
            align_corners=False,
        ).reshape(*shape[:-1], target)

    # A rational approximation or resampler backend may miss by one sample.
    # Correct the residual by interpolation so silence is never introduced.
    if int(conformed.shape[-1]) != target:
        shape = tuple(conformed.shape)
        conformed = torch.nn.functional.interpolate(
            conformed.reshape(-1, 1, int(conformed.shape[-1])),
            size=target,
            mode="linear",
            align_corners=False,
        ).reshape(*shape[:-1], target)

    _LOG.info(
        "%s time-conformed %d -> %d audio samples (%+.4f%%) to match "
        "the exact video timeline",
        label, current, target,
        (target - current) / float(current) * 100.0)
    return conformed
