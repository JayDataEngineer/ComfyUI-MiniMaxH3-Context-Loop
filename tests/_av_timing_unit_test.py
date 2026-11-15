#!/usr/bin/env python3
"""CPU regressions for exact H3 PCM boundaries and time-conformance."""

import importlib.util
import pathlib

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "h3_av_timing_unit", ROOT / "av_timing.py")
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)


def main():
    short = torch.ones((1, 2, 346_400), dtype=torch.float32)
    conformed = timing.conform_waveform_length(
        short, 346_667, "260-frame test")
    assert tuple(conformed.shape) == (1, 2, 346_667)
    assert torch.all(conformed > 0.98)

    long = torch.linspace(-1.0, 1.0, 165_600).reshape(1, 1, -1)
    contracted = timing.conform_waveform_length(
        long, 165_333, "124-frame test")
    assert tuple(contracted.shape) == (1, 1, 165_333)
    assert torch.isfinite(contracted).all()

    try:
        timing.conform_waveform_length(
            torch.ones((1, 1, 100)), 200, "bad wiring")
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("large audio mismatch was silently conformed")

    # THE 22-FRAME GRID-STEP ARM (2026-11-15, the refine handoff burn):
    # a decoded 29,600-sample window (37 audio-latent steps @ 40 Hz)
    # against the frame-exact 29,333 (22 frames @ 24 fps @ 32 kHz) is
    # ONE grid step off — 0.910%, over the 0.5% fractional cap, but
    # nothing wrong wired. With grid_step_samples it conforms; two
    # steps off still refuses.
    decoded_22f = torch.linspace(-1.0, 1.0, 29_600).reshape(1, 1, -1)
    conformed_22f = timing.conform_waveform_length(
        decoded_22f, 29_333, "22-frame full audio",
        grid_step_samples=800)
    assert tuple(conformed_22f.shape) == (1, 1, 29_333)
    assert torch.isfinite(conformed_22f).all()
    try:
        timing.conform_waveform_length(
            decoded_22f, 29_333 - 800, "two steps off",
            grid_step_samples=800)
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError(
            "beyond-one-grid-step mismatch was silently conformed")

    start = 0.3 / 32_000.0
    first = timing.sample_boundary_from_seconds(start, 32_000)
    last = timing.sample_boundary_from_seconds(start + 124 / 24.0, 32_000)
    assert first == 0
    assert last - first == 165_334
    assert timing.sample_boundary_from_frames(124, 32_000) == 165_333

    print("H3 AV timing: small grid drift is conformed without silence; absolute boundaries are exact")


if __name__ == "__main__":
    main()
