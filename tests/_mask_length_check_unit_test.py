#!/usr/bin/env python3
"""The per-frame edit mask length-match law (own-video-mask door).

An edit whose mask dial names a VIDEO carries a per-frame mask
timeline. That timeline must length-match the clip the edit rides —
never shorter, never longer, never silently broadcast: the check
refuses loud, naming both frame counts.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "h3_mask_length_test_pkg"


def _install_stubs():
    package = types.ModuleType(PACKAGE)
    package.__path__ = [ROOT]
    sys.modules[PACKAGE] = package

    nodes = types.ModuleType("%s.nodes" % PACKAGE)
    nodes.FPS = 24
    nodes.AUDIO_HZ = 40
    nodes._resize = lambda images, width, height, crop: images
    sys.modules["%s.nodes" % PACKAGE] = nodes

    masking_ops = types.ModuleType("%s.masking_ops" % PACKAGE)

    def normalize_comfy_mask(mask):
        if isinstance(mask, (list, tuple)):
            mask = torch.stack(list(mask))
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() == 4:
            mask = mask.squeeze(3)
        return mask

    masking_ops.normalize_comfy_mask = normalize_comfy_mask
    sys.modules["%s.masking_ops" % PACKAGE] = masking_ops

    masked_context = types.ModuleType("%s.masked_context" % PACKAGE)
    masked_context._validate_target_streams = lambda *args, **kwargs: None
    sys.modules["%s.masked_context" % PACKAGE] = masked_context


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "%s.source_av_target" % PACKAGE,
        os.path.join(ROOT, "source_av_target.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(count, height=4, width=4):
    return torch.zeros(count, height, width, 3)


def _masks(count, height=4, width=4):
    return torch.ones(count, height, width)


class MaskLengthCheckTest(unittest.TestCase):
    def setUp(self):
        if PACKAGE not in sys.modules:
            _install_stubs()
        self.module = _load_module()

    def _node(self):
        return self.module.MiniMaxH3ContexLoopMaskLengthCheck()

    def test_match_passes_and_names_both_counts(self):
        mask, frames, status = self._node().check(
            _frames(5), _masks(5))
        self.assertEqual(int(frames), 5)
        self.assertEqual(tuple(mask.shape), (5, 4, 4))
        self.assertIn("5", status)

    def test_short_mask_refuses_naming_both_counts(self):
        with self.assertRaises(ValueError) as raised:
            self._node().check(_frames(5), _masks(4))
        message = str(raised.exception)
        self.assertIn("4", message)
        self.assertIn("5", message)

    def test_long_mask_refuses_never_truncates(self):
        with self.assertRaises(ValueError) as raised:
            self._node().check(_frames(5), _masks(6))
        message = str(raised.exception)
        self.assertIn("6", message)
        self.assertIn("5", message)

    def test_single_frame_mask_video_refuses_no_silent_broadcast(self):
        with self.assertRaises(ValueError) as raised:
            self._node().check(_frames(5), _masks(1))
        message = str(raised.exception)
        self.assertIn("1", message)
        self.assertIn("5", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
