#!/usr/bin/env python3
"""The advance lookup's id algebra: merged top-level ids ride verbatim.

The 2026-11-13 multi-scene burn: the estate's planMerge namespaces every
stage node "sid::nid.idx"; _explore_output_nodes re-joined such advance
targets with a full split(".") into "sid::a.0.sid::b.0" and _recurse died
on dynprompt.get_node with NodeNotFoundError before the second scene
delivered. Pins the split-on-the-last-dot law (bare output ids in dotted
parents stay subgraph-relative), the verbatim ride for namespaced and
dotted output ids, and the plain top-level case.

Offline like _alternate_take_unit_test: folder_paths/server/nodes are
stubs, no serving ComfyUI.
"""

import importlib
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "h3_explore_output_nodes_unit"

folder_paths = types.ModuleType("folder_paths")
folder_paths.output_directory = str(ROOT)
folder_paths.get_output_directory = lambda: folder_paths.output_directory
sys.modules["folder_paths"] = folder_paths

server = types.ModuleType("server")
server.PromptServer = type("PromptServer", (), {"instance": None})
sys.modules["server"] = server

nodes = types.ModuleType("nodes")
nodes.NODE_CLASS_MAPPINGS = {}
sys.modules["nodes"] = nodes

package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

shared_nodes = types.ModuleType(PACKAGE + ".nodes")
shared_nodes.MiniMaxH3MotionContext = object
shared_nodes._claim_inline_patch_ownership = lambda _conditioning=None: "test patch owner"
shared_nodes._prepare_native_guide_conditioning = lambda *args: None
shared_nodes._resize = lambda *args: None
shared_nodes._streams_from_latent = lambda *args: None
sys.modules[shared_nodes.__name__] = shared_nodes

upscale_nodes = importlib.import_module(PACKAGE + ".upscale_nodes")


class _FakeDynPrompt:
    """get_original_prompt + get_display_node_id, nothing else."""

    def __init__(self, prompt, display=None):
        self._prompt = prompt
        self._display = display or {}

    def get_original_prompt(self):
        return self._prompt

    def get_display_node_id(self, node_id):
        return self._display.get(node_id, node_id)


class _TestOutputNode:
    OUTPUT_NODE = True


TEST_NODE_KEY = "_ExploreOutputNodesUnitTestOutput"


class ExploreOutputNodesTests(unittest.TestCase):
    def _advance(self, prompt, parents, display=None):
        loop_end = upscale_nodes.MiniMaxH3ChainUpscaleLoopEnd.__new__(
            upscale_nodes.MiniMaxH3ChainUpscaleLoopEnd)
        # `parents` + `upstream` arrive from _explore_dependencies (the
        # loop-end ancestry); the prompt carries the graph's OUTPUT nodes
        upstream = {pid: [] for pid in parents}
        nodes.NODE_CLASS_MAPPINGS[TEST_NODE_KEY] = _TestOutputNode
        try:
            loop_end._explore_output_nodes(
                _FakeDynPrompt(prompt, display), upstream, list(parents))
        finally:
            nodes.NODE_CLASS_MAPPINGS.pop(TEST_NODE_KEY, None)
        return upstream

    def test_merged_ids_ride_verbatim(self):
        # THE BURN: the full split re-joined "s_1::KimodoExportFBX.0" +
        # "s_1::SaveRiggedModel.0" into
        # "s_1::KimodoExportFBX.s_1::SaveRiggedModel.0" — a node no
        # graph ever carried (NodeNotFoundError in _recurse).
        parent = "s_1::KimodoExportFBX.0"
        out = "s_1::SaveRiggedModel.0"
        upstream = self._advance(
            {out: {"class_type": TEST_NODE_KEY, "inputs": {"x": [parent, 0]}}},
            [parent])
        self.assertEqual(upstream[parent], [out])

    def test_multi_scene_advance_chain_stays_typed(self):
        # two merged stages, each parent advanced by its OWN stage's
        # output node; no dotted fabrication across the window seam
        p1 = "s_1::window1_MeliteAudioTensorSlice.0"
        p2 = "s_2::window1_MeliteAudioTensorSlice.0"
        o1, o2 = "s_1::SaveVideo.0", "s_2::SaveVideo.0"
        upstream = self._advance(
            {
                o1: {"class_type": TEST_NODE_KEY, "inputs": {"a": [p1, 0]}},
                o2: {"class_type": TEST_NODE_KEY, "inputs": {"a": [p2, 0]}},
            },
            [p1, p2])
        self.assertEqual(upstream[p1], [o1])
        self.assertEqual(upstream[p2], [o2])

    def test_subgraph_bare_output_stays_relative(self):
        # ComfyUI subgraph notation: a BARE output id in a dotted
        # parent swaps the LAST segment only (rsplit keeps multi-dot
        # prefixes whole)
        upstream = self._advance(
            {"9": {"class_type": TEST_NODE_KEY, "inputs": {"x": ["1.2.7", 0]}}},
            ["1.2.7"])
        self.assertEqual(upstream["1.2.7"], ["1.2.9"])

    def test_plain_top_level_rides_verbatim(self):
        upstream = self._advance(
            {"9": {"class_type": TEST_NODE_KEY, "inputs": {"x": ["7", 0]}}},
            ["7"])
        self.assertEqual(upstream["7"], ["9"])


if __name__ == "__main__":
    unittest.main()
