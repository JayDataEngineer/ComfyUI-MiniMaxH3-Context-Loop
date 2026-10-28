#!/usr/bin/env python3
"""The recursion ceiling: >2-scene recursive plans refuse at hop 1.

The 2026-10-28 swap-chain burn: recursive execution expands each next
scene as a nested subgraph, and subgraph-qualified node ids stop
resolving at depth >= 3 — a >2-scene recursive plan renders scene 2,
then dies inside the engine with NodeNotFoundError after the GPU work,
with no diagnosis. The cure is Loop End refusing at the FIRST
expansion boundary (scene 1's checkpoint transaction has already
committed; nothing is wasted) and naming the mode that survives.

Covered:

- A 3-scene recursive plan raises at scene 1's Loop End with a message
  naming ``top_level_requeue`` (the refusal is the diagnosis).
- A 2-scene recursive plan does NOT refuse (the head/tail workaround's
  legality is pinned — the ceiling gates owed clips, not the mode).
- A 4-scene plan with ``end_clip`` capped at 2 by scene_range does NOT
  refuse at scene 1 (scene_range is respected: the range is the debt).
- ``top_level_requeue`` never trips the ceiling (the surviving mode is
  exempt by construction, pinned anyway).
"""

import importlib.util
import json
import pathlib
import sys
import types

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "h3_recursive_ceiling"

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: str(ROOT)
folder_paths.get_temp_directory = lambda: str(ROOT)
folder_paths.get_input_directory = lambda: str(ROOT)
folder_paths.get_annotated_filepath = lambda value: str(value)
sys.modules["folder_paths"] = folder_paths

package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

LATENT_FRAMES = torch.zeros(2, 1, 1, 1)
LATENT_AUDIO = torch.zeros(2)
shared_nodes = types.ModuleType(PACKAGE + ".nodes")
shared_nodes.MiniMaxH3MotionContext = object
shared_nodes._claim_inline_patch_ownership = lambda _conditioning=None: "test"
shared_nodes._prepare_native_guide_conditioning = lambda value: value
shared_nodes._resize = lambda *args: None
shared_nodes._streams_from_latent = lambda _latent: [LATENT_FRAMES,
                                                     LATENT_AUDIO]
sys.modules[shared_nodes.__name__] = shared_nodes

spec = importlib.util.spec_from_file_location(
    PACKAGE + ".chain_nodes", ROOT / "chain_nodes.py")
chain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chain
spec.loader.exec_module(chain)


def make_plan(run_name, shot_count):
    policy = chain._contract_compose_chain_policy(
        chain._contract_audio_policy("source", "on", "off"),
        chain._contract_transition_policy(
            "guide", expert_override=True,
            continuation_mode="guide", context_length=5),
        audio_context_length=5)
    shots = [
        {"id": "shot_%d" % (index + 1),
         "prompt": "Shot %d action." % (index + 1),
         "length": 124}
        for index in range(shot_count)
    ]
    return chain._normalize_plan(
        json.dumps({"shots": shots}),
        run_name, 64, 64, 5, "video", "head", "disabled",
        "source_track", 5, 1.0, 8, 7, 18, "stack:auto:v1", 0,
        "guide", policy)


def make_inputs(plan, end_clip):
    state = {
        "plan": plan,
        "index": 1,
        "range_start": 1,
        "end_clip": end_clip,
        "segments": [],
        "resumed_from": 0,
    }
    images = torch.zeros(5, 1, 1, 3)
    sampled_latent = {"samples": [LATENT_FRAMES, LATENT_AUDIO]}
    segment = {
        "index": 1,
        "id": plan["shots"][0]["id"],
        "revision": "ab" * 16,
        "raw_frames": 124,
        "delivered_frames": 124,
        "history_hash": "hist",
        "seed": plan["shots"][0]["seed"],
    }
    return state, images, sampled_latent, segment


def ceiling_refuses_three_scene_recursive():
    """Scene 1 of a 3-scene recursive plan refuses loud, naming the mode."""
    plan = make_plan("ceiling_refuse", 3)
    state, images, latent, segment = make_inputs(plan, 3)
    try:
        chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="recursive")
    except ValueError as exc:
        message = str(exc)
        assert "recursive execution cannot expand past clip 2" in message, message
        assert "top_level_requeue" in message, \
            "the refusal must name the surviving mode"
        assert "3 clips" in message, \
            "the refusal must state the owed clip count"
        return
    raise AssertionError("a 3-scene recursive plan must refuse at hop 1")


def two_scene_recursive_stays_legal():
    """The head/tail workaround: 2 owed clips recurse exactly as before."""
    plan = make_plan("ceiling_legal", 2)
    state, images, latent, segment = make_inputs(plan, 2)
    recursed = []
    original_recurse = chain.MiniMaxH3ChainLoopEnd._recurse

    def recorder(self, flow, next_state, dynprompt, unique_id):
        recursed.append(next_state)
        return {"result": ("expansion", None, None, None), "expand": None}

    chain.MiniMaxH3ChainLoopEnd._recurse = recorder
    try:
        chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="recursive")
    finally:
        chain.MiniMaxH3ChainLoopEnd._recurse = original_recurse
    assert len(recursed) == 1, \
        "a 2-scene recursive plan must still expand into scene 2"
    assert recursed[0]["index"] == 2


def scene_range_cap_stays_legal():
    """end_clip (scene_range) is the debt: a 2-clip range on a 4-scene
    plan does not refuse at scene 1."""
    plan = make_plan("ceiling_range", 4)
    state, images, latent, segment = make_inputs(plan, 2)
    recursed = []
    original_recurse = chain.MiniMaxH3ChainLoopEnd._recurse

    def recorder(self, flow, next_state, dynprompt, unique_id):
        recursed.append(next_state)
        return {"result": ("expansion", None, None, None), "expand": None}

    chain.MiniMaxH3ChainLoopEnd._recurse = recorder
    try:
        chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="recursive")
    finally:
        chain.MiniMaxH3ChainLoopEnd._recurse = original_recurse
    assert len(recursed) == 1, \
        "a scene_range-capped 2-clip window must recurse, not refuse"


def requeue_mode_is_exempt():
    """The surviving mode never trips the ceiling — 3 owed clips, scene
    1, top_level_requeue: no refusal, the handoff path runs instead."""
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("ceiling_requeue", 3)
        state, images, latent, segment = make_inputs(plan, 3)
        run_dir = (pathlib.Path(temporary) / "h3_chains" / "ceiling_requeue")
        (run_dir / "segments").mkdir(parents=True, exist_ok=True)
        result = chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="top_level_requeue")
        assert "h3_chain_top_level_requeue" in result.get("ui", {}), \
            "requeue mode owns >2-scene chains; the ceiling is recursive-only"


def main():
    ceiling_refuses_three_scene_recursive()
    two_scene_recursive_stays_legal()
    scene_range_cap_stays_legal()
    requeue_mode_is_exempt()
    print("H3 Recursion Ceiling: hop-1 refusal for >2-scene recursive "
          "plans, 2-clip legality and scene_range respect pinned")


if __name__ == "__main__":
    main()
