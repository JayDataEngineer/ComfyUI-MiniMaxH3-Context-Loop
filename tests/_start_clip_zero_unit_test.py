#!/usr/bin/env python3
"""Loop Start start_clip=0: auto-resume from the saved lineage.

The 2026-10-28 retest gap 1 (bare requeue re-POST re-rendered scene 1
three times): the recursion-ceiling message promised "will resume from
it", but nothing resolved the resume point — a bare re-POST carried the
default start_clip 1 and re-rendered scene 1 from scratch. The cure
makes 0 the auto-resume sentinel: Loop Start resolves it from the
durable lineage (newest pending next-scene handoff, else one past the
highest contiguous saved checkpoint, else 1) BEFORE acquiring the run
lock, and the ceiling message names the door.

Covered:

- A pending next_scene handoff resolves to its scene (the bare requeue
  continues exactly where top_level_requeue stopped).
- A consumed/cancelled handoff is ignored (terminal history is never
  new work).
- The contiguous checkpoint lineage resolves one past its highest
  scene (a crashed partial chain resumes where it stopped).
- A fresh run (no handoffs, no checkpoints) resolves to scene 1.
- The handoff wins over the checkpoint scan when both exist.
- A handoff scene beyond the plan's shot count refuses loudly (the
  lineage belongs to a different plan).
- start_clip 0 with a non-empty scene_range refuses at the door
  (a range needs an explicit scene number).
"""

import importlib.util
import json
import pathlib
import sys
import tempfile
import types

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "h3_start_clip_zero"

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


def make_handoff(run_name, scene, status="pending"):
    store = chain._HandoffStore(chain._output_root())
    record = store.create(
        run_name, action="next_scene",
        scene=scene, start_clip=scene, end_clip=scene + 2,
        seed=7,
        source_revision="ab" * 16,
        source_checkpoint_sha256=None,
        workflow_fingerprint="hash",
        predecessor_scene=scene - 1,
        transition_key="tk_%d" % scene,
        handoff_id="next_scene_%04d_%s" % (scene, "tk" * 8),
        working_branch_id="main")
    if status != "pending":
        # Legal path per _HANDOFF_TRANSITIONS: pending -> claimed ->
        # queued -> consumed (the coordinator's claim/queue/consume
        # cycle, compressed for the fixture).
        store.transition(run_name, record["handoff_id"], "claimed")
        store.transition(run_name, record["handoff_id"], "queued")
        store.transition(run_name, record["handoff_id"], status)
    return record


def pending_handoff_resolves():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_handoff", 4)
        make_handoff("auto_handoff", 3)
        resolved = chain._resolve_auto_start_clip(plan)
        assert resolved == 3, \
            "a pending next_scene handoff at 3 must resume at 3, got %d" \
            % resolved


def consumed_handoff_is_ignored():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_consumed", 4)
        make_handoff("auto_consumed", 3, status="consumed")
        resolved = chain._resolve_auto_start_clip(plan)
        assert resolved == 1, \
            "a consumed handoff is terminal history, not new work; " \
            "fresh run expected, got %d" % resolved


def checkpoint_lineage_resumes_next():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_ckpt", 4)
        checkpoint_dir = (pathlib.Path(temporary) / "h3_chains"
                          / "auto_ckpt" / "checkpoints")
        checkpoint_dir.mkdir(parents=True)
        for scene in (1, 2):
            (checkpoint_dir / ("clip_%04d.json" % scene)).write_text("{}")
        resolved = chain._resolve_auto_start_clip(plan)
        assert resolved == 3, \
            "scenes 1-2 checkpointed must resume at 3, got %d" % resolved


def fresh_run_starts_at_one():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_fresh", 4)
        resolved = chain._resolve_auto_start_clip(plan)
        assert resolved == 1, \
            "no lineage anywhere must start at 1, got %d" % resolved


def handoff_wins_over_checkpoints():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_both", 5)
        make_handoff("auto_both", 4)
        checkpoint_dir = (pathlib.Path(temporary) / "h3_chains"
                          / "auto_both" / "checkpoints")
        checkpoint_dir.mkdir(parents=True)
        for scene in (1, 2):
            (checkpoint_dir / ("clip_%04d.json" % scene)).write_text("{}")
        resolved = chain._resolve_auto_start_clip(plan)
        assert resolved == 4, \
            "the pending handoff names the exact continuation scene; " \
            "got %d" % resolved


def out_of_plan_handoff_refuses():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_over", 3)
        make_handoff("auto_over", 5)
        try:
            chain._resolve_auto_start_clip(plan)
        except ValueError as exc:
            assert "different plan" in str(exc), str(exc)
            assert "explicit start_clip" in str(exc), str(exc)
            return
        raise AssertionError(
            "a handoff scene beyond the plan must refuse loudly")


def zero_with_scene_range_refuses_at_start():
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("auto_range", 3)
        try:
            chain.MiniMaxH3ChainLoopStart().start(
                plan, 0, scene_range="2:3")
        except ValueError as exc:
            message = str(exc)
            assert "scene_range requires an explicit start_clip" in message, \
                message
            return
        raise AssertionError(
            "start_clip 0 with a scene_range must refuse at the door")


def main():
    pending_handoff_resolves()
    consumed_handoff_is_ignored()
    checkpoint_lineage_resumes_next()
    fresh_run_starts_at_one()
    handoff_wins_over_checkpoints()
    out_of_plan_handoff_refuses()
    zero_with_scene_range_refuses_at_start()
    print("H3 start_clip 0: auto-resume from handoff, else contiguous "
          "checkpoints, else fresh; explicit ranges refused")


if __name__ == "__main__":
    main()
