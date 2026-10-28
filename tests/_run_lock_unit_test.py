#!/usr/bin/env python3
"""The run-level chain lock: one prompt owns a run_name at a time.

The 2026-10-28 swap-chain double-press (h3-swap-chain-failures fix 3):
checkpoint_run_lock guards each individual mutation, but two
concurrent top-level prompts against ONE run_name interleave scene
checkpoints, revisions, and manifests across a whole render. The cure
is a run-level lock — Loop Start acquires before preflight (a refused
preflight releases), Loop End heartbeats per completed scene and
releases at every prompt-terminal return.

The 2026-10-28 retest gap 2 cure (the census deadlock): the old
PromptServer census counted ANY executing chain prompt — including the
contender's own — so the stale-takeover branch was unreachable from
inside a chain prompt and a dead holder bricked the run_name until
process restart. Holds now record the holder prompt's id (captured at
acquire, sequential queue), takeover follows prompt aliveness, and
PromptQueue.task_done releases ended holders on every outcome.

Covered:

- A second acquire while the run is held REFUSES with the named voice
  (interleaved checkpoint revisions, wait/interrupt/fresh run_name).
- Release allows a clean re-acquire (the handoff driver's next prompt
  rides this).
- Loop End's top_level_requeue handoff return releases the run.
- Loop End's final-completion return releases the run.
- A stale heartbeat with an UNKNOWN holder prompt allows takeover
  after the stale window (the no-queue fallback).
- A LIVE holder prompt refuses the takeover whatever the heartbeat age
  (prompt aliveness is the authority, not the census).
- A DEAD holder prompt (errored mid-scene, no longer executing) is
  taken over immediately — no stale window, no restart.
- Acquire records the acquiring prompt's own id (the single running
  queue entry) as the hold's holder identity.
- The task_done wrapper releases an ended holder's holds, delegates
  exactly once, and passes the return value through.
- The wrapper's release fires even when the original task_done raises
  (the queue's error path must not strand the lock).
"""

import importlib.util
import json
import pathlib
import sys
import tempfile
import types

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "h3_run_lock"

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


def make_inputs(plan, index, end_clip):
    state = {
        "plan": plan,
        "index": index,
        "range_start": 1,
        "end_clip": end_clip,
        "segments": [],
        "resumed_from": 0,
    }
    images = torch.zeros(5, 1, 1, 3)
    sampled_latent = {"samples": [LATENT_FRAMES, LATENT_AUDIO]}
    shot = plan["shots"][index - 1]
    segment = {
        "index": index,
        "id": shot["id"],
        "revision": "ab" * 16,
        "raw_frames": 124,
        "delivered_frames": 124,
        "history_hash": "hist",
        "seed": shot["seed"],
    }
    return state, images, sampled_latent, segment


def reset_lock():
    with chain._ACTIVE_CHAIN_RUNS_GUARD:
        chain._ACTIVE_CHAIN_RUNS.clear()


def second_acquire_refuses():
    """One prompt holds the run: a second press refuses, naming the
    interleaved-revision risk and the exits."""
    reset_lock()
    chain._acquire_chain_run("lock_refuse")
    try:
        chain._acquire_chain_run("lock_refuse")
    except ValueError as exc:
        message = str(exc)
        assert "already executing in another prompt" in message, message
        assert "interleave" in message, \
            "the refusal must name the corruption class"
        assert "fresh run_name" in message, \
            "the refusal must name an exit"
        return
    finally:
        reset_lock()
    raise AssertionError("a held run_name must refuse a second acquire")


def release_allows_reacquire():
    """The handoff driver's next prompt: release then acquire rides."""
    reset_lock()
    chain._acquire_chain_run("lock_cycle")
    chain._release_chain_run("lock_cycle")
    chain._acquire_chain_run("lock_cycle")
    with chain._ACTIVE_CHAIN_RUNS_GUARD:
        held = "lock_cycle" in chain._ACTIVE_CHAIN_RUNS
    reset_lock()
    assert held, "re-acquire after release must hold the run again"


def requeue_handoff_releases():
    """Loop End's prompt-terminal handoff return drops the lock."""
    reset_lock()
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("lock_handoff", 3)
        state, images, latent, segment = make_inputs(plan, 1, 3)
        run_dir = (pathlib.Path(temporary) / "h3_chains" / "lock_handoff")
        (run_dir / "segments").mkdir(parents=True, exist_ok=True)
        chain._acquire_chain_run("lock_handoff")
        result = chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="top_level_requeue")
        assert "h3_chain_top_level_requeue" in result.get("ui", {}), \
            "the handoff arm is the path under test"
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            still_held = "lock_handoff" in chain._ACTIVE_CHAIN_RUNS
        reset_lock()
        assert not still_held, \
            "the handoff return is prompt-terminal: it must release"


def final_completion_releases():
    """Recursive final completion (index == end_clip) releases too."""
    reset_lock()
    with tempfile.TemporaryDirectory() as temporary:
        chain._output_root = lambda: str(temporary)
        plan = make_plan("lock_final", 1)
        state, images, latent, segment = make_inputs(plan, 1, 1)
        chain._acquire_chain_run("lock_final")
        chain.MiniMaxH3ChainLoopEnd().end(
            None, state, images, latent, segment,
            execution_mode="recursive")
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            still_held = "lock_final" in chain._ACTIVE_CHAIN_RUNS
        reset_lock()
        assert not still_held, \
            "the final-completion return is prompt-terminal: it must release"


def stale_takeover_when_holder_unknown():
    """A crashed holder (no resolvable prompt id) cannot brick the run:
    stale heartbeat + holder prompt unknown = takeover after the stale
    window."""
    reset_lock()
    chain._acquire_chain_run("lock_stale")
    with chain._ACTIVE_CHAIN_RUNS_GUARD:
        chain._ACTIVE_CHAIN_RUNS["lock_stale"]["holder_prompt_id"] = None
        chain._ACTIVE_CHAIN_RUNS["lock_stale"]["heartbeat_at"] = (
            chain.time.monotonic() - chain._CHAIN_RUN_STALE_SECONDS - 60.0)
    # No PromptServer in the test process: holder aliveness is
    # unresolvable — the stale-only fallback takeover path.
    chain._acquire_chain_run("lock_stale")
    with chain._ACTIVE_CHAIN_RUNS_GUARD:
        heartbeat = chain._ACTIVE_CHAIN_RUNS["lock_stale"]["heartbeat_at"]
    reset_lock()
    assert heartbeat > chain.time.monotonic() - 5.0, \
        "takeover must refresh the heartbeat for the new holder"


class FakeQueue:
    """PromptQueue stand-in: int-keyed currently_running items whose
    item[1] is the prompt_id, plus a counting task_done."""

    def __init__(self, *prompt_ids):
        self.mutex = __import__("threading").Lock()
        self.currently_running = {
            index + 1: (index + 1, prompt_id, {})
            for index, prompt_id in enumerate(prompt_ids)}
        self.task_done_calls = []
        self.task_done_raises = None

    def _original_task_done(self, item_id, *args, **kwargs):
        self.task_done_calls.append(item_id)
        self.currently_running.pop(item_id, None)
        if self.task_done_raises is not None:
            raise self.task_done_raises
        return "done:%s" % item_id


class FakeServer:
    def __init__(self, queue):
        self.prompt_queue = queue


def install_fake_promptserver(*prompt_ids):
    queue = FakeQueue(*prompt_ids)
    queue.task_done = queue._original_task_done
    chain.PromptServer = types.SimpleNamespace(instance=FakeServer(queue))
    return queue


def restore_promptserver(original):
    chain.PromptServer = original


def alive_holder_prompt_refuses_even_when_stale():
    """Double-press protection under the new authority: the holder's
    prompt IS executing (fresh or stale heartbeat alike) — refuse."""
    reset_lock()
    original = chain.PromptServer
    install_fake_promptserver("holder-live")
    try:
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            chain._ACTIVE_CHAIN_RUNS["lock_alive"] = {
                "acquired_at": chain.time.monotonic(),
                "heartbeat_at": (chain.time.monotonic()
                                 - chain._CHAIN_RUN_STALE_SECONDS - 60.0),
                "holder_prompt_id": "holder-live"}
        try:
            chain._acquire_chain_run("lock_alive")
        except ValueError as exc:
            message = str(exc)
            assert "already executing in another prompt" in message, message
            assert "interleave" in message, message
            assert "fresh run_name" in message, message
            return
        finally:
            reset_lock()
        raise AssertionError(
            "a live holder prompt must refuse the takeover whatever the "
            "heartbeat age")
    finally:
        restore_promptserver(original)


def dead_holder_prompt_takeover_is_immediate():
    """The 2026-10-28 retest gap 2 cure: a holder that errored mid-scene
    is no longer executing — the contender takes over at once, no stale
    window, no process restart."""
    reset_lock()
    original = chain.PromptServer
    install_fake_promptserver("contender-self")
    try:
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            chain._ACTIVE_CHAIN_RUNS["lock_dead"] = {
                "acquired_at": chain.time.monotonic(),
                "heartbeat_at": chain.time.monotonic(),
                "holder_prompt_id": "prompt-that-errored"}
        chain._acquire_chain_run("lock_dead")
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            hold = dict(chain._ACTIVE_CHAIN_RUNS["lock_dead"])
        reset_lock()
        assert hold["holder_prompt_id"] == "contender-self", \
            "takeover must re-key the hold to the new holder's prompt"
        assert hold["heartbeat_at"] > chain.time.monotonic() - 5.0, \
            "takeover must refresh the heartbeat"
    finally:
        restore_promptserver(original)


def acquire_captures_self_prompt_id():
    """Under sequential execution the single running prompt is self:
    the hold must record its prompt_id as the holder identity."""
    reset_lock()
    original = chain.PromptServer
    install_fake_promptserver("self-prompt")
    try:
        chain._acquire_chain_run("lock_self")
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            holder = chain._ACTIVE_CHAIN_RUNS["lock_self"].get(
                "holder_prompt_id")
        reset_lock()
        assert holder == "self-prompt", \
            "the hold must carry the acquiring prompt's own id"
    finally:
        restore_promptserver(original)


def task_done_wrapper_releases_ended_holder():
    """PromptQueue.task_done fires on success AND error: the wrapper
    releases the ended holder's run holds, delegates exactly once, and
    passes the return value through."""
    reset_lock()
    original = chain.PromptServer
    queue = install_fake_promptserver("holder-a", "other-b")
    try:
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            chain._ACTIVE_CHAIN_RUNS["lock_wrap"] = {
                "acquired_at": chain.time.monotonic(),
                "heartbeat_at": chain.time.monotonic(),
                "holder_prompt_id": "holder-a"}
        chain._install_chain_run_task_done_release()
        chain._install_chain_run_task_done_release()  # idempotent
        result = queue.task_done(1, "history-result", None)
        assert result == "done:1", result
        assert queue.task_done_calls == [1], queue.task_done_calls
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            still_held = "lock_wrap" in chain._ACTIVE_CHAIN_RUNS
        reset_lock()
        assert not still_held, \
            "the holder prompt ended: task_done must release its run"
    finally:
        restore_promptserver(original)


def task_done_wrapper_survives_original_failure():
    """Even when the original task_done raises, the release still fires
    (the queue's own error path must not strand the run lock)."""
    reset_lock()
    original = chain.PromptServer
    queue = install_fake_promptserver("holder-x")
    try:
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            chain._ACTIVE_CHAIN_RUNS["lock_raise"] = {
                "acquired_at": chain.time.monotonic(),
                "heartbeat_at": chain.time.monotonic(),
                "holder_prompt_id": "holder-x"}
        chain._install_chain_run_task_done_release()
        queue.task_done_raises = RuntimeError("queue blew up")
        try:
            queue.task_done(1, "history-result", None)
        except RuntimeError as exc:
            assert "queue blew up" in str(exc), str(exc)
        else:
            raise AssertionError("the original exception must propagate")
        with chain._ACTIVE_CHAIN_RUNS_GUARD:
            still_held = "lock_raise" in chain._ACTIVE_CHAIN_RUNS
        reset_lock()
        assert not still_held, \
            "the finally release must fire even on the error path"
    finally:
        restore_promptserver(original)


def main():
    second_acquire_refuses()
    release_allows_reacquire()
    requeue_handoff_releases()
    final_completion_releases()
    stale_takeover_when_holder_unknown()
    alive_holder_prompt_refuses_even_when_stale()
    dead_holder_prompt_takeover_is_immediate()
    acquire_captures_self_prompt_id()
    task_done_wrapper_releases_ended_holder()
    task_done_wrapper_survives_original_failure()
    print("H3 Run Lock: one prompt per run_name, prompt-terminal release, "
          "prompt-id aliveness takeover, task_done error-path release")


if __name__ == "__main__":
    main()
