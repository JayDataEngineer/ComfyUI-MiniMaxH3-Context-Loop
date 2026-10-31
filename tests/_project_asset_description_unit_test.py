#!/usr/bin/env python3
"""Project asset tag-type, tag prefix, and description persistence checks."""

import pathlib
import sys
import tempfile
import types

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

folder_paths = types.ModuleType("folder_paths")
folder_paths.get_output_directory = lambda: str(ROOT)
folder_paths.get_temp_directory = lambda: str(ROOT)
folder_paths.get_input_directory = lambda: str(ROOT)
sys.modules.setdefault("folder_paths", folder_paths)

from project_assets import (  # noqa: E402
    ProjectAssetConflictError, ProjectAssetStore, normalize_asset_tag_type)


def rejected(function, *args):
    try:
        function(*args)
    except ValueError as exc:
        return str(exc)
    raise AssertionError("Expected ValueError")


def store_checks(root):
    input_root = root / "input"
    loose = input_root / "loose"
    loose.mkdir(parents=True)
    picture = loose / "alice.png"
    Image.new("RGB", (16, 16), (200, 10, 10)).save(picture)
    store = ProjectAssetStore(str(input_root), str(root / "output"))
    asset_id = store.import_file(
        "episode_1", picture, role="picture", tag="alice")["asset"]["id"]
    revision = store.load("episode_1")["revision"]

    assert normalize_asset_tag_type(" Char ") == "char"
    assert normalize_asset_tag_type("Hero Prop!") == "hero-prop"
    assert normalize_asset_tag_type("") == ""
    assert "begin with a letter" in rejected(normalize_asset_tag_type, "9x")

    entry = store.update("episode_1", asset_id, {"tag_type": "char"})["asset"]
    assert entry["tag_type"] == "char" and entry["tag"] == "alice"
    entry = store.update(
        "episode_1", asset_id, {"tag_type_prefix": True})["asset"]
    assert entry["tag"] == "char_alice"
    # Switching type swaps the prefix instead of stacking it.
    entry = store.update("episode_1", asset_id, {"tag_type": "style"})["asset"]
    assert entry["tag"] == "style_alice"
    # A typed tag keeps the prefix whether or not the user typed it.
    assert store.update("episode_1", asset_id, {"tag": "bob"})[
        "asset"]["tag"] == "style_bob"
    assert store.update("episode_1", asset_id, {"tag": "style_bob"})[
        "asset"]["tag"] == "style_bob"
    entry = store.update(
        "episode_1", asset_id, {"tag_type_prefix": False})["asset"]
    assert entry["tag"] == "bob" and "tag_type_prefix" not in entry
    entry = store.update("episode_1", asset_id, {"tag_type": ""})["asset"]
    assert "tag_type" not in entry

    entry = store.update("episode_1", asset_id, {
        "description": "Red square.\r\nFlat color."})["asset"]
    assert entry["description"] == "Red square.\nFlat color."
    # The description is persisted with the asset in the project catalog.
    reloaded = ProjectAssetStore(str(input_root), str(root / "output"))
    stored = next(item for item in reloaded.public_catalog("episode_1")[
        "assets"] if item["id"] == asset_id)
    assert stored["description"] == "Red square.\nFlat color."
    # Description and tag-type are notes; they never change the reference
    # fingerprint the way options do (tag renames still do).
    store.update("episode_1", asset_id, {"tag": "alice"})
    assert store.load("episode_1")["revision"] == revision
    # A late generated description is saved only over the text it was
    # generated from; a newer manual edit raises a conflict and is kept.
    store.update("episode_1", asset_id, {"description": "Manual newer."})
    try:
        store.update("episode_1", asset_id, {
            "description": "Generated late.",
            "expected_description": "Red square.\r\nFlat color."})
    except ProjectAssetConflictError as exc:
        assert "changed while a new one was being generated" in str(exc)
    else:
        raise AssertionError("A stale generated description was saved")
    assert store.load("episode_1")["assets"][0]["description"] == "Manual newer."
    assert store.update("episode_1", asset_id, {
        "description": "Generated.",
        "expected_description": "Manual newer.\n"})["asset"][
            "description"] == "Generated."
    entry = store.update("episode_1", asset_id, {"description": "  "})["asset"]
    assert "description" not in entry
    assert store.update("episode_1", asset_id, {
        "description": "From empty.", "expected_description": ""})[
            "asset"]["description"] == "From empty."
    store.update("episode_1", asset_id, {"description": ""})
    entry = store.update(
        "episode_1", asset_id, {"subject": "  the red\n umbrella "})["asset"]
    assert entry["subject"] == "the red umbrella"
    assert "subject" not in store.update(
        "episode_1", asset_id, {"subject": ""})["asset"]

    second = store.import_file(
        "episode_1", picture, role="picture", tag="char_alice")["asset"]["id"]
    store.update("episode_1", second, {"tag_type": "char"})
    assert "already uses @char_alice" in rejected(
        store.update, "episode_1", asset_id,
        {"tag_type": "char", "tag_type_prefix": True})

    # A cross-project import keeps notes without doubling the prefix.
    store.update("episode_1", second, {
        "tag_type_prefix": True, "description": "Hero.",
        "subject": "the scarf"})
    imported = store.import_project_asset(
        "episode_2", "episode_1", second)["asset"]
    assert imported["tag"] == "char_alice"
    assert imported["tag_type"] == "char"
    assert imported["tag_type_prefix"] is True
    assert imported["description"] == "Hero."
    assert imported["subject"] == "the scarf"
    return picture


def main():
    with tempfile.TemporaryDirectory() as temporary:
        store_checks(pathlib.Path(temporary))
    print("H3 Project Asset catalog: tag-type, prefix naming, description, "
          "subject, and late-result guard pass")


if __name__ == "__main__":
    main()
