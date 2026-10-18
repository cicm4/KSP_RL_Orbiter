"""Build/install contracts; these checks never start KSP or control a flight."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from orbiter import mod


@pytest.fixture
def game(tmp_path):
    game = tmp_path / "Kerbal Space Program"
    managed = game / "KSP.app/Contents/Resources/Data/Managed"
    managed.mkdir(parents=True)
    for name in mod.ASSEMBLIES:
        (managed / name).write_bytes(b"game-owned assembly")
    return game


def test_build_references_game_assemblies_without_shipping_them(
    game, tmp_path, monkeypatch
):
    called = {}

    def run(command, **kwargs):
        called.update(command=command, **kwargs)
        output = Path(next(value[5:] for value in command if value.startswith("-out:")))
        output.write_bytes(b"compiled addon")

    monkeypatch.setattr(mod.subprocess, "run", run)
    output = tmp_path / "build with spaces/OrbiterAutopilot.dll"
    assert (
        mod.compile_addon(game, output, compiler="/compiler with spaces/mcs") == output
    )
    assert called["command"][0] == "/compiler with spaces/mcs"
    assert "-noconfig" in called["command"] and "-nostdlib+" in called["command"]
    assert called["check"] and called["timeout"] == 120
    references = [value[3:] for value in called["command"] if value.startswith("-r:")]
    assert len(references) == len(mod.ASSEMBLIES)
    assert all(Path(value).is_relative_to(game) for value in references)
    assert list(output.parent.iterdir()) == [output]


def test_missing_game_assemblies_fail_before_compiler(game, tmp_path, monkeypatch):
    (mod.managed_directory(game) / "Assembly-CSharp.dll").unlink()
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *args, **kwargs: pytest.fail("Compiler ran")
    )
    with pytest.raises(ValueError, match="assemblies"):
        mod.compile_addon(game, tmp_path / "test.dll", compiler="mcs")


def test_settings_preserve_venv_interpreter_symlink(tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    text = mod.settings_text(tmp_path, python)
    assert f"python = {python}" in text
    assert f"project_root = {tmp_path}" in text
    assert "auto_start = true" in text


@pytest.mark.parametrize("port", [0, 65536, True, 8765.5])
def test_invalid_ports_do_not_produce_settings(tmp_path, port):
    with pytest.raises(ValueError, match="Ports"):
        mod.settings_text(tmp_path, sys.executable, port=port)


def test_install_receipt_owns_only_addon_files_and_replacement_is_backed_up(
    game, tmp_path
):
    assembly = tmp_path / "OrbiterAutopilot.dll"
    assembly.write_bytes(b"v1")
    settings = mod.settings_text(tmp_path, sys.executable)
    dest = mod.install_addon(game, assembly, settings)
    receipt = json.loads((dest / ".orbiter-install.json").read_text())
    assert receipt["dll_sha256"] == hashlib.sha256(b"v1").hexdigest()
    assert not list(dest.rglob("Assembly-CSharp.dll"))
    with pytest.raises(ValueError, match="--replace"):
        mod.install_addon(game, assembly, settings)
    assembly.write_bytes(b"v2")
    mod.install_addon(game, assembly, settings, replace=True)
    assert (dest / "Plugins/OrbiterAutopilot.dll").read_bytes() == b"v2"
    assert len(list(dest.rglob("*.dll"))) == 1  # Backups must never load as addons.
    backup = next((dest / "backups").glob("*/OrbiterAutopilot.dll.backup"))
    assert backup.read_bytes() == b"v1"


def test_installer_refuses_modified_or_foreign_addons(game, tmp_path):
    assembly = tmp_path / "OrbiterAutopilot.dll"
    assembly.write_bytes(b"ours")
    settings = mod.settings_text(tmp_path, sys.executable)
    dest = mod.install_addon(game, assembly, settings)
    installed = dest / "Plugins/OrbiterAutopilot.dll"
    installed.write_bytes(b"someone else's changes")
    with pytest.raises(ValueError, match="modified"):
        mod.install_addon(game, assembly, settings, replace=True)
    assert installed.read_bytes() == b"someone else's changes"
    (dest / ".orbiter-install.json").unlink()
    with pytest.raises(ValueError, match="receipt"):
        mod.install_addon(game, assembly, settings, replace=True)
