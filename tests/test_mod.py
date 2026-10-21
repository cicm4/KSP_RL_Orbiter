"""Build/install contracts; these checks never start KSP or control a flight."""

import hashlib
import json
from pathlib import Path

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


@pytest.fixture
def policies(tmp_path):
    folder = tmp_path / "policies"
    folder.mkdir()
    for name in mod.POLICIES:
        (folder / name).write_bytes(b"ORBPOL1\n" + name.encode())
    return folder


def test_build_compiles_every_native_source_against_game_assemblies(game, tmp_path, monkeypatch):
    called = {}

    def run(command, **kwargs):
        called.update(command=command, **kwargs)
        sources = [value for value in command if value.endswith(".cs")]
        called["sources"] = [Path(value).read_bytes() for value in sources]
        output = Path(next(value[5:] for value in command if value.startswith("-out:")))
        output.write_bytes(b"compiled addon")

    monkeypatch.setattr(mod.subprocess, "run", run)
    output = tmp_path / "build with spaces/OrbiterAutopilot.dll"
    assert mod.compile_addon(game, output, compiler="/compiler with spaces/mcs") == output
    assert called["command"][0] == "/compiler with spaces/mcs"
    assert "-noconfig" in called["command"] and "-nostdlib+" in called["command"]
    assert called["check"] and called["timeout"] == 180
    references = [value[3:] for value in called["command"] if value.startswith("-r:")]
    assert len(references) == len(mod.ASSEMBLIES)
    assert all(Path(value).is_relative_to(game) for value in references)
    expected = [source.read_bytes() for source in mod.native_sources()]
    assert called["sources"] == expected
    names = {source.name for source in mod.native_sources()}
    assert {"Policy.cs", "Planner.cs", "Mission.cs", "Attitude.cs", "Addon.cs"} <= names
    assert list(output.parent.iterdir()) == [output]


def test_native_addon_has_no_python_krpc_or_network_dependency():
    text = "\n".join(source.read_text() for source in mod.native_sources())
    for forbidden in ("System.Net", "Process.Start", "KRPC", "krpc", "python -m"):
        assert forbidden not in text


def test_missing_game_assemblies_fail_before_compiler(game, tmp_path, monkeypatch):
    (mod.managed_directory(game) / "Assembly-CSharp.dll").unlink()
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: pytest.fail("Compiler ran"))
    with pytest.raises(ValueError, match="assemblies"):
        mod.compile_addon(game, tmp_path / "test.dll", compiler="mcs")


def test_install_receipt_owns_dll_and_policies_and_replacement_is_backed_up(game, tmp_path, policies):
    first = tmp_path / "first.dll"
    first.write_bytes(b"MZ first")
    destination = mod.install_addon(game, first, policies)
    receipt = json.loads((destination / ".orbiter-install.json").read_text())
    assert receipt["application"] == "orbiter-autopilot" and receipt["runtime"] == "native"
    assert receipt["dll_sha256"] == hashlib.sha256(b"MZ first").hexdigest()
    assert sorted(path.name for path in (destination / "Policies").iterdir()) == sorted(mod.POLICIES)
    assert not (destination / "settings.cfg").exists()
    second = tmp_path / "second.dll"
    second.write_bytes(b"MZ second")
    with pytest.raises(ValueError, match="--replace"):
        mod.install_addon(game, second, policies)
    mod.install_addon(game, second, policies, replace=True)
    assert (destination / "Plugins/OrbiterAutopilot.dll").read_bytes() == b"MZ second"
    backups = list((game / "OrbiterAutopilot-backups").iterdir())
    assert len(backups) == 1 and (backups[0] / "OrbiterAutopilot.dll").read_bytes() == b"MZ first"
    # KSP loads every DLL below GameData; backups must stay outside it.
    assert not list((game / "GameData").rglob("*.dll.backup"))
    assert [p for p in (game / "GameData").rglob("*.dll")] == [destination / "Plugins/OrbiterAutopilot.dll"]


def test_installer_refuses_modified_or_foreign_addons(game, tmp_path, policies):
    assembly = tmp_path / "addon.dll"
    assembly.write_bytes(b"MZ addon")
    destination = mod.install_addon(game, assembly, policies)
    (destination / "Plugins/OrbiterAutopilot.dll").write_bytes(b"MZ user modified")
    with pytest.raises(ValueError, match="modified"):
        mod.install_addon(game, assembly, policies, replace=True)
    (destination / ".orbiter-install.json").write_text(json.dumps({"application": "other"}))
    with pytest.raises(ValueError, match="owned"):
        mod.install_addon(game, assembly, policies, replace=True)
