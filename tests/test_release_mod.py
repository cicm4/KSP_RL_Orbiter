"""Release archive contracts for the self-contained addon; no game calls."""

import hashlib
import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pytest

from orbiter import release_mod

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assembly(tmp_path):
    path = tmp_path / "OrbiterAutopilot.dll"
    path.write_bytes(b"MZ owned addon test assembly OrbiterRL.Core")
    return path


@pytest.fixture(scope="module")
def policies(tmp_path_factory):
    from orbiter.native_export import export_release

    folder = tmp_path_factory.mktemp("policies")
    export_release(ROOT, folder)
    return folder


def test_archive_is_gamedata_only_with_dll_policies_and_manifest(tmp_path, assembly, policies):
    archive = tmp_path / "release.zip"
    manifest = release_mod.build_release(ROOT, assembly, archive, policies=policies)
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert names == {
        release_mod.ASSEMBLY_PATH,
        release_mod.MANIFEST_PATH,
        "GameData/OrbiterAutopilot/LICENSE",
        "GameData/OrbiterAutopilot/README.txt",
        "GameData/OrbiterAutopilot/Policies/ascent.policy",
        "GameData/OrbiterAutopilot/Policies/capture.policy",
    }
    assert all(name.startswith("GameData/OrbiterAutopilot/") for name in names)
    assert not any(name.endswith((".py", ".zip", ".craft", ".cfg")) for name in names)
    assert manifest["runtime"] == "native" and manifest["external_requirements"] == ["KSP 1.12.5"]
    validated = json.loads((ROOT / "validation/missions/manifest.json").read_text())["models"]
    for role in ("ascent", "capture"):
        assert manifest["live_models"][role]["sha256"] == validated[role]["sha256"]


def test_archive_is_deterministic_and_never_overwrites(tmp_path, assembly, policies):
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    release_mod.build_release(ROOT, assembly, first, policies=policies)
    release_mod.build_release(ROOT, assembly, second, policies=policies)
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        release_mod.build_release(ROOT, assembly, first, policies=policies)


def test_rejects_wrong_assembly_bridge_build_local_paths_and_foreign_policies(tmp_path, assembly, policies):
    game_dll = tmp_path / "Assembly-CSharp.dll"
    game_dll.write_bytes(b"MZ")
    with pytest.raises(ValueError, match="OrbiterAutopilot.dll"):
        release_mod.build_release(ROOT, game_dll, tmp_path / "x.zip", policies=policies)
    bridge = tmp_path / "bridge/OrbiterAutopilot.dll"
    bridge.parent.mkdir()
    bridge.write_bytes(b"MZ OrbiterRL.Core Bridge")
    with pytest.raises(ValueError, match="native"):
        release_mod.build_release(ROOT, bridge, tmp_path / "y.zip", policies=policies)
    local = tmp_path / "local/OrbiterAutopilot.dll"
    local.parent.mkdir()
    local.write_bytes(b"MZ OrbiterRL.Core /" + b"Users/someone/")
    with pytest.raises(ValueError, match="home path"):
        release_mod.build_release(ROOT, local, tmp_path / "z.zip", policies=policies)
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    (swapped / "ascent.policy").write_bytes((policies / "capture.policy").read_bytes())
    (swapped / "capture.policy").write_bytes((policies / "capture.policy").read_bytes())
    with pytest.raises(ValueError, match="validated checkpoint"):
        release_mod.build_release(ROOT, assembly, tmp_path / "w.zip", policies=swapped)
    assert not any((tmp_path / name).exists() for name in ("x.zip", "y.zip", "z.zip", "w.zip"))


def test_extracted_distribution_verifies_and_tampering_fails(tmp_path, assembly, policies):
    archive = tmp_path / "release.zip"
    release_mod.build_release(ROOT, assembly, archive, version="0.2.0rc1", policies=policies)
    folder = tmp_path / "KSP folder with spaces"
    with ZipFile(archive) as bundle:
        bundle.extractall(folder)
    assert release_mod.verify_distribution(folder)["version"] == "0.2.0rc1"
    readme = (folder / "GameData/OrbiterAutopilot/README.txt").read_text()
    assert "nothing else to install" in readme and "kRPC" in readme
    policy = folder / "GameData/OrbiterAutopilot/Policies/ascent.policy"
    policy.write_bytes(policy.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        release_mod.verify_distribution(folder)


def test_license_scope_and_package_metadata_exclude_historical_material():
    license_data = (ROOT / "LICENSE").read_text()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "Copyright (c) 2026 Camilo Correa Llano" in license_data
    assert "does not relicense legacy/" in license_data
    assert metadata["license"] == "MIT" and metadata["license-files"] == ["LICENSE"]
    assert metadata["scripts"]["orbiter-train-generalized"] == "orbiter.train_generalized:main"


def test_policy_files_round_trip_the_validated_contract(policies):
    header = release_mod._policy_header((policies / "ascent.policy").read_bytes())
    assert header["vehicle.name"] == "OrbiterRL Transfer"
    assert header["vehicle.stage_plan"] == "3,2" and header["vehicle.fuel_stages"] == "2,1"
    zip_sha = hashlib.sha256((ROOT / "models/transfer_ascent_ppo.zip").read_bytes()).hexdigest()
    assert header["source_sha256"] == zip_sha
