"""Release archives and installer contracts; no game/RPC or real setup calls."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile

import pytest

from orbiter import release_install, release_mod

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assembly(tmp_path):
    path = tmp_path / "OrbiterAutopilot.dll"
    path.write_bytes(b"MZ owned addon test assembly")
    return path


@pytest.fixture
def distribution(tmp_path, assembly):
    archive = tmp_path / "OrbiterAutopilot.zip"
    release_mod.build_release(ROOT, assembly, archive, version="0.1.0rc1")
    folder = tmp_path / "extracted release with spaces"
    with ZipFile(archive) as bundle:
        bundle.extractall(folder)
    return folder


def test_archive_contains_only_owned_engine_addon_and_validated_default_models(
    tmp_path, assembly
):
    archive = tmp_path / "release.zip"
    manifest = release_mod.build_release(ROOT, assembly, archive)
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert release_install.REQUIRED <= names
        assert {name for name in names if name.lower().endswith(".dll")} == {
            release_mod.ASSEMBLY_PATH
        }
        assert not any(
            any(
                part in name.lower()
                for part in (
                    "legacy/",
                    ".git/",
                    ".venv/",
                    "stock_capture_experimental",
                    "credential",
                    ".env",
                )
            )
            for name in names
        )
        assert not any("settings.cfg" == Path(name).name for name in names)
        assert "GameData/OrbiterAutopilot/settings.cfg.example" in names
        for name, record in manifest["files"].items():
            data = bundle.read(name)
            assert record == dict(
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
            )
            assert not any(marker in data for marker in release_mod.PATH_MARKERS)
        assert (
            manifest["files"]["OrbiterEngine/orbiter/ksp_autopilot.cs"]["sha256"]
            == hashlib.sha256(
                (ROOT / "orbiter/ksp_autopilot.cs").read_bytes()
            ).hexdigest()
        )
        approved = json.loads((ROOT / "validation/missions/manifest.json").read_text())[
            "models"
        ]
        assert manifest["live_models"] == {
            role: approved[role] for role in ("ascent", "capture")
        }
        assert b"Camilo Correa Llano" in bundle.read(
            "GameData/OrbiterAutopilot/LICENSE"
        )


def test_archive_is_deterministic_and_never_overwrites(tmp_path, assembly):
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    release_mod.build_release(ROOT, assembly, first)
    release_mod.build_release(ROOT, assembly, second)
    assert first.read_bytes() == second.read_bytes()
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        release_mod.build_release(ROOT, assembly, first)
    assert first.read_bytes() == before


@pytest.mark.parametrize("change", ["model", "source_path", "game_dll"])
def test_unapproved_models_local_paths_and_game_dll_inputs_are_rejected(
    tmp_path, assembly, change
):
    project = tmp_path / "source"
    project.mkdir()
    for relative in ("orbiter", "models", "validation/missions"):
        shutil.copytree(ROOT / relative, project / relative)
    for name in ("LICENSE", "pyproject.toml", "README.md"):
        shutil.copyfile(ROOT / name, project / name)
    if change == "model":
        (project / "models/moon_capture.zip").write_bytes(b"changed checkpoint")
    elif change == "source_path":
        (project / "orbiter/local.py").write_text(
            "path = '" + "/" + "Users/developer/private'\n"
        )
    else:
        assembly = tmp_path / "Assembly-CSharp.dll"
        assembly.write_bytes(b"MZ game assembly")
    with pytest.raises(ValueError, match="hash mismatch|home path|not a game"):
        release_mod.build_release(project, assembly, tmp_path / "rejected.zip")
    assert not (tmp_path / "rejected.zip").exists()


def test_extracted_distribution_verifies_and_tampering_fails_before_setup(
    distribution, monkeypatch
):
    assert release_install.verify_distribution(distribution)["version"] == "0.1.0rc1"
    (distribution / release_mod.ASSEMBLY_PATH).write_bytes(b"changed addon")
    monkeypatch.setattr(
        release_install.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Tampered release ran setup"),
    )
    with pytest.raises(ValueError, match="integrity"):
        release_install.install_distribution(
            distribution, distribution / "nonexistent game"
        )


def test_extracted_installer_help_needs_only_the_standard_library(distribution):
    result = subprocess.run(
        [sys.executable, "-S", str(distribution / "install.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "--ksp-root" in result.stdout and "--environment" in result.stdout


def test_manifest_cannot_escape_distribution(distribution):
    path = distribution / "release-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = dict(bytes=0, sha256="0" * 64)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        release_install.verify_distribution(distribution)


def test_installer_uses_supplied_dll_user_paths_and_existing_environment(
    distribution, tmp_path, monkeypatch
):
    game = tmp_path / "user selected game"
    (game / "GameData").mkdir(parents=True)
    environment = tmp_path / "user selected environment"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fake prepared interpreter")
    (environment / "pyvenv.cfg").write_text("test environment")
    calls = []
    monkeypatch.setattr(
        release_install.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    result = release_install.install_distribution(
        distribution,
        game,
        environment=environment,
        skip_dependencies=True,
        skip_craft=True,
        port=8877,
        rpc_port=50001,
    )
    destination = game / "GameData/OrbiterAutopilot"
    assert (destination / "Plugins/OrbiterAutopilot.dll").read_bytes() == (
        distribution / release_mod.ASSEMBLY_PATH
    ).read_bytes()
    settings = (destination / "settings.cfg").read_text()
    assert (
        f"python = {python}" in settings
        and f"project_root = {distribution / 'OrbiterEngine'}" in settings
    )
    assert "port = 8877" in settings and "rpc_port = 50001" in settings
    assert (destination / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert result["python"] == str(python)
    assert len(calls) == 1 and calls[0][0][:2] == [str(python), "-c"]
    assert "krpc.connect" not in calls[0][0][2]
    assert "compile" not in repr(calls) and "mcs" not in repr(calls)


def test_installer_prepares_dependencies_but_does_not_start_backend(
    distribution, tmp_path, monkeypatch, stock_templates
):
    game = stock_templates
    environment = tmp_path / "new runtime"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            python.parent.mkdir(parents=True)
            python.write_bytes(b"new prepared interpreter")
            (environment / "pyvenv.cfg").write_text("test")

    monkeypatch.setattr(release_install.subprocess, "run", run)
    result = release_install.install_distribution(
        distribution, game, environment=environment
    )
    assert calls[0] == [sys.executable, "-m", "venv", str(environment)]
    assert calls[1][1:4] == ["-m", "pip", "install"]
    assert calls[1][-1] == str(distribution / "OrbiterEngine") + "[train,ksp]"
    assert all("orbiter.control" not in command for command in calls)
    assert result["craft"]["wet_mass"] == 14310
    assert (
        Path(result["craft"]["output"])
        == distribution / "Ships/OrbiterRL Transfer.craft"
    )


def test_license_scope_and_package_metadata_exclude_historical_material():
    license_data = (ROOT / "LICENSE").read_text()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "Copyright (c) 2026 Camilo Correa Llano" in license_data
    assert "does not relicense legacy/" in license_data
    assert metadata["license"] == "MIT" and metadata["license-files"] == ["LICENSE"]
    assert (
        metadata["scripts"]["orbiter-train-generalized"]
        == "orbiter.train_generalized:main"
    )


@pytest.fixture
def stock_templates(tmp_path):
    # Reuse the existing synthetic stack, never a redistributed stock craft.
    from tests.test_make_craft import terrier_source, part

    source = terrier_source.__wrapped__()
    source = source.replace("ship = OrbiterRL Terrier", "ship = Synthetic Orbiter One")
    source = source.replace("liquidEngine3.v2_5", "liquidEngine2_5")
    source = source.replace(
        "part = liquidEngine2_13\n\tistg = 3", "part = liquidEngine2_13\n\tistg = 5"
    )
    source = source.replace(
        "part = Decoupler.1_6\n\tistg = 2", "part = Decoupler.1_6\n\tistg = 3"
    )
    _, blocks = __import__(
        "orbiter.make_craft", fromlist=["_split_parts"]
    )._split_parts(source)
    blocks[0] = (
        blocks[0][:-2]
        + "\tRESOURCE\n\t{\n\t\tname = MonoPropellant\n\t\tamount = 10\n\t}\n}\n"
    )
    source = "ship = Synthetic Orbiter One\n" + "".join(blocks)
    source += "".join(
        part(identifier, 4, 0)
        for identifier in ("radialDecoupler_90", "solidBooster.v2_91", "noseCone_92")
    )
    template = "ship = Synthetic Science Jr\n" + part(
        "liquidEngine3.v2_93",
        2,
        0,
        parent="fuelTankSmall_94",
        child="Decoupler.1_95",
        top=0,
        bottom=-0.8,
    )
    game = tmp_path / "synthetic game"
    (game / "GameData").mkdir(parents=True)
    folder = game / "Ships/VAB"
    folder.mkdir(parents=True)
    (folder / "Orbiter One.craft").write_text(source)
    (folder / "Science Jr.craft").write_text(template)
    return game


def test_local_craft_generation_matches_supported_profile_without_touching_stock(
    distribution, stock_templates, tmp_path
):
    source_dir = stock_templates / "Ships/VAB"
    originals = {p.name: p.read_bytes() for p in source_dir.iterdir()}
    output = tmp_path / "player selected craft/OrbiterRL Transfer.craft"
    result = release_install.generate_transfer_craft(
        distribution / "OrbiterEngine", stock_templates, output
    )
    assert output.is_file() and result["part_count"] == 20
    assert result["wet_mass"] == 14310
    assert result["stage_plan"] == [3, 2] and result["fuel_stages"] == [2, 1]
    assert result["stage_propellant_masses"] == [6000, 4000]
    assert output.read_text().startswith("ship = OrbiterRL Transfer\n")
    assert {p.name: p.read_bytes() for p in source_dir.iterdir()} == originals
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        release_install.generate_transfer_craft(
            distribution / "OrbiterEngine", stock_templates, output
        )
    assert output.read_bytes() == previous


def test_missing_stock_template_explains_skip_craft_and_writes_nothing(
    distribution, stock_templates, tmp_path
):
    (stock_templates / "Ships/VAB/Science Jr.craft").unlink()
    output = tmp_path / "not-created.craft"
    with pytest.raises(ValueError, match="--skip-craft"):
        release_install.generate_transfer_craft(
            distribution / "OrbiterEngine", stock_templates, output
        )
    assert not output.exists()


def test_craft_generation_rejects_wrong_fuel_profile(
    distribution, stock_templates, tmp_path
):
    source = stock_templates / "Ships/VAB/Orbiter One.craft"
    source.write_text(source.read_text().replace("amount = 90", "amount = 80"))
    with pytest.raises(ValueError, match="full stock|propellant"):
        release_install.generate_transfer_craft(
            distribution / "OrbiterEngine", stock_templates, tmp_path / "invalid.craft"
        )


@pytest.mark.skipif(os.name == "nt", reason="Unix GameData symlink containment")
@pytest.mark.parametrize("nested", ["environment", "engine"])
def test_symlinked_gamedata_rejects_runtime_inside_real_target_before_setup(
    distribution, tmp_path, monkeypatch, nested
):
    game, actual = tmp_path / "game", tmp_path / "actual GameData"
    game.mkdir()
    actual.mkdir()
    (game / "GameData").symlink_to(actual, target_is_directory=True)
    environment = tmp_path / "outside environment"
    if nested == "environment":
        environment = actual / "forbidden environment"
    else:
        copied = actual / "forbidden release"
        shutil.copytree(distribution, copied)
        distribution = copied
    before = sorted(str(path.relative_to(actual)) for path in actual.rglob("*"))
    monkeypatch.setattr(
        release_install.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Unsafe runtime triggered setup"),
    )
    with pytest.raises(ValueError, match="outside KSP GameData"):
        release_install.install_distribution(
            distribution, game, environment=environment, skip_craft=True
        )
    assert not environment.exists()
    assert sorted(str(path.relative_to(actual)) for path in actual.rglob("*")) == before
