"""Package our native addon, Python source and validated live policies.

No game, Unity, kRPC or Python dependency binaries are redistributed. Supply a
current OrbiterAutopilot.dll built from the included addon source separately.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

LIVE_MODELS = {"ascent": "transfer_ascent_ppo", "capture": "moon_capture"}
ASSEMBLY_PATH = "GameData/OrbiterAutopilot/Plugins/OrbiterAutopilot.dll"
MANIFEST_PATH = "release-manifest.json"
PATH_MARKERS = (
    b"/" + b"Users" + b"/",
    b"/" + b"home" + b"/",
    b"C:" + bytes([92]) + b"Users" + bytes([92]),
)

INSTALL_TEXT = """Orbiter Autopilot installation

Extract the entire archive into a permanent writable directory outside KSP.
Keep OrbiterEngine beside install.py; moving it later requires reinstalling
the runtime settings. Copying GameData alone does not configure the engine.

Requirements: your own KSP 1.12.5 installation, the separately installed kRPC
mod (tested server 0.5.4), Python 3.12+, internet access and room for PyTorch.
No KSP/Unity/kRPC binaries, Python interpreter or dependency wheels are bundled.

Close KSP before installation. From this extracted directory run:
  python3 install.py --ksp-root "<your KSP directory>"
On Windows use an installed Python 3.12+ interpreter, such as py -3.12.
Optional --python selects the base interpreter; --environment selects where
its virtual environment is created. --port and --rpc-port select local ports.
The installer downloads [train,ksp] dependencies, verifies this archive's
hashes, and configures our prebuilt DLL without Mono or a C# compiler.
--skip-dependencies uses an already prepared environment without pip; it
does not make an unprepared environment usable. --replace updates only an
existing receipt-verified Orbiter installation and preserves its backups.
Install directly from the extracted archive before manually copying GameData;
an unreceipted existing addon directory is deliberately not overwritten.

Normal setup also reads your own KSP Ships/VAB/Orbiter One.craft and Science
Jr.craft templates and generates the supported rocket locally. No stock craft
is distributed. Its new output defaults to Ships/OrbiterRL Transfer.craft in
this extracted release, outside KSP. --craft-output selects another NEW .craft
path; existing files are never overwritten. If stock templates are unavailable
or you already have the supported rocket, use --skip-craft explicitly.
The generator checks all 20 parts, serial engine/decoupler groups, 14310 kg wet
mass, and 6000/4000 kg stage propellant against the released vehicle profile.

Copy the generated OrbiterRL Transfer.craft into your chosen KSP save's
saves/<save name>/Ships/VAB/ directory. Restart KSP, enable its kRPC server,
load OrbiterRL Transfer in the VAB, assign crew and launch onto the pad.
Keep its full tanks and stages 3,2 / fuel groups 2,1, then enter flight.
Click Orbiter in the stock toolbar, choose Mun or Minmus and 40-60 km, then Run.
Other craft and destinations are not qualified by these released live policies.
Closing the window leaves a flight running; Stop waits for control cleanup.

The Python backend runs locally and records flights under OrbiterEngine/runs.
The two supplied policy pairs are the actual-KSP-validated defaults. This does
not establish a repeated-flight success rate or arbitrary-mission reliability.
Hash verification detects changed files; it is not a publisher signature.

License: MIT for this original addon, engine and new policies; see LICENSE.
Legacy class-project material, game assets and third-party runtimes are not
included and are not relicensed. The distribution can be hosted as a ZIP on
SpaceDock or CurseForge; automated mod-manager Python setup is not provided.
"""

INSTALL_WRAPPER = """#!/usr/bin/env python3
\"\"\"Install the extracted Orbiter release with standard-library Python.\"\"\"
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent / "OrbiterEngine/orbiter/release_install.py"), run_name="__main__")
"""

SETTINGS_EXAMPLE = """// install.py writes settings.cfg with your local paths.
// This example is intentionally inactive and is not loaded by KSP.
ORBITER_AUTOPILOT
{
    python = <absolute path to your prepared Python environment>
    project_root = <absolute path to extracted OrbiterEngine>
    port = 8765
    rpc_port = 50000
    auto_start = true
}
"""


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Release input must be an ordinary file: {path.name}")
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f"Release input changed while reading: {path.name}")
    if any(marker in data for marker in PATH_MARKERS):
        raise ValueError(f"Developer home path found in release input: {path.name}")
    return data


def build_release(project_root, assembly, output, *, version=None):
    """Write a new ZIP from an explicit owned-file allowlist; never overwrite."""
    root, assembly, output = Path(project_root).resolve(), Path(assembly), Path(output)
    if assembly.name != "OrbiterAutopilot.dll":
        raise ValueError("Supply our OrbiterAutopilot.dll, not a game/dependency DLL")
    dll = _read(assembly)
    if not dll.startswith(b"MZ"):
        raise ValueError("OrbiterAutopilot.dll is not a PE assembly")
    project = tomllib.loads(_read(root / "pyproject.toml").decode())
    version = project["project"]["version"] if version is None else version
    if not isinstance(version, str) or not re.fullmatch(
        r"[0-9][A-Za-z0-9._+-]{0,63}", version
    ):
        raise ValueError("Release version must be a plain version identifier")
    license_data = _read(root / "LICENSE")
    if b"Copyright (c) 2026 Camilo Correa Llano" not in license_data:
        raise ValueError("The owned MIT license is required")
    validation_data = _read(root / "validation/missions/manifest.json")
    validated = json.loads(validation_data)["models"]
    payload = {
        ASSEMBLY_PATH: dll,
        "LICENSE": license_data,
        "GameData/OrbiterAutopilot/LICENSE": license_data,
        "GameData/OrbiterAutopilot/settings.cfg.example": SETTINGS_EXAMPLE.encode(),
        "INSTALL.txt": INSTALL_TEXT.encode(),
        "install.py": INSTALL_WRAPPER.encode(),
    }
    for relative in ("pyproject.toml", "README.md", "LICENSE"):
        payload["OrbiterEngine/" + relative] = _read(root / relative)
    sources = sorted(
        {
            *root.glob("orbiter/*.py"),
            *root.glob("orbiter/*.cs"),
            *root.glob("orbiter/configs/*.json"),
            *root.glob("orbiter/data/*.json"),
            root / "orbiter/control.html",
        }
    )
    required = {
        "orbiter/release_install.py",
        "orbiter/ksp_autopilot.cs",
        "orbiter/control.py",
        "orbiter/mission_live.py",
        "orbiter/ownership.py",
        "orbiter/data/transfer_vehicle.json",
    }
    if not required <= {path.relative_to(root).as_posix() for path in sources}:
        raise ValueError("Required engine/addon/installer sources are missing")
    for path in sources:
        payload["OrbiterEngine/" + path.relative_to(root).as_posix()] = _read(path)
    models = {}
    for role, stem in LIVE_MODELS.items():
        record = validated[role]
        expected_paths = (f"models/{stem}.zip", f"models/{stem}.metadata.json")
        if (record.get("checkpoint"), record.get("metadata")) != expected_paths:
            raise ValueError(
                "Validation manifest does not identify approved live defaults"
            )
        for relative, key in zip(expected_paths, ("sha256", "metadata_sha256")):
            data = _read(root / relative)
            if hashlib.sha256(data).hexdigest() != record[key]:
                raise ValueError(f"Validated live model hash mismatch: {relative}")
            if relative.endswith(".metadata.json"):
                metadata = json.loads(data)
                if (
                    metadata.get("algorithm") != "ppo"
                    or type(metadata.get("num_timesteps")) is not int
                    or metadata["num_timesteps"] < 0
                ):
                    raise ValueError(
                        "Approved live metadata must identify a PPO policy and its step counter"
                    )
                if role == "capture" and metadata.get("task") != "orbital_capture_v2":
                    raise ValueError("Approved capture model must use the v2 task")
            else:
                with ZipFile(BytesIO(data)) as checkpoint:
                    if any(
                        any(marker in checkpoint.read(name) for marker in PATH_MARKERS)
                        for name in checkpoint.namelist()
                    ):
                        raise ValueError(
                            "Developer home path found inside model archive"
                        )
            payload["OrbiterEngine/" + relative] = data
        models[role] = dict(record)
    manifest = dict(
        schema_version=1,
        application="orbiter-autopilot",
        version=version,
        license="MIT",
        live_models=models,
        validation_manifest_sha256=hashlib.sha256(validation_data).hexdigest(),
        assembly_source_correspondence="Caller supplies the latest assembly built from the included source; packaging hashes do not prove compilation provenance.",
        external_requirements=[
            "KSP 1.12.5",
            "kRPC server (tested 0.5.4)",
            "Python >=3.12",
            "Python train,ksp dependencies installed separately",
        ],
        files={
            name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            for name, data in sorted(payload.items())
        },
    )
    payload[MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(payload.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = (0o755 if name == "install.py" else 0o644) << 16
            bundle.writestr(info, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(archive.getvalue())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    try:
        manifest = build_release(
            args.project_root, args.assembly, args.output, version=args.version
        )
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Release packaging stopped: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            dict(
                version=manifest["version"],
                files=len(manifest["files"]),
                zip_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
