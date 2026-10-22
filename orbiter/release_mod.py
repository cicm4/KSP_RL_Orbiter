"""Package the self-contained KSP addon as a drop-in GameData ZIP.

The archive holds only our DLL, the two exported policies, the license and a
readme. Players extract it into their KSP folder; nothing else is installed.
No game, Unity, kRPC or Python binaries are redistributed, and no stock-derived
craft: the addon builds the supported rocket from the player's own stock craft.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import tempfile
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

LIVE_MODELS = {"ascent": "transfer_ascent_ppo", "capture": "moon_capture"}
# Any-rocket policies; simulation results travel with them in the manifest.
GENERALIZED_MODELS = {"generalized_ascent": "generalized_ascent_gru", "generalized_capture": "generalized_capture_lstm"}
ADDON = "GameData/OrbiterAutopilot"
ASSEMBLY_PATH = ADDON + "/Plugins/OrbiterAutopilot.dll"
MANIFEST_PATH = ADDON + "/release-manifest.json"
PATH_MARKERS = (
    b"/" + b"Users" + b"/",
    b"/" + b"home" + b"/",
    b"C:" + bytes([92]) + b"Users" + bytes([92]),
)

README_TEXT = """Orbiter Autopilot {version}

Flies your rocket from the Kerbin launchpad to orbit, and on to the Mun or
Minmus, by itself. The flying is done by neural networks that learned to
launch, stage and brake into orbit. Nothing else to install: no other mods,
no Python, no setup.

Install
  Unzip into your KSP 1.12 folder so GameData/OrbiterAutopilot sits next to
  GameData/Squad. Start KSP.

Fly
  1. Put a rocket on the launchpad (see "Which rockets" below), or use the
     ready-made "OrbiterRL Transfer", which the mod builds from your stock
     Orbiter One and Science Jr craft. Find it in the VAB under stock vessels.
  2. Click the Orbiter button on the toolbar.
  3. Pick Kerbin orbit, Mun or Minmus and an altitude, then press Run.
  4. Press Stop at any time to take back control.

Which rockets
  Up to four stages stacked on top of each other, liquid engines only, each
  stage dropping its engine and tanks when empty. No solid boosters, side
  boosters or fuel lines. It needs a launch thrust-to-weight above about 1.2
  and enough fuel for orbit. The window tells you if your rocket can't be flown.

Good to know
  - OrbiterRL Transfer to the Mun or Minmus uses the original policies, which
    have flown full missions in real KSP. Every other rocket uses the newer
    "any rocket" policies: they have reached orbit in real KSP with a few
    test rockets, but they are new, so expect the odd failure.
  - Career mode needs maneuver nodes unlocked for Mun and Minmus trips.
  - Each flight is logged in GameData/OrbiterAutopilot/Logs.

License: MIT (see LICENSE). Game assets and other mods are not included.
"""


def _read(path):
    path = Path(path)
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


def _policy_header(data):
    if not data.startswith((b"ORBPOL1\n", b"ORBSEQ1\n")):
        raise ValueError("Not an Orbiter policy file")
    length = int.from_bytes(data[8:12], "little")
    text = data[12:12 + length].decode()
    return dict(line.split("=", 1) for line in text.splitlines() if line)


def build_release(project_root, assembly, output, *, version=None, policies=None):
    """Write a new ZIP from an explicit owned-file allowlist; never overwrite."""
    root, assembly, output = Path(project_root).resolve(), Path(assembly), Path(output)
    if assembly.name != "OrbiterAutopilot.dll":
        raise ValueError("Supply our OrbiterAutopilot.dll, not a game/dependency DLL")
    dll = _read(assembly)
    if not dll.startswith(b"MZ"):
        raise ValueError("OrbiterAutopilot.dll is not a PE assembly")
    if b"OrbiterRL.Core" not in dll or b"Bridge" in dll:
        raise ValueError("Supply the self-contained native addon build")
    project = tomllib.loads(_read(root / "pyproject.toml").decode())
    version = project["project"]["version"] if version is None else version
    if not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9._+-]{0,63}", version):
        raise ValueError("Release version must be a plain version identifier")
    license_data = _read(root / "LICENSE")
    if b"Copyright (c) 2026 Camilo Correa Llano" not in license_data:
        raise ValueError("The owned MIT license is required")
    validation_data = _read(root / "validation/missions/manifest.json")
    validated = json.loads(validation_data)["models"]
    with tempfile.TemporaryDirectory() as exported:
        if policies is None:
            from .native_export import export_release

            export_release(root, exported)
            policies = exported
        payload = {
            ASSEMBLY_PATH: dll,
            ADDON + "/LICENSE": license_data,
            ADDON + "/README.txt": README_TEXT.format(version=version).encode(),
        }
        models = {}
        for role, stem in LIVE_MODELS.items():
            record = validated[role]
            if record.get("checkpoint") != f"models/{stem}.zip":
                raise ValueError("Validation manifest does not identify approved live defaults")
            checkpoint = _read(root / record["checkpoint"])
            if hashlib.sha256(checkpoint).hexdigest() != record["sha256"]:
                raise ValueError(f"Validated live model hash mismatch: {record['checkpoint']}")
            data = _read(Path(policies) / f"{role}.policy")
            header = _policy_header(data)
            if header.get("kind") != role or header.get("source_sha256") != record["sha256"]:
                raise ValueError(f"{role}.policy was not exported from the validated checkpoint")
            payload[f"{ADDON}/Policies/{role}.policy"] = data
            models[role] = dict(record, policy_sha256=hashlib.sha256(data).hexdigest())
        for role, stem in GENERALIZED_MODELS.items():
            checkpoint = _read(root / f"models/{stem}.pt")
            summary = json.loads(_read(root / f"models/{stem}.summary.json"))
            data = _read(Path(policies) / f"{role}.policy")
            header = _policy_header(data)
            if header.get("kind") != role or header.get("source_sha256") != hashlib.sha256(checkpoint).hexdigest():
                raise ValueError(f"{role}.policy was not exported from models/{stem}.pt")
            payload[f"{ADDON}/Policies/{role}.policy"] = data
            comparison = summary["comparison"]
            models[role] = dict(checkpoint=f"models/{stem}.pt", sha256=hashlib.sha256(checkpoint).hexdigest(),
                                policy_sha256=hashlib.sha256(data).hexdigest(),
                                simulation_test=dict(episodes=comparison["episodes"], successes=comparison["successes"],
                                                     per_body={k: v["successes"] for k, v in comparison["per_body"].items()}),
                                live_ksp_validated=False)
    manifest = dict(
        schema_version=2,
        application="orbiter-autopilot",
        runtime="native",
        version=version,
        license="MIT",
        live_models=models,
        validation_manifest_sha256=hashlib.sha256(validation_data).hexdigest(),
        assembly_source_correspondence="Caller supplies the assembly built from the included source; packaging hashes do not prove compilation provenance.",
        external_requirements=["KSP 1.12.5"],
        files={
            name: dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            for name, data in sorted(payload.items())
        },
    )
    payload[MANIFEST_PATH] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    archive = BytesIO()
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(payload.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(archive.getvalue())
    return manifest


def verify_distribution(directory):
    """Check an extracted archive (or KSP folder) against its manifest hashes."""
    root = Path(directory)
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    for name, record in manifest["files"].items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
            raise ValueError(f"Release file changed: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    try:
        manifest = build_release(args.project_root, args.assembly, args.output, version=args.version)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"Release packaging stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(dict(version=manifest["version"], files=len(manifest["files"]),
                          zip_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
