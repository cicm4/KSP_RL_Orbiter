"""Build and install the self-contained KSP autopilot addon.

The addon flies the whole mission inside KSP: no Python, kRPC or external
process at runtime. KSP/Unity assemblies are referenced from the user's
installation to compile it and are never copied into the distribution.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ASSEMBLIES = (
    "mscorlib.dll",
    "System.dll",
    "System.Core.dll",
    "Assembly-CSharp.dll",
    "UnityEngine.dll",
    "UnityEngine.CoreModule.dll",
    "UnityEngine.AnimationModule.dll",
    "UnityEngine.IMGUIModule.dll",
    "UnityEngine.PhysicsModule.dll",
    "UnityEngine.InputLegacyModule.dll",
    "UnityEngine.UI.dll",
)
ADDON = "GameData/OrbiterAutopilot"
POLICIES = ("ascent.policy", "capture.policy")


def managed_directory(ksp_root):
    root = Path(ksp_root)
    for relative in (
        "KSP.app/Contents/Resources/Data/Managed",
        "KSP_x64_Data/Managed",
        "KSP_Data/Managed",
    ):
        candidate = root / relative
        if all((candidate / name).is_file() for name in ASSEMBLIES):
            return candidate
    raise ValueError("KSP 1.12 game/Unity managed assemblies were not found")


def native_sources():
    """Sorted addon sources: the Unity-free core and the KSP adapter."""
    root = resources.files("orbiter").joinpath("native")
    sources = []
    for folder in ("Core", "Ksp"):
        directory = root.joinpath(folder)
        sources.extend(sorted((item for item in directory.iterdir() if item.name.endswith(".cs")),
                              key=lambda item: item.name))
    if not sources:
        raise ValueError("Native addon sources are missing from the package")
    return sources


def compile_addon(ksp_root, output, *, compiler=None):
    managed = managed_directory(ksp_root)
    compiler = compiler or shutil.which("mcs")
    if compiler is None:
        raise ValueError("The Mono C# compiler mcs is required to build the addon")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, source in enumerate(native_sources()):
            path = Path(directory) / f"{index:02d}_{source.name}"
            path.write_bytes(source.read_bytes())
            paths.append(str(path))
        subprocess.run(
            [
                str(compiler),
                "-noconfig",
                "-nostdlib+",
                "-langversion:7",
                "-target:library",
                "-optimize+",
                "-deterministic",
                f"-out:{output}",
                *(f"-r:{managed / name}" for name in ASSEMBLIES),
                *paths,
            ],
            check=True,
            timeout=180,
        )
    return output


def install_addon(ksp_root, assembly, policies, *, replace=False):
    """Install the DLL and exported policies under an ownership receipt."""
    destination = Path(ksp_root) / ADDON
    dll = destination / "Plugins/OrbiterAutopilot.dll"
    receipt = destination / ".orbiter-install.json"
    if destination.exists():
        if not replace or not receipt.is_file():
            raise ValueError(
                "Existing addon directory requires --replace and an Orbiter install receipt"
            )
        previous = json.loads(receipt.read_text())
        if previous.get("application") != "orbiter-autopilot" or not dll.is_file():
            raise ValueError("Existing addon is not an owned Orbiter installation")
        if hashlib.sha256(dll.read_bytes()).hexdigest() != previous.get("dll_sha256"):
            raise ValueError("Installed DLL was modified; refusing to overwrite it")
    content = {dll: Path(assembly).read_bytes()}
    for name in POLICIES:
        content[destination / "Policies" / name] = (Path(policies) / name).read_bytes()
    content[receipt] = (
        json.dumps(
            dict(
                application="orbiter-autopilot",
                runtime="native",
                installed_at=datetime.now(timezone.utc).isoformat(),
                dll_sha256=hashlib.sha256(content[dll]).hexdigest(),
                policy_sha256={name: hashlib.sha256(content[destination / "Policies" / name]).hexdigest()
                               for name in POLICIES},
            ),
            indent=2,
        )
        + "\n"
    ).encode()
    if replace:
        # Outside GameData: KSP recursively loads every DLL it finds there.
        backup = (Path(ksp_root) / "OrbiterAutopilot-backups"
                  / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
        backup.mkdir(parents=True)
        for path in content:
            if path.is_file():
                shutil.copyfile(path, backup / path.name)
    for path in content:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Stage complete files on the destination filesystem before replacing them.
    for path, payload in content.items():
        descriptor, temporary = tempfile.mkstemp(prefix=".orbiter-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ksp-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("runs/mod/OrbiterAutopilot.dll")
    )
    parser.add_argument("--install", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Update only a receipt-verified Orbiter installation, preserving a backup",
    )
    args = parser.parse_args(argv)
    if args.replace and not args.install:
        parser.error("--replace requires --install")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        output = compile_addon(args.ksp_root, args.output)
        print(f"Compiled: {output}")
        if args.install:
            from .native_export import export_release

            with tempfile.TemporaryDirectory() as policies:
                export_release(args.project_root, policies)
                destination = install_addon(args.ksp_root, output, policies, replace=args.replace)
            print(f"Installed: {destination}")
            print("Restart KSP, enter flight, and click the Orbiter toolbar button.")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Cannot install Orbiter Autopilot: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
