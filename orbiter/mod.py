"""Build and install the stock-toolbar KSP autopilot frontend.

KSP/Unity assemblies are referenced from the user's installation and are never
copied into the distribution. The addon starts the configured local Python
engine; its policies and kRPC remain separate runtime dependencies.
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
    "UnityEngine.JSONSerializeModule.dll",
)


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


def settings_text(project_root, python, *, port=8765, rpc_port=50000):
    # Preserve a venv interpreter path: resolving its symlink would select the
    # base interpreter without the installed training and kRPC dependencies.
    python = Path(os.path.abspath(python))
    project_root = Path(project_root).resolve()
    for value in (str(project_root), str(python)):
        if any(char in value for char in "\r\n{}"):
            raise ValueError(
                "Paths with newlines or braces cannot be represented in KSP settings"
            )
    if not python.is_file() or not project_root.is_dir():
        raise ValueError("The Python executable and project directory must exist")
    for value in (port, rpc_port):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65535
        ):
            raise ValueError("Ports must be between 1 and 65535")
    return (
        "ORBITER_AUTOPILOT\n{\n"
        f"    python = {python}\n"
        f"    project_root = {project_root}\n"
        f"    port = {port}\n"
        f"    rpc_port = {rpc_port}\n"
        "    auto_start = true\n}\n"
    )


def compile_addon(ksp_root, output, *, compiler=None):
    managed = managed_directory(ksp_root)
    compiler = compiler or shutil.which("mcs")
    if compiler is None:
        raise ValueError("The Mono C# compiler mcs is required to build the addon")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = resources.files("orbiter").joinpath("ksp_autopilot.cs")
    with resources.as_file(source) as source_path:
        subprocess.run(
            [
                str(compiler),
                "-noconfig",
                "-nostdlib+",
                "-target:library",
                "-optimize+",
                f"-out:{output}",
                *(f"-r:{managed / name}" for name in ASSEMBLIES),
                str(source_path),
            ],
            check=True,
            timeout=120,
        )
    return output


def install_addon(ksp_root, assembly, settings, *, replace=False):
    destination = Path(ksp_root) / "GameData/OrbiterAutopilot"
    dll = destination / "Plugins/OrbiterAutopilot.dll"
    config = destination / "settings.cfg"
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
    else:
        destination.mkdir(parents=True)
    dll.parent.mkdir(parents=True, exist_ok=True)
    # Stage complete files on the destination filesystem before replacing them.
    data = Path(assembly).read_bytes()
    content = {
        dll: data,
        config: settings.encode(),
        receipt: (
            json.dumps(
                dict(
                    application="orbiter-autopilot",
                    installed_at=datetime.now(timezone.utc).isoformat(),
                    dll_sha256=hashlib.sha256(data).hexdigest(),
                    settings_sha256=hashlib.sha256(settings.encode()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        ).encode(),
    }
    if replace:
        backup = (
            destination
            / "backups"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        backup.mkdir(parents=True)
        for path in content:
            if path.is_file():
                # KSP recursively loads DLLs in GameData, including backups.
                shutil.copyfile(path, backup / (path.name + ".backup"))
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
        "--python",
        default=sys.executable,
        help="Absolute Python environment with train and ksp extras",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/mod/OrbiterAutopilot.dll")
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rpc-port", type=int, default=50000)
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
        settings = settings_text(
            args.project_root, args.python, port=args.port, rpc_port=args.rpc_port
        )
        if args.install:
            for model in ("transfer_ascent_ppo", "moon_capture"):
                for suffix in (".zip", ".metadata.json"):
                    path = args.project_root / "models" / (model + suffix)
                    if not path.is_file():
                        raise ValueError(f"Autopilot runtime model is missing: {path}")
        output = compile_addon(args.ksp_root, args.output)
        print(f"Compiled: {output}")
        if args.install:
            destination = install_addon(
                args.ksp_root, output, settings, replace=args.replace
            )
            print(f"Installed: {destination}")
            print("Restart KSP, enter flight, and click the Orbiter toolbar button.")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Cannot install Orbiter Autopilot: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
