"""Standard-library installer for an extracted Orbiter Autopilot release.

This file also runs directly, without importing the dependency-bearing package.
It never launches KSP, starts an engine or connects to kRPC.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

DLL = "GameData/OrbiterAutopilot/Plugins/OrbiterAutopilot.dll"
REQUIRED = {
    DLL,
    "LICENSE",
    "OrbiterEngine/pyproject.toml",
    "OrbiterEngine/orbiter/mod.py",
    "OrbiterEngine/orbiter/release_install.py",
    "OrbiterEngine/orbiter/control.py",
    "OrbiterEngine/orbiter/make_craft.py",
    "OrbiterEngine/orbiter/data/transfer_vehicle.json",
    "OrbiterEngine/models/transfer_ascent_ppo.zip",
    "OrbiterEngine/models/transfer_ascent_ppo.metadata.json",
    "OrbiterEngine/models/moon_capture.zip",
    "OrbiterEngine/models/moon_capture.metadata.json",
}


def verify_distribution(distribution):
    """Check safe relative paths and every shipped file before any installation."""
    root = Path(distribution).resolve()
    manifest = json.loads((root / "release-manifest.json").read_text())
    if (
        manifest.get("application") != "orbiter-autopilot"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("Not a supported Orbiter Autopilot release")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not REQUIRED <= files.keys():
        raise ValueError("Release manifest omits required addon/engine/model files")
    for name, record in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("Release manifest contains an unsafe file path")
        path = root.joinpath(*relative.parts)
        if (
            not path.resolve().is_relative_to(root)
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(f"Missing or escaped release file: {name}")
        data = path.read_bytes()
        if len(data) != record.get("bytes") or hashlib.sha256(
            data
        ).hexdigest() != record.get("sha256"):
            raise ValueError(f"Release file failed integrity verification: {name}")
    return manifest


def _mod_module(engine):
    spec = importlib.util.spec_from_file_location(
        "orbiter_release_mod_installer", engine / "orbiter/mod.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_transfer_craft(engine, ksp_root):
    """Render from local templates, then bind the known stock mass/stage profile."""
    engine, game = Path(engine), Path(ksp_root)
    sources = [
        game / "Ships/VAB" / name for name in ("Orbiter One.craft", "Science Jr.craft")
    ]
    if not all(path.is_file() for path in sources):
        raise ValueError(
            "Stock Orbiter One.craft and Science Jr.craft are required; use --skip-craft to install without generating the rocket"
        )
    spec = importlib.util.spec_from_file_location(
        "orbiter_release_craft", engine / "orbiter/make_craft.py"
    )
    craft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(craft)
    inputs = [path.read_bytes() for path in sources]
    result = craft.make_transfer_craft(
        craft.make_terrier_craft(*(data.decode(errors="replace") for data in inputs))
    )
    _, blocks = craft._split_parts(result)
    parts = {craft._part_id(block): block for block in blocks}
    names = {identifier: identifier.rsplit("_", 1)[0] for identifier in parts}
    profile = json.loads((engine / "orbiter/data/transfer_vehicle.json").read_text())
    if len(parts) != len(blocks) or sorted(names.values()) != sorted(profile["parts"]):
        raise ValueError("Generated craft does not match the supported 20-part profile")
    stages = {
        identifier: int(re.search(r"^\tistg = (-?\d+)$", block, re.MULTILINE).group(1))
        for identifier, block in parts.items()
    }
    parents = {}
    for identifier, block in parts.items():
        for child in re.findall(r"^\tlink = (\S+)$", block, re.MULTILINE):
            if child not in parts or child in parents:
                raise ValueError("Generated craft has an invalid attachment tree")
            parents[child] = identifier
    roots = set(parts) - parents.keys()
    if len(roots) != 1 or names[next(iter(roots))] != "mk1pod.v2":
        raise ValueError("Generated craft must retain its single crew-pod root")

    def group(identifier):
        seen = set()
        while True:
            if identifier in seen:
                raise ValueError("Generated craft has a cyclic attachment tree")
            seen.add(identifier)
            if names[identifier] == "Decoupler.1":
                return stages[identifier]
            if identifier not in parents:
                return None
            identifier = parents[identifier]

    actual_engines = sorted(
        (names[item], stages[item], group(item))
        for item in parts
        if names[item].startswith("liquidEngine")
    )
    expected_engines = sorted(
        (item["name"], item["activation_stage"], item["decouple_stage"])
        for item in profile["engines"]
    )
    actual_decouplers = sorted(
        stages[item] for item in parts if names[item] == "Decoupler.1"
    )
    if actual_engines != expected_engines or actual_decouplers != [1, 2]:
        raise ValueError(
            "Generated engine/decoupler stages differ from the supported profile"
        )
    # Nominal stock dry masses of the exact checked part multiset, in kg.
    dry = {
        "mk1pod.v2": 800.0,
        "Decoupler.1": 40.0,
        "fuelTankSmall": 125.0,
        "liquidEngine3.v2": 500.0,
        "liquidEngine2": 1500.0,
        "basicFin": 10.0,
        "parachuteSingle": 100.0,
    }
    inert, fuel = {None: 0.0, 1: 0.0, 2: 0.0}, {1: 0.0, 2: 0.0}
    for identifier, block in parts.items():
        bucket = group(identifier)
        if bucket not in inert:
            raise ValueError("Generated craft has an unexpected fuel/decouple group")
        inert[bucket] += dry[names[identifier]]
        for resource in re.findall(r"\tRESOURCE\n\t\{\n(.*?)\n\t\}", block, re.DOTALL):
            fields = dict(re.findall(r"^\t\t(\w+) = ([^\n]+)$", resource, re.MULTILINE))
            amount = float(fields["amount"])
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Generated craft has invalid propellant")
            if fields["name"] in {"LiquidFuel", "Oxidizer"}:
                if bucket not in fuel:
                    raise ValueError(
                        "Generated craft has propellant outside the staged tanks"
                    )
                fuel[bucket] += amount * 5.0
            elif fields["name"] == "MonoPropellant":
                inert[bucket] += amount * 4.0
    expected = profile["vehicle_contract"]
    stage_groups = profile["fuel_stages"]
    if inert[None] != expected["payload_mass"] or any(
        inert[bucket] != stage["dry_mass"] or fuel[bucket] != stage["fuel_mass"]
        for bucket, stage in zip(stage_groups, expected["stages"])
    ):
        raise ValueError(
            "Generated craft propellant/dry mass profile differs from the supported model"
        )
    return result, dict(
        part_count=len(parts),
        stage_plan=profile["stage_plan"],
        fuel_stages=stage_groups,
        stage_propellant_masses=[fuel[bucket] for bucket in stage_groups],
        wet_mass=sum(inert.values()) + sum(fuel.values()),
        source_sha256={
            path.name: hashlib.sha256(data).hexdigest()
            for path, data in zip(sources, inputs)
        },
    )


def _write_craft(output, text, profile):
    output = Path(output).absolute()
    if output.suffix.lower() != ".craft":
        raise ValueError("--craft-output must name a new .craft file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return dict(
        profile, output=str(output), sha256=hashlib.sha256(text.encode()).hexdigest()
    )


def generate_transfer_craft(engine, ksp_root, output):
    """Generate one new local craft; stock inputs and existing outputs are immutable."""
    text, profile = _prepare_transfer_craft(engine, ksp_root)
    return _write_craft(output, text, profile)


def install_distribution(
    distribution,
    ksp_root,
    *,
    python=sys.executable,
    environment=None,
    port=8765,
    rpc_port=50000,
    replace=False,
    skip_dependencies=False,
    craft_output=None,
    skip_craft=False,
):
    """Prepare a user-owned Python environment, then install the supplied DLL."""
    root, game = Path(distribution).resolve(), Path(ksp_root).resolve()
    manifest = verify_distribution(root)
    if not (game / "GameData").is_dir():
        raise ValueError("--ksp-root must contain the existing KSP GameData directory")
    game_data = (game / "GameData").resolve()
    engine = root / "OrbiterEngine"
    if skip_craft and craft_output is not None:
        raise ValueError("--skip-craft and --craft-output cannot be combined")
    craft_output = (
        Path(craft_output).absolute()
        if craft_output
        else root / "Ships/OrbiterRL Transfer.craft"
    )
    prepared_craft = None
    if not skip_craft:
        if craft_output.suffix.lower() != ".craft":
            raise ValueError("--craft-output must name a new .craft file")
        if craft_output.exists():
            raise FileExistsError(
                f"Refusing to overwrite generated craft: {craft_output}; choose another --craft-output or use --skip-craft"
            )
        prepared_craft = _prepare_transfer_craft(engine, game)
    environment = Path(environment).absolute() if environment else engine / ".venv"
    if engine.resolve().is_relative_to(
        game_data
    ) or environment.resolve().is_relative_to(game_data):
        raise ValueError("Keep the Python engine/environment outside KSP GameData")
    base_python = Path(os.path.abspath(python))
    if not base_python.is_file():
        raise ValueError("--python must name an existing Python 3.12+ executable")
    env_python = environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if skip_dependencies and not (
        env_python.is_file() and (environment / "pyvenv.cfg").is_file()
    ):
        raise ValueError(
            "--skip-dependencies requires an already prepared virtual environment"
        )
    if environment.exists():
        if not (env_python.is_file() and (environment / "pyvenv.cfg").is_file()):
            raise ValueError(
                "Refusing to modify a directory that is not a Python virtual environment"
            )
    else:
        subprocess.run(
            [str(base_python), "-m", "venv", str(environment)], check=True, timeout=120
        )
    if not skip_dependencies:
        subprocess.run(
            [str(env_python), "-m", "pip", "install", str(engine) + "[train,ksp]"],
            check=True,
            timeout=1800,
        )
    # This verifies dependency/model compatibility without any KSP/RPC access.
    probe = (
        "import sys; from pathlib import Path; "
        "assert sys.version_info >= (3,12), 'Python 3.12+ is required'; "
        "from orbiter.checkpoints import load_controller; "
        "from orbiter.train_capture import load_capture_controller; "
        "import krpc; root=Path(sys.argv[1]); "
        "load_controller(root/'models/transfer_ascent_ppo.zip'); "
        "load_capture_controller(root/'models/moon_capture.zip')"
    )
    subprocess.run(
        [str(env_python), "-c", probe, str(engine)], check=True, timeout=180, cwd=engine
    )
    # Recheck immutable payloads after dependency setup, before touching GameData.
    verify_distribution(root)
    mod = _mod_module(engine)
    settings = mod.settings_text(engine, env_python, port=port, rpc_port=rpc_port)
    generated = _write_craft(craft_output, *prepared_craft) if prepared_craft else None
    destination = mod.install_addon(game, root / DLL, settings, replace=replace)
    shutil.copyfile(root / "LICENSE", destination / "LICENSE")
    return dict(
        application="orbiter-autopilot",
        version=manifest["version"],
        installed=str(destination),
        project_root=str(engine),
        python=str(env_python),
        craft=generated,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ksp-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--skip-dependencies", action="store_true")
    craft_options = parser.add_mutually_exclusive_group()
    craft_options.add_argument(
        "--craft-output",
        type=Path,
        help="New .craft file; default extracted release Ships/OrbiterRL Transfer.craft",
    )
    craft_options.add_argument(
        "--skip-craft",
        action="store_true",
        help="Skip local stock-template craft generation",
    )
    args = parser.parse_args(argv)
    try:
        result = install_distribution(
            Path(__file__).resolve().parents[2],
            args.ksp_root,
            python=args.python,
            environment=args.environment,
            port=args.port,
            rpc_port=args.rpc_port,
            replace=args.replace,
            skip_dependencies=args.skip_dependencies,
            craft_output=args.craft_output,
            skip_craft=args.skip_craft,
        )
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"Orbiter installation stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    if result["craft"]:
        print(
            f"Copy {result['craft']['output']} into your chosen KSP save's Ships/VAB directory, load it in the VAB, assign crew, and launch onto the pad."
        )
    print("Restart KSP, enable kRPC, then open the Orbiter toolbar window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
