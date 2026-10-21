"""Run against an installed wheel outside the checkout, using python -I.

Pass --core-only in a clean environment containing no training or KSP extras.
Pass --with-train to check the training console script as well.
"""
import argparse
from importlib import metadata, resources, util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy as np

from orbiter import (CaptureConfig, CaptureEnv, EnvConfig, HeadlessOrbiterEnv,
                     Stage, SystemCatalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--with-train", action="store_true")
    parser.add_argument("--model", type=Path, action="append", default=[],
                        help="Portable checkpoint to load and predict with; requires --with-train")
    parser.add_argument("--capture-model", type=Path, action="append", default=[],
                        help="Portable goal-conditioned capture checkpoint; requires --with-train")
    args = parser.parse_args()
    if args.core_only and args.with_train:
        parser.error("--core-only and --with-train are mutually exclusive")
    if (args.model or args.capture_model) and not args.with_train:
        parser.error("Checkpoint checks require --with-train")
    assert not {"torch", "stable_baselines3", "sb3_contrib", "krpc"} & sys.modules.keys()
    if args.core_only:
        for name in ("torch", "stable_baselines3", "sb3_contrib", "krpc"):
            assert util.find_spec(name) is None, f"Core environment unexpectedly contains {name}"

    environment = HeadlessOrbiterEnv(EnvConfig(randomization=0))
    observation, _ = environment.reset(seed=0)
    assert observation.shape == environment.observation_space.shape
    assert isinstance(environment.config.stages[0], Stage)
    observation, reward, _, _, _ = environment.step([1., 1.])
    assert np.isfinite(observation).all() and np.isfinite(reward)
    environment.close()

    system = SystemCatalog.load()
    assert system.route('Kerbin', 'Minmus') == ['Kerbin', 'Minmus']
    environment = CaptureEnv(CaptureConfig())
    observation, _ = environment.reset(seed=0)
    assert observation.shape == (27,)
    observation, reward, _, _, _ = environment.step(np.zeros(3))
    assert np.isfinite(observation).all() and np.isfinite(reward)
    environment.close()

    package = resources.files("orbiter")
    config_files = sorted(package.joinpath("configs").iterdir(), key=lambda p: p.name)
    assert {"orbiter_rl_terrier.json", "orbiter_rl_terrier_aero.json"} <= {p.name for p in config_files}
    for item in config_files:
        if item.name.endswith(".json"):
            config = EnvConfig.from_dict(json.loads(item.read_text()))
            assert config.stages
    assert "OrbiterBootstrap" in package.joinpath("ksp_bootstrap.cs").read_text()
    assert "OrbiterAutopilot" in package.joinpath("ksp_autopilot.cs").read_text()
    for module in ("generalized_env.py", "generalized_aero_env.py", "train_generalized.py", "ablation_suite.py", "overnight_suite.py", "ascent_planner.py", "distill_ascent.py", "train_dagger_student.py", "release_mod.py", "release_install.py", "tensor_memo.py", "sequence_policy.py", "batched_capture.py", "train_architectures.py"):
        assert package.joinpath(module).is_file(), module
    assert "Orbiter Autopilot" in package.joinpath("control.html").read_text()
    vehicle_profile = json.loads(package.joinpath("data/transfer_vehicle.json").read_text())
    assert vehicle_profile["supported_profile"] == "OrbiterRL Transfer"
    assert len(vehicle_profile["parts"]) == 20 and len(vehicle_profile["engines"]) == 2

    if args.model:
        from orbiter.live import load_controller
        for checkpoint in args.model:
            controller, config = load_controller(checkpoint)
            env = HeadlessOrbiterEnv(config)
            observation, _ = env.reset(seed=0)
            action, _ = controller.predict(observation, deterministic=True)
            assert action.shape == (2,) and np.isfinite(action).all()
            env.close()

    if args.capture_model:
        from orbiter.train_capture import load_capture_controller
        for checkpoint in args.capture_model:
            controller, contract = load_capture_controller(checkpoint)
            env = CaptureEnv(CaptureConfig.from_dict(contract['environment']))
            observation, _ = env.reset(seed=0)
            action, _ = controller.predict(observation, deterministic=True)
            assert action.shape == (3,) and np.isfinite(action).all()
            env.close()

    distribution = metadata.distribution("ksp-rl-orbiter")
    scripts = {
        entry.name: entry for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    expected = {
        "orbiter-bootstrap", "orbiter-calibrate", "orbiter-capture",
        "orbiter-fit-aero", "orbiter-live", "orbiter-make-craft",
        "orbiter-train", "orbiter-train-capture", "orbiter-train-generalized",
        "orbiter-mission", "orbiter-control", "orbiter-mod", "orbiter-validate",
    }
    assert scripts.keys() == expected
    assert distribution.files
    assert not any(str(item).startswith(("tests/", "runs/", "data/", "old_data/"))
                   for item in distribution.files)
    scripts_dir = Path(sysconfig.get_path("scripts"))
    for name in sorted(expected):
        if name in {"orbiter-train", "orbiter-train-capture", "orbiter-train-generalized"} and not args.with_train:
            continue
        executable = scripts_dir / (name + (".exe" if sys.platform == "win32" else ""))
        result = subprocess.run([str(executable), "--help"], check=True,
                                capture_output=True, text=True, timeout=60)
        assert "usage:" in result.stdout.lower(), (name, result.stdout, result.stderr)
    print(f"Installed wheel smoke passed: public API, {len(config_files)} configurations, C# resource, console scripts, {len(args.model)} ascent / {len(args.capture_model)} capture models")


if __name__ == "__main__":
    main()
