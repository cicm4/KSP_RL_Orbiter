"""Export released SB3 policies to the self-contained KSP addon's weight format.

The native addon evaluates the deterministic actor only:
``clip(action_net(tanh(W2 tanh(W1 obs + b1) + b2)), -1, 1)``. A ``.policy``
file is a short ``key=value`` text header followed by little-endian float32
layers, so the addon needs neither Python nor a JSON parser at runtime. The
header carries the contract constants the C# observation builders use; the
exporter takes them from the checkpoint metadata, never from defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np

MAGIC = b"ORBPOL1\n"
ACTOR = ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net")

ASCENT_KEYS = ("mu", "radius", "atmosphere_depth", "target_altitude", "orbit_tolerance",
               "max_time", "control_dt")
CAPTURE_KEYS = ("target_altitude", "orbit_tolerance", "surface_clearance",
                "atmosphere_clearance", "plane_tolerance_deg",
                "acceleration_gravity_limit", "control_dt", "max_time",
                "arrival_radius_ratio", "success_hold_steps")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actor_layers(model_zip):
    """Return [(weight[out,in], bias[out]), ...] for the deterministic actor."""
    import torch

    with zipfile.ZipFile(model_zip) as archive:
        data = json.loads(archive.read("data"))
        state = torch.load(io.BytesIO(archive.read("policy.pth")),
                           map_location="cpu", weights_only=True)
    kwargs = data.get("policy_kwargs", {})
    activation = kwargs.get("activation_fn", "torch.nn.modules.activation.Tanh")
    if "Tanh" not in str(activation) or kwargs.get("squash_output") or data.get("use_sde"):
        raise ValueError("native export supports deterministic tanh MLP actors only")
    expected = {f"{name}.{part}" for name in ACTOR for part in ("weight", "bias")}
    actor = {key for key in state if key.startswith(("mlp_extractor.policy_net", "action_net"))}
    if actor != expected:
        raise ValueError(f"unexpected actor layers: {sorted(actor)}")
    layers = []
    for name in ACTOR:
        weight = state[f"{name}.weight"].numpy().astype(np.float32)
        bias = state[f"{name}.bias"].numpy().astype(np.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"malformed layer {name}")
        layers.append((weight, bias))
    for (previous, _), (current, _) in zip(layers, layers[1:]):
        if current.shape[1] != previous.shape[0]:
            raise ValueError("actor layer shapes do not chain")
    return layers


def forward(layers, observation):
    """Reference float64 evaluation of the exported actor (tests and checks)."""
    x = np.asarray(observation, dtype=np.float32).astype(np.float64)
    for index, (weight, bias) in enumerate(layers):
        x = weight.astype(np.float64) @ x + bias.astype(np.float64)
        if index < len(layers) - 1:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)


def contract(kind, metadata):
    environment = metadata["environment"]
    if kind == "ascent":
        stages = environment["stages"]
        values = {key: float(environment[key]) for key in ASCENT_KEYS}
        values["stage_count"] = float(len(stages))
        values["wet_mass"] = float(environment["payload_mass"]) + sum(
            stage["dry_mass"] + stage["fuel_mass"] for stage in stages)
        for index, stage in enumerate(stages):
            values[f"fuel_mass.{index}"] = float(stage["fuel_mass"])
        return values
    if kind == "capture":
        if metadata.get("task") != "orbital_capture_v2":
            raise ValueError("capture export requires an orbital_capture_v2 checkpoint")
        return {key: float(environment[key]) for key in CAPTURE_KEYS}
    raise ValueError(f"unknown policy kind {kind}")


def vehicle_header(profile_path, metadata):
    """Preflight signature of the one craft the ascent policy was trained on."""
    profile = json.loads(Path(profile_path).read_text())
    contract_stages = metadata["environment"]["stages"]
    if profile["vehicle_contract"]["stages"] != contract_stages:
        raise ValueError("vehicle profile and ascent checkpoint describe different stages")
    if abs(profile["wet_mass"] - contract(metadata=metadata, kind="ascent")["wet_mass"]) > 1e-6:
        raise ValueError("vehicle profile wet mass disagrees with the ascent checkpoint")
    return {
        "vehicle.name": profile["supported_profile"],
        "vehicle.parts": ",".join(sorted(profile["parts"])),
        "vehicle.engines": ";".join(f"{e['name']}:{e['activation_stage']}:{e['decouple_stage']}"
                                    for e in profile["engines"]),
        "vehicle.stage_plan": ",".join(str(x) for x in profile["stage_plan"]),
        "vehicle.fuel_stages": ",".join(str(x) for x in profile["fuel_stages"]),
    }


def write_policy(path, kind, model_zip, metadata_path, vehicle_profile=None):
    metadata = json.loads(Path(metadata_path).read_text())
    layers = actor_layers(model_zip)
    header = {"kind": kind, "source_sha256": sha256(model_zip),
              "observations": layers[0][0].shape[1], "actions": layers[-1][0].shape[0],
              "layers": len(layers)}
    if vehicle_profile is not None:
        header.update(vehicle_header(vehicle_profile, metadata))
    for key, value in contract(kind, metadata).items():
        if not math.isfinite(value):
            raise ValueError(f"nonfinite contract value {key}")
        header[f"contract.{key}"] = repr(value)
    text = "".join(f"{key}={value}\n" for key, value in header.items()).encode()
    blob = io.BytesIO()
    blob.write(MAGIC)
    blob.write(struct.pack("<i", len(text)))
    blob.write(text)
    blob.write(struct.pack("<i", len(layers)))
    for weight, bias in layers:
        blob.write(struct.pack("<ii", *weight.shape))
        blob.write(np.ascontiguousarray(weight, dtype="<f4").tobytes())
        blob.write(np.ascontiguousarray(bias, dtype="<f4").tobytes())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob.getvalue())
    return header


SEQUENCE_MAGIC = b"ORBSEQ1\n"
SEQUENCE_KINDS = {
    "generalized_ascent": ("ascent", 56, 2),
    "generalized_capture": ("orbital", 35, 3),
}


def generalized_contract(kind):
    """Simulator constants the C# builders must use, read from the Python code."""
    if kind == "generalized_ascent":
        from .generalized_aero_env import OBSERVATIONS, TASK
        from .generalized_env import MAX_STAGES
        from .train_architectures import make_env

        env = make_env("ascent", 1, "cpu", 0)
        e = env.envelope
        values = dict(mu=env.mu, radius=env.radius, rotation_period=env.rotation_period,
                      atmosphere_depth=env.atmosphere_depth, max_time=env.max_time,
                      orbit_tolerance=env.orbit_tolerance, control_dt=env.control_dt,
                      max_stages=MAX_STAGES, observations=OBSERVATIONS)
        for name in ("payload", "target", "attitude_rate", "attitude_tau",
                     "drag_area_per_tonne", "drag_alpha_area_per_tonne",
                     "lift_area_per_tonne", "isp", "sea_level_isp_ratio",
                     "stage_dry_fraction", "initial_twr", "upper_twr"):
            values[f"{name}_min"] = getattr(e, f"{name}_min")
            values[f"{name}_max"] = getattr(e, f"{name}_max")
        values["stages_max"] = e.stages_max
        return TASK, values
    if kind == "generalized_capture":
        from .batched_capture import VERSIONS

        task, observations = VERSIONS[2]
        # Rules from BatchedCapture.reset/_telemetry (tests pin them to the code).
        return task, dict(observations=observations, control_dt_over_T=0.01, max_time_over_T=4.0,
                          authority_gravity_limit=32.0, plane_tolerance_deg=5.0, success_hold_steps=2.0,
                          arrival_radius_ratio=1.04, altitude_floor=50000.0, altitude_atmosphere_margin=20000.0,
                          altitude_radius_fraction=0.1, altitude_scale_min=0.9, altitude_scale_max=1.2,
                          tolerance_cap=5000.0, tolerance_fraction=0.15, safe_margin=1000.0,
                          mass_min=300.0, mass_max=30000.0, thrust_accel_min=0.5, thrust_accel_max=30.0,
                          isp_min=280.0, isp_max=450.0, dry_fraction_min=0.05, dry_fraction_max=0.4,
                          vinf_fraction_min=0.2, vinf_fraction_max=0.7)
    raise ValueError(f"unknown generalized policy kind {kind}")


def write_sequence_policy(path, kind, checkpoint):
    """Actor tensors of a SequencePolicy as named little-endian float32 arrays."""
    from .train_architectures import load

    benchmark, observations, actions = SEQUENCE_KINDS[kind]
    model, metadata = load(Path(checkpoint))
    if metadata["contract"]["benchmark"] != benchmark:
        raise ValueError(f"{kind} needs a {benchmark} architecture checkpoint")
    spec = model.spec
    if (spec["observations"], spec["actions"]) != (observations, actions):
        raise ValueError("checkpoint observation/action sizes differ from the contract")
    if spec["architecture"] not in ("gru", "lstm") and not spec["architecture"].startswith("mlp"):
        raise ValueError("native export supports MLP, GRU and LSTM sequence policies")
    task, constants = generalized_contract(kind)
    header = {"kind": kind, "source_sha256": sha256(checkpoint), "simulator_task": task,
              "architecture": spec["architecture"], "observations": observations,
              "actions": actions, "context": spec["context"],
              "action_transform": spec.get("action_transform", "tanh"),
              "action_scale": repr(float(spec.get("action_scale", 1.0))),
              "deadzone": repr(float(spec.get("deadzone", 0.0)))}
    for key, value in constants.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"nonfinite contract value {key}")
        header[f"contract.{key}"] = repr(float(value))
    tensors = [(name, value.detach().cpu().numpy().astype(np.float32))
               for name, value in model.state_dict().items() if name.startswith("actor.")]
    text = "".join(f"{key}={value}\n" for key, value in header.items()).encode()
    blob = io.BytesIO()
    blob.write(SEQUENCE_MAGIC)
    blob.write(struct.pack("<i", len(text)))
    blob.write(text)
    blob.write(struct.pack("<i", len(tensors)))
    for name, array in tensors:
        encoded = name.encode()
        blob.write(struct.pack("<i", len(encoded)))
        blob.write(encoded)
        blob.write(struct.pack("<i", array.ndim))
        blob.write(struct.pack(f"<{array.ndim}i", *array.shape))
        blob.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob.getvalue())
    return header


GENERALIZED_MODELS = {
    "generalized_ascent": "models/generalized_ascent_gru.pt",
    "generalized_capture": "models/generalized_capture_lstm.pt",
}


def export_release(project_root, output_directory, *, generalized=True):
    root = Path(project_root)
    out = Path(output_directory)
    headers = {
        "ascent": write_policy(out / "ascent.policy", "ascent",
                               root / "models/transfer_ascent_ppo.zip",
                               root / "models/transfer_ascent_ppo.metadata.json",
                               root / "orbiter/data/transfer_vehicle.json"),
        "capture": write_policy(out / "capture.policy", "capture",
                                root / "models/moon_capture.zip",
                                root / "models/moon_capture.metadata.json"),
    }
    if generalized:
        for kind, checkpoint in GENERALIZED_MODELS.items():
            headers[kind] = write_sequence_policy(out / f"{kind}.policy", kind, root / checkpoint)
    return headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_release(args.project_root, args.output), indent=2))


if __name__ == "__main__":
    main()
