"""Parity of the C# generalized policies and observations with the Python originals."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from orbiter.batched_capture import BatchedCapture
from orbiter.sequence_policy import History
from orbiter.train_architectures import load, make_env
from tests.test_native_core import native, pytestmark  # noqa: F401  (shared Mono fixture)

ROOT = Path(__file__).resolve().parents[1]
ASCENT = ROOT / "models/generalized_ascent_gru.pt"
CAPTURE = ROOT / "models/generalized_capture_lstm.pt"


def _policy(native, kind):
    return str(Path(native.ascent).parent / f"{kind}.policy")


@pytest.mark.parametrize("kind,checkpoint", [("generalized_ascent", ASCENT), ("generalized_capture", CAPTURE)])
def test_sequence_policy_matches_torch(native, kind, checkpoint):
    model, _ = load(checkpoint)
    spec = model.spec
    rng = np.random.default_rng(21)
    features = spec["observations"] + spec["actions"] + 1
    windows = rng.uniform(-2, 2, (24, spec["context"], features)).astype(np.float32)
    windows[:, :, -1] = 1.0
    windows[:6, :5] = 0.0  # zero-padded start of an episode
    rows = native([("seq_act", _policy(native, kind), *w.ravel()) for w in windows])
    with torch.no_grad():
        expected = model(torch.from_numpy(windows)).numpy()
    for row, reference in zip(rows, expected):
        np.testing.assert_allclose(row, reference, atol=2e-5)


def test_history_matches_python(native):
    rng = np.random.default_rng(4)
    history = History(1, 5, 4, 2, "cpu")
    line = ["history", 5, 4, 2, 7]
    for step in range(7):
        obs = rng.normal(size=4).astype(np.float32)
        action = None if step == 0 else rng.uniform(-1, 1, 2)
        history.append(torch.from_numpy(obs)[None],
                       None if action is None else torch.tensor(action, dtype=torch.float32)[None])
        line += [*obs, 0] if action is None else [*obs, 1, *action]
    row = native([tuple(line)])[0]
    np.testing.assert_allclose(row, history.data[0].numpy().ravel(), atol=1e-7)


def _ascent_lines(native, env):
    lines = []
    d = env.telemetry()
    mass, thrust = env.mass(), env.available_thrust()
    for i in range(env.count):
        n = int(env.stage_count[i])
        stages = env.stages[i, :n].tolist()
        lines.append(("gen_ascent_obs", _policy(native, "generalized_ascent"), n, *[x for row in stages for x in row],
                      float(env.payload[i]), float(env.drag_area[i]), float(env.drag_alpha_area[i]),
                      float(env.lift_area[i]), float(env.attitude_rate[i]), float(env.attitude_tau[i]),
                      float(env.target[i]),
                      *(float(d[k][i]) for k in ("altitude", "vertical_speed", "horizontal_speed", "apoapsis",
                                                   "periapsis", "energy", "dynamic_pressure")),
                      float(env.fuel[i]), float(mass[i]), float(env.pitch[i]), float(env.throttle[i]),
                      float(env.time[i]), float(thrust[i]), int(env.stage[i])))
    return lines


@pytest.mark.parametrize("four_stage", [False, True])
def test_generalized_ascent_observation_matches_batched_env(native, four_stage):
    env = make_env("ascent", 48, "cpu", 17 + four_stage, four_stage=four_stage)
    torch.manual_seed(3)
    lines, expected = [], []
    for step in range(120):
        if step in (0, 3, 25, 60, 119):
            lines += _ascent_lines(native, env)
            expected += list(env.observation().numpy())
        action = (env.teacher() + 0.2 * torch.randn(env.count, 2)).clamp(-1, 1)
        env.step(action)
    for row, reference in zip(native(lines), expected):
        np.testing.assert_allclose(row, reference.astype(np.float64), rtol=2e-6, atol=2e-6)


def test_generalized_capture_observation_rules_and_decode_match(native):
    env = BatchedCapture(68, device="cpu", seed=9, version=2)
    lines, expected = [], []
    rng = np.random.default_rng(8)
    for step in range(30):
        if step in (0, 4, 17, 29):
            action = rng.uniform(-1.4, 1.4, (env.count, 3))
            d = env.telemetry()
            obs = env.observation().numpy()
            for i in range(env.count):
                name = env.names[int(env.body[i])]
                lines.append(("gen_capture", _policy(native, "generalized_capture"), name, float(env.mu[i]),
                              float(env.radius[i]), float(env.atmosphere_depth[i]), float(env.soi[i]),
                              float(env.goal[i] - env.radius[i]), *env.r[i].tolist(), *env.v[i].tolist(),
                              *env.direction[i].tolist(), *env.goal_normal[i].tolist(), float(env.mass[i]),
                              float(env.initial_mass[i]), float(env.dry_mass[i]), float(env.thrust[i]),
                              float(env.isp[i]), float(env.time[i]), *action[i]))
                a = torch.tensor(action[i])
                throttle = float(min(1.0, a.norm()) * d["authority"][i] / (env.thrust[i] / env.mass[i]))
                direction = (a / a.norm()).tolist()
                expected.append((
                    [float(env.goal[i]), float(env.safe[i]), float(env.tolerance[i]), float(env.V[i]),
                     float(env.T[i]), float(env.control_dt[i]), float(env.max_time[i]), throttle, *direction,
                     float(d["in_band"][i])], obs[i].astype(np.float64)))
        env.step(env.teacher())
    for row, (head, observation) in zip(native(lines), expected):
        np.testing.assert_allclose(row[:12], head, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(row[12:], observation, rtol=3e-6, atol=3e-6)


def test_capture_altitude_band_matches_reset_rule(native):
    from orbiter.system import SystemCatalog

    catalog = SystemCatalog.load()
    lines, expected = [], []
    for name, body in sorted(catalog.bodies.items()):
        radius, atmosphere = body["equatorial_radius"], body["atmosphere_depth"]
        base = max(50000, max(atmosphere + 20000, 0.1 * radius))
        expected.append([max(base * 0.9, atmosphere + 10000), max(base * 1.2, atmosphere + 10000)])
        lines.append(("gen_band", _policy(native, "generalized_capture"), radius, atmosphere))
    for row, reference in zip(native(lines), expected):
        np.testing.assert_allclose(row, reference)
