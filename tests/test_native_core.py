"""Parity between the self-contained C# addon core and the Python contracts.

The C# core (orbiter/native/Core) is compiled with Mono together with a small
stdin/stdout harness. Every case feeds identical inputs to both sides.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from orbiter import native_export
from orbiter.capture_env import CaptureConfig, build_capture_observation, decode_capture_action
from orbiter.env import EnvConfig, build_observation
from orbiter.mission import plan_child_transfer
from orbiter.mission_live import incoming_radius_time
from orbiter.orbital import lambert, propagate
from orbiter.system import SystemCatalog

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (shutil.which("mcs") and shutil.which("mono")),
                                reason="Mono C# compiler and runtime required")


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    out = tmp_path_factory.mktemp("native")
    policies = native_export.export_release(ROOT, out)
    sources = sorted(str(p) for p in (ROOT / "orbiter/native/Core").glob("*.cs"))
    exe = out / "harness.exe"
    subprocess.run(["mcs", "-langversion:7", f"-out:{exe}", "-r:System.Core.dll", *sources,
                    str(ROOT / "tests/native/ParityHarness.cs")], check=True, capture_output=True)

    def run(lines):
        text = "\n".join(" ".join(_word(x) for x in line) for line in lines) + "\n"
        result = subprocess.run(["mono", str(exe)], input=text, capture_output=True, text=True, check=True)
        rows = result.stdout.strip().split("\n")
        assert len(rows) == len(lines)
        for row in rows:
            assert not row.startswith("error"), row
        return [np.array([float(x) for x in row.split()]) if row != "none" else None for row in rows]

    run.ascent = str(out / "ascent.policy")
    run.capture = str(out / "capture.policy")
    run.headers = policies
    return run


def _word(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return repr(float(value)) if math.isfinite(value) else ("Infinity" if value > 0 else "-Infinity")


def _flat(*items):
    out = []
    for item in items:
        if isinstance(item, (list, tuple, np.ndarray)):
            out.extend(float(x) for x in np.ravel(item))
        else:
            out.append(item)
    return out


def test_policy_matches_sb3_predict(native):
    from stable_baselines3 import PPO

    rng = np.random.default_rng(7)
    for path, zip_path, size in ((native.ascent, "models/transfer_ascent_ppo.zip", 16),
                                 (native.capture, "models/moon_capture.zip", 27)):
        model = PPO.load(ROOT / zip_path, device="cpu")
        observations = rng.uniform(-4, 4, (64, size)).astype(np.float32)
        rows = native([("act", path, *o) for o in observations])
        for observation, row in zip(observations, rows):
            expected = model.predict(observation, deterministic=True)[0]
            np.testing.assert_allclose(row, expected, atol=2e-6)


def test_policy_header_carries_metadata_contract(native):
    ascent = json.loads((ROOT / "models/transfer_ascent_ppo.metadata.json").read_text())["environment"]
    header = native.headers["ascent"]
    assert float(header["contract.target_altitude"]) == ascent["target_altitude"] == 100000.0
    assert float(header["contract.mu"]) == ascent["mu"]
    assert header["source_sha256"] == native_export.sha256(ROOT / "models/transfer_ascent_ppo.zip")


def test_ascent_observation_and_action(native):
    metadata = json.loads((ROOT / "models/transfer_ascent_ppo.metadata.json").read_text())
    config = EnvConfig.from_dict(metadata["environment"]) if hasattr(EnvConfig, "from_dict") else None
    if config is None:
        from orbiter.checkpoints import load_controller
        _, config = load_controller(ROOT / "models/transfer_ascent_ppo.zip")
    rng = np.random.default_rng(3)
    cases = []
    for index in range(80):
        d = dict(altitude=rng.uniform(-50, 400e3), vertical_speed=rng.uniform(-500, 2500),
                 horizontal_speed=rng.uniform(0, 3000),
                 apoapsis=math.inf if index % 7 == 0 else rng.uniform(-6e5, 5e6),
                 periapsis=rng.uniform(-6e5, 4e5), energy=rng.uniform(-6e6, 1e6),
                 dynamic_pressure=rng.uniform(0, 6e4))
        extra = dict(fuel_fraction=rng.uniform(0, 1), mass_fraction=rng.uniform(.2, 1),
                     pitch=rng.uniform(-10, 90), throttle=rng.uniform(0, 1),
                     elapsed=rng.uniform(0, 1400), thrust_to_weight=rng.uniform(0, 4),
                     stage=int(rng.integers(0, 2)))
        cases.append((d, extra))
    rows = native([("ascent_obs", native.ascent, d["altitude"], d["vertical_speed"], d["horizontal_speed"],
                    d["apoapsis"], d["periapsis"], d["energy"], d["dynamic_pressure"], e["fuel_fraction"],
                    e["mass_fraction"], e["pitch"], e["throttle"], e["elapsed"], e["thrust_to_weight"],
                    e["stage"]) for d, e in cases])
    for (d, e), row in zip(cases, rows):
        expected = build_observation(config, d, **e)
        np.testing.assert_allclose(row, expected.astype(np.float64), rtol=0, atol=0)
    decoded = native([("ascent_decode", a, b) for a, b in ((-1, -1), (1, 1), (0.3, -0.4), (2, -3))])
    for (a, b), row in zip(((-1, -1), (1, 1), (0.3, -0.4), (2, -3)), decoded):
        a, b = np.clip([a, b], -1, 1)
        np.testing.assert_allclose(row, [(a + 1) / 2, (b + 1) * 50 - 10])


def _capture_config(body, altitude):
    metadata = json.loads((ROOT / "models/moon_capture.metadata.json").read_text())
    overrides = asdict(CaptureConfig.from_dict(metadata["environment"]))
    for name in ("body_name", "mu", "radius", "atmosphere_depth", "sphere_of_influence"):
        overrides.pop(name)
    overrides.update(target_altitude=altitude, target_altitude_randomization=0.0,
                     orbit_tolerance=5000.0, randomization=0.0)
    return CaptureConfig.for_body(body, **overrides)


def test_capture_observation_and_decode(native):
    system = SystemCatalog.load()
    rng = np.random.default_rng(11)
    lines, expected_obs, expected_decode = [], [], []
    for index in range(60):
        name = ("Mun", "Minmus")[index % 2]
        body = dict(system.bodies[name], name=name)
        altitude = rng.uniform(40e3, 60e3)
        config = _capture_config(body, altitude)
        L = config.goal_radius
        r = rng.normal(size=3) * L * rng.uniform(.9, 3)
        v = rng.normal(size=3) * config.velocity_scale * rng.uniform(.3, 2)
        mass, dry = rng.uniform(2000, 4000), rng.uniform(1000, 1900)
        telemetry = dict(r=r, v=v, direction=rng.normal(size=3), goal_normal=rng.normal(size=3),
                         mass=mass, initial_mass=mass * rng.uniform(1, 1.3), dry_mass=dry,
                         thrust=rng.uniform(0, 6e4) if index % 9 else 0.0,
                         isp=rng.uniform(300, 350), time=rng.uniform(0, 1800))
        action = rng.uniform(-1.5, 1.5, 3) if index % 10 else np.zeros(3)
        head = (native.capture, name, body["gravitational_parameter"], body["equatorial_radius"],
                body["atmosphere_depth"], body["sphere_of_influence"], altitude)
        t = _flat(r, v, telemetry["direction"], telemetry["goal_normal"], telemetry["mass"],
                  telemetry["initial_mass"], dry, telemetry["thrust"], telemetry["isp"], telemetry["time"])
        lines.append(("capture_obs", *head, *t))
        lines.append(("capture_decode", *head, *t, *action))
        expected_obs.append(build_capture_observation(config, telemetry))
        throttle, direction = decode_capture_action(config, telemetry, action)
        expected_decode.append(np.array([throttle, *direction]))
    rows = native(lines)
    for index, expected in enumerate(expected_obs):
        np.testing.assert_allclose(rows[2 * index], expected.astype(np.float64), rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(rows[2 * index + 1], expected_decode[index], rtol=1e-12, atol=1e-12)


def test_kepler_lambert_and_ephemeris(native):
    mu = 3.5316e12
    rng = np.random.default_rng(5)
    lines, expected = [], []
    for _ in range(40):
        r = rng.normal(size=3) * 7e5
        v = rng.normal(size=3) * 2200
        dt = rng.uniform(-4000, 4000)
        try:
            expected.append(np.concatenate(propagate(r, v, dt, mu)))
        except ValueError:
            continue
        lines.append(("propagate", *r, *v, dt, mu))
    for _ in range(40):
        p1, p2 = rng.normal(size=3) * 7e5, rng.normal(size=3) * 1.2e7
        dt, long_way = rng.uniform(2e4, 2e5), bool(rng.integers(0, 2))
        try:
            expected.append(np.concatenate(lambert(p1, p2, dt, mu, long_way=long_way)))
        except ValueError:
            continue
        lines.append(("lambert", *p1, *p2, dt, mu, long_way))
    system = SystemCatalog.load()
    for name in ("Mun", "Minmus"):
        orbit = system.bodies[name]["orbit"]
        for ut in rng.uniform(0, 5e7, 10):
            expected.append(np.concatenate(system.state(name, ut, relative_to="Kerbin")))
            lines.append(("ephemeris", mu, orbit["semi_major_axis"], orbit["eccentricity"],
                          orbit["mean_anomaly_at_epoch"], orbit["epoch"], *orbit["basis_p"],
                          *orbit["basis_q"], ut))
    for row, reference in zip(native(lines), expected):
        np.testing.assert_allclose(row, reference, rtol=1e-9, atol=1e-6)


def _parking_state(radius, inclination, phase, mu):
    speed = math.sqrt(mu / radius)
    position = radius * np.array([math.cos(phase), math.sin(phase), 0.0])
    velocity = speed * np.array([-math.sin(phase) * math.cos(inclination),
                                 math.cos(phase) * math.cos(inclination), math.sin(inclination)])
    return position, velocity


def test_transfer_plan_matches_python(native):
    system = SystemCatalog.load()
    kerbin = system.bodies["Kerbin"]
    mu = kerbin["gravitational_parameter"]
    rng = np.random.default_rng(13)

    def body_words(body, semi_major):
        return (body["gravitational_parameter"], body["equatorial_radius"], body["atmosphere_depth"],
                body.get("sphere_of_influence", 1e12), semi_major)

    lines, expected = [], []
    for index in range(6):
        target = ("Mun", "Minmus")[index % 2]
        destination = system.bodies[target]
        orbit = destination["orbit"]
        r, v = _parking_state(600e3 + rng.uniform(80e3, 150e3), rng.uniform(0, .1), rng.uniform(0, 6.28), mu)
        epoch, altitude = rng.uniform(1e5, 5e6), rng.uniform(40e3, 60e3)
        plan = plan_child_transfer(system, "Kerbin", target, r, v, epoch, target_altitude=altitude)
        expected.append(np.array([plan.departure_ut, plan.arrival_ut, plan.arrival_vinf,
                                  plan.estimated_capture_delta_v, 0, *plan.delta_v, *plan.aim_offset]))
        lines.append(("plan", *body_words(kerbin, 0), *body_words(destination, orbit["semi_major_axis"]),
                      orbit["semi_major_axis"], orbit["eccentricity"], orbit["mean_anomaly_at_epoch"],
                      orbit["epoch"], *orbit["basis_p"], *orbit["basis_q"], *r, *v, epoch, altitude))
    for row, reference in zip(native(lines), expected):
        reference[4] = row[4]  # branch flag is not exported by the Python plan
        np.testing.assert_allclose(row, reference, rtol=1e-8, atol=1e-5)


def test_capture_geometry_helpers(native):
    system = SystemCatalog.load()
    mun = system.bodies["Mun"]
    mu, radius = mun["gravitational_parameter"], mun["equatorial_radius"]
    rng = np.random.default_rng(17)
    lines, expected = [], []
    for _ in range(20):
        # Inbound hyperbola from about 2,000 km aimed near 50-70 km periapsis.
        r = np.array([2.0e6, rng.uniform(1e5, 4e5), rng.uniform(-5e4, 5e4)])
        v = np.array([-rng.uniform(500, 900), rng.uniform(-20, 20), rng.uniform(-5, 5)])
        target_radius = radius + rng.uniform(40e3, 60e3)
        lead = 180.0
        pr, pv = propagate(r, v, lead, mu)
        rr = float(np.linalg.norm(pr))
        radial = pr / rr
        vr = float(pv @ radial)
        tangential = pv - vr * radial
        speed = float(np.linalg.norm(tangential))
        desired = target_radius * math.sqrt((vr * vr + 2 * mu * (1 / target_radius - 1 / rr))
                                            / (rr * rr - target_radius * target_radius))
        delta = (desired - speed) * tangential / speed
        expected.append(None if np.linalg.norm(delta) < 0.02 else delta)
        lines.append(("correction", *r, *v, mu, radius, target_radius, lead))
        # Time-to-periapsis from the conic, then the 1.04 L crossing time.
        energy = float(v @ v) / 2 - mu / float(np.linalg.norm(r))
        h = np.cross(r, v)
        e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
        e = float(np.linalg.norm(e_vec))
        a = -mu / (2 * energy)
        cos_nu = float(e_vec @ r) / (e * np.linalg.norm(r))
        F = math.acosh((e + cos_nu) / (1 + e * cos_nu))
        ttp = math.sqrt((-a) ** 3 / mu) * (e * math.sinh(F) - F)
        goal = 1.04 * (radius + 50e3)
        try:
            expected.append(np.array([incoming_radius_time(r, v, mu, goal, ttp)]))
        except ValueError:
            continue
        lines.append(("incoming", *r, *v, mu, goal, ttp))
    for row, reference in zip(native(lines), expected):
        if reference is None:
            assert row is None
        else:
            np.testing.assert_allclose(row, reference, rtol=1e-9, atol=1e-6)


def test_g10_matches_python_format(native):
    values = [0.0, -0.0, 1.0, -2.5, 0.1, 1e-5, 1.2345678901234, 123456789012.0, 9999999999.0,
              -0.000123456789, 0.30000000000000004, 1e10, 1.5e-7, 7.0e21]
    rows = subprocess_g10(native, values)
    assert rows == [format(v, ".10g") for v in values]


def subprocess_g10(native, values):
    import subprocess as sp
    exe = Path(native.ascent).parent / "harness.exe"
    text = "".join(f"g10 {v!r}\n" for v in values)
    return sp.run(["mono", str(exe)], input=text, capture_output=True, text=True, check=True).stdout.split()


STOCK = Path.home() / "Library/Application Support/Steam/steamapps/common/Kerbal Space Program/Ships/VAB"


@pytest.mark.skipif(not (STOCK / "Orbiter One.craft").exists(), reason="local stock craft templates required")
def test_craft_builder_matches_python(native, tmp_path):
    from orbiter.make_craft import make_terrier_craft, make_transfer_craft

    one = (STOCK / "Orbiter One.craft").read_text(errors="replace")
    science = (STOCK / "Science Jr.craft").read_text(errors="replace")
    expected = make_transfer_craft(make_terrier_craft(one, science))
    output = tmp_path / "transfer.craft"
    # The harness splits on spaces; stock names contain them.
    shutil.copy(STOCK / "Orbiter One.craft", tmp_path / "one.craft")
    shutil.copy(STOCK / "Science Jr.craft", tmp_path / "science.craft")
    native([("craft", str(tmp_path / "one.craft"), str(tmp_path / "science.craft"), str(output))])
    assert output.read_text() == expected


def _synthetic_templates():
    # The synthetic stack from tests/test_make_craft, never a stock craft.
    from orbiter.make_craft import _split_parts
    from tests.test_make_craft import part, terrier_source

    source = terrier_source.__wrapped__()
    source = source.replace("ship = OrbiterRL Terrier", "ship = Synthetic Orbiter One")
    source = source.replace("liquidEngine3.v2_5", "liquidEngine2_5")
    source = source.replace("part = liquidEngine2_13\n\tistg = 3", "part = liquidEngine2_13\n\tistg = 5")
    source = source.replace("part = Decoupler.1_6\n\tistg = 2", "part = Decoupler.1_6\n\tistg = 3")
    _, blocks = _split_parts(source)
    blocks[0] = blocks[0][:-2] + "\tRESOURCE\n\t{\n\t\tname = MonoPropellant\n\t\tamount = 10\n\t}\n}\n"
    source = "ship = Synthetic Orbiter One\n" + "".join(blocks)
    source += "".join(part(identifier, 4, 0)
                      for identifier in ("radialDecoupler_90", "solidBooster.v2_91", "noseCone_92"))
    template = "ship = Synthetic Science Jr\n" + part(
        "liquidEngine3.v2_93", 2, 0, parent="fuelTankSmall_94", child="Decoupler.1_95", top=0, bottom=-0.8)
    return source, template


def test_craft_builder_matches_python_on_synthetic_templates(native, tmp_path):
    from orbiter.make_craft import make_terrier_craft, make_transfer_craft

    one, science = _synthetic_templates()
    (tmp_path / "one.craft").write_text(one.replace("\n", "\r\n"))  # CRLF like Windows installs
    (tmp_path / "science.craft").write_text(science)
    output = tmp_path / "transfer.craft"
    native([("craft", str(tmp_path / "one.craft"), str(tmp_path / "science.craft"), str(output))])
    assert output.read_text() == make_transfer_craft(make_terrier_craft(one, science))
