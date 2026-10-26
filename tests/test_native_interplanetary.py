"""Planet missions in the C# core: feedback capture, launch windows, escape burns, course corrections."""
import math

import numpy as np
import pytest
import torch

from orbiter.batched_capture import BatchedCapture
from orbiter.mission import plan_interplanetary_window
from orbiter.orbital import propagate
from orbiter.system import SystemCatalog
from tests.test_native_core import native, pytestmark  # noqa: F401  (shared Mono fixture)
from tests.test_native_generalized import _policy

PLANETS = ("Moho", "Eve", "Duna", "Dres", "Jool", "Eeloo")


def _body(body, semi_major):
    return (body["gravitational_parameter"], body["equatorial_radius"], body["atmosphere_depth"],
            body.get("sphere_of_influence") or 1e12, semi_major)


def _elements(orbit):
    return (orbit["semi_major_axis"], orbit["eccentricity"], orbit["mean_anomaly_at_epoch"], orbit["epoch"],
            *orbit["basis_p"], *orbit["basis_q"])


def test_teacher_matches_batched_capture(native):
    env = BatchedCapture(64, seed=5, version=2)
    lines, expected = [], []
    for _ in range(4):
        teacher = env.teacher().numpy()
        for i in range(env.count):
            name = env.names[int(env.body[i])]
            lines.append(("gen_teacher", _policy(native, "generalized_capture"), name, float(env.mu[i]),
                          float(env.radius[i]), float(env.atmosphere_depth[i]), float(env.soi[i]),
                          float(env.goal[i] - env.radius[i]), *env.r[i].tolist(), *env.v[i].tolist(),
                          *env.direction[i].tolist(), *env.goal_normal[i].tolist(), float(env.mass[i]),
                          float(env.initial_mass[i]), float(env.dry_mass[i]), float(env.thrust[i]),
                          float(env.isp[i]), float(env.time[i])))
            expected.append(teacher[i])
        env.step(env.teacher())
    for row, reference in zip(native(lines), expected):
        np.testing.assert_allclose(row, reference, rtol=2e-6, atol=2e-6)


def test_window_search_is_at_least_as_good_as_python(native):
    system = SystemCatalog.load()
    kerbin, sun = system.bodies["Kerbin"], system.bodies["Sun"]
    epoch, parking = 2.0e6, 100000.0
    lines, references, waits = [], [], []
    for name in PLANETS:
        target = system.bodies[name]
        reference = plan_interplanetary_window(system, "Kerbin", name, epoch, parking_altitude=parking)
        p1, p2 = kerbin["orbit"]["period"], target["orbit"]["period"]
        inverse = abs(1 / p1 - 1 / p2)
        wait = min(2 / inverse if inverse > 1e-12 else p1, 3 * max(p1, p2))
        lines.append(("interplanetary", sun["gravitational_parameter"],
                      *_body(kerbin, kerbin["orbit"]["semi_major_axis"]), *_body(target, target["orbit"]["semi_major_axis"]),
                      *_elements(kerbin["orbit"]), *_elements(target["orbit"]), epoch + 180,
                      kerbin["equatorial_radius"] + parking, reference["target_altitude"], wait))
        references.append(reference)
        waits.append(wait)
    for name, row, reference, wait in zip(PLANETS, native(lines), references, waits):
        departure, arrival, ejection, capture, vinf = row[:5]
        # The C# search takes the earliest departure on Python's grid within 5% of
        # its cheapest and refines around it.
        assert ejection + capture <= 1.05 * reference["estimated_total_delta_v"] + 1e-6, name
        assert ejection + capture >= 0.85 * reference["estimated_total_delta_v"], name
        assert departure <= reference["departure_ut"] + wait / 47 + 1e-6, name
        # The departure excess velocity is a Lambert arc that reaches the planet.
        r1, v1 = system.state("Kerbin", departure, relative_to="Sun")
        r2, _ = system.state(name, arrival, relative_to="Sun")
        position, _ = propagate(r1, v1 + np.array(row[6:9]), arrival - departure, sun["gravitational_parameter"])
        assert np.linalg.norm(position - r2) < 1e-6 * np.linalg.norm(r2), name


def test_escape_burn_leaves_along_the_requested_asymptote(native):
    mu = SystemCatalog.load().bodies["Kerbin"]["gravitational_parameter"]
    rng = np.random.default_rng(31)
    lines, cases = [], []
    for _ in range(24):
        radius = 600e3 + rng.uniform(80e3, 150e3)
        speed, phase = math.sqrt(mu / radius), rng.uniform(0, 2 * math.pi)
        r = radius * np.array([math.cos(phase), math.sin(phase), 0.0])
        v = speed * np.array([-math.sin(phase), math.cos(phase), 0.0])
        declination, azimuth = math.radians(rng.uniform(-20, 20)), rng.uniform(0, 2 * math.pi)
        vinf = rng.uniform(400, 3000) * np.array([math.cos(declination) * math.cos(azimuth),
                                                  math.cos(declination) * math.sin(azimuth), math.sin(declination)])
        epoch, center = 1e6, 1e6 + rng.uniform(0, 5000)
        lines.append(("ejection", *r, *v, epoch, mu, *vinf, center))
        cases.append((r, v, epoch, vinf, center))
    for row, (r, v, epoch, vinf, center) in zip(native(lines), cases):
        assert len(row) == 4
        ut, delta = row[0], np.array(row[1:4])
        period = 2 * math.pi * math.sqrt(np.linalg.norm(r) ** 3 / mu)
        assert abs(ut - center) <= period / 2 + 1e-6
        rb, vb = propagate(r, v, ut - epoch, mu)
        after = vb + delta
        assert abs(float(rb @ after)) < 1e-6 * np.linalg.norm(rb) * np.linalg.norm(after)  # tangential: periapsis burn
        # Outgoing asymptote of the hyperbola: true anomaly acos(-1/e) from periapsis.
        h = np.cross(rb, after)
        e = np.cross(after, h) / mu - rb / np.linalg.norm(rb)
        p_hat = e / np.linalg.norm(e)
        q_hat = np.cross(h / np.linalg.norm(h), p_hat)
        nu = math.acos(-1 / np.linalg.norm(e))
        asymptote = math.cos(nu) * p_hat + math.sin(nu) * q_hat
        cosine = float(asymptote @ vinf) / np.linalg.norm(vinf)
        assert math.degrees(math.acos(min(1.0, cosine))) < 1e-4
        energy = float(after @ after) / 2 - mu / np.linalg.norm(rb)
        assert abs(math.sqrt(2 * energy) / np.linalg.norm(vinf) - 1) < 1e-9


def test_course_correction_reaches_the_aimed_point(native):
    system = SystemCatalog.load()
    kerbin, sun, duna = system.bodies["Kerbin"], system.bodies["Sun"], system.bodies["Duna"]
    mu = sun["gravitational_parameter"]
    plan = plan_interplanetary_window(system, "Kerbin", "Duna", 2.0e6)
    start = plan["departure_ut"] + 40 * 21600
    r0, v0 = system.state("Kerbin", plan["departure_ut"], relative_to="Sun")
    r, v = propagate(r0, v0 + np.array(plan["departure_vinf_vector"]) + np.array([3.0, -2.0, 1.5]),
                     start - plan["departure_ut"], mu)
    row = native([("course", *r, *v, start, mu, *_body(duna, duna["orbit"]["semi_major_axis"]),
                   *_elements(duna["orbit"]), plan["arrival_ut"], 70000.0, 600.0)])[0]
    departure, arrival = row[0], row[1]
    delta, offset, position = np.array(row[4:7]), np.array(row[7:10]), np.array(row[10:13])
    assert departure == start + 600.0
    assert abs(arrival - plan["arrival_ut"]) <= 0.15 * (plan["arrival_ut"] - departure) + 1e-6
    lead_r, lead_v = propagate(r, v, 600.0, mu)
    np.testing.assert_allclose(position, lead_r, rtol=1e-10)
    target, _ = system.state("Duna", arrival, relative_to="Sun")
    final, _ = propagate(lead_r, lead_v + delta, arrival - departure, mu)
    assert np.linalg.norm(final - (target + offset)) < 1.0
    assert 0.5 * duna["equatorial_radius"] < np.linalg.norm(offset) < duna["sphere_of_influence"]
    assert np.linalg.norm(delta) < 50.0


def test_ejection_cost_matches_the_solved_burn(native):
    mu = SystemCatalog.load().bodies["Kerbin"]["gravitational_parameter"]
    rng = np.random.default_rng(37)
    lines, costs = [], []
    for _ in range(24):
        radius = 600e3 + rng.uniform(80e3, 150e3)
        speed, phase = math.sqrt(mu / radius), rng.uniform(0, 2 * math.pi)
        r = radius * np.array([math.cos(phase), math.sin(phase), 0.0])
        v = speed * np.array([-math.sin(phase), math.cos(phase), 0.0])
        declination, azimuth = math.radians(rng.uniform(-30, 30)), rng.uniform(0, 2 * math.pi)
        vinf = rng.uniform(400, 3000) * np.array([math.cos(declination) * math.cos(azimuth),
                                                  math.cos(declination) * math.sin(azimuth), math.sin(declination)])
        lines.append(("ejection", *r, *v, 1e6, mu, *vinf, 1e6))
        costs.append(("ejection_cost", mu, radius, *vinf, 0.0, 0.0, 1.0))
    rows = native(lines + costs)
    solved = 0
    for burn, cost, line in zip(rows[:24], rows[24:], costs):
        if burn is None:
            # Slow, steep departures have no asymptote reachable from this orbit: priced prohibitively.
            vinf = np.linalg.norm(line[3:6])
            assert cost[0] == pytest.approx(math.sqrt(vinf**2 + 2 * mu / line[2]) + math.sqrt(mu / line[2]))
            continue
        solved += 1
        np.testing.assert_allclose(cost[0], np.linalg.norm(burn[1:4]), rtol=1e-6)
    assert solved >= 16
    # In the orbit plane it is the classic Oberth escape.
    in_plane = native([("ejection_cost", mu, 700e3, 1000.0, 0.0, 0.0, 0.0, 0.0, 1.0)])[0][0]
    assert abs(in_plane - (math.sqrt(1e6 + 2 * mu / 700e3) - math.sqrt(mu / 700e3))) < 1e-9
