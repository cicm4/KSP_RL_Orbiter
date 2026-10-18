"""Three-dimensional encounter planning from a captured KSP system.

Planning is pure mathematics. An ephemeris match is a candidate trajectory, not
proof of a KSP encounter: the live adapter must check the game's patched conics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .orbital import elements, lambert, propagate, rocket_delta_v
from .system import SystemCatalog


@dataclass
class TransferPlan:
    source: str
    target: str
    kind: str
    departure_ut: float
    arrival_ut: float
    departure_position: list[float]
    departure_velocity: list[float]
    delta_v: list[float]
    arrival_vinf: float
    estimated_capture_delta_v: float
    target_altitude: float
    target_periapsis_radius: float
    aim_offset: list[float]
    route: list[str]
    assumptions: list[str]

    @property
    def departure_delta_v(self):
        return float(np.linalg.norm(self.delta_v))

    @property
    def estimated_total_delta_v(self):
        return self.departure_delta_v + self.estimated_capture_delta_v

    def to_dict(self):
        return dict(
            asdict(self),
            departure_delta_v=self.departure_delta_v,
            estimated_total_delta_v=self.estimated_total_delta_v,
            actual_ksp_verified=False,
        )


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        raise ValueError("Undefined transfer direction")
    return np.asarray(vector, dtype=float) / norm


def capture_delta_v(mu, periapsis_radius, vinf):
    if (
        not all(math.isfinite(x) and x > 0 for x in (mu, periapsis_radius))
        or not math.isfinite(vinf)
        or vinf < 0
    ):
        raise ValueError("Invalid capture budget inputs")
    return math.sqrt(vinf * vinf + 2 * mu / periapsis_radius) - math.sqrt(
        mu / periapsis_radius
    )


def require_delta_v(plan, *, mass, dry_mass, isp, reserve=100.0):
    available = rocket_delta_v(mass, dry_mass, isp)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Reserve must be finite and nonnegative")
    estimated = plan.estimated_total_delta_v
    if not math.isfinite(estimated) or estimated < 0:
        raise ValueError("Plan estimated total delta-v must be finite and nonnegative")
    required = estimated + reserve
    if not math.isfinite(required):
        raise ValueError("Total delta-v including reserve must be finite")
    if available < required:
        raise ValueError(
            f"Insufficient vehicle delta-v: {available:.1f} m/s available, "
            f"{required:.1f} m/s estimated including {reserve:g} m/s reserve"
        )
    return dict(
        available_delta_v=available,
        estimated_required_delta_v=required,
        reserve_delta_v=reserve,
        remaining_margin=available - required,
        limitation="Two-body impulse estimate; finite burns and corrections can cost more.",
    )


def plan_child_transfer(
    system: SystemCatalog,
    source,
    target,
    position,
    velocity,
    epoch,
    *,
    target_altitude=None,
    lead_time=180.0,
    departure_samples=48,
    duration_samples=20,
    maximum_wait=None,
):
    """Search coast departure/flight time and aim a finite safe B-plane offset.

    Supports a spacecraft already orbiting the target's parent: e.g. Kerbin to
    either moon, Jool to its moons, or a heliocentric spacecraft to a planet.
    Inclination and eccentricity enter through full 3D ephemerides and states.
    """
    if system.bodies[target]["parent"] != source:
        raise ValueError(
            "Direct-child planning requires the spacecraft to orbit the destination parent"
        )
    if min(departure_samples, duration_samples) < 2 or not math.isfinite(epoch):
        raise ValueError(
            "Need a finite epoch and at least two search samples per dimension"
        )
    body, destination = system.bodies[source], system.bodies[target]
    mu, target_mu = (
        body["gravitational_parameter"],
        destination["gravitational_parameter"],
    )
    altitude = (
        system.default_target_altitude(target)
        if target_altitude is None
        else float(target_altitude)
    )
    periapsis = destination["equatorial_radius"] + altitude
    if (
        altitude <= destination["atmosphere_depth"] + 1000
        or periapsis >= 0.8 * destination["sphere_of_influence"]
    ):
        raise ValueError(
            "Target orbit must be safely above the atmosphere and inside its SOI"
        )
    orbit = elements(position, velocity, mu)
    if orbit["energy"] >= 0:
        raise ValueError("Departure state must be a bound parking orbit")
    if orbit["periapsis"] < body["equatorial_radius"] + body["atmosphere_depth"] + 1000:
        raise ValueError(
            "Departure parking orbit must clear the surface and atmosphere"
        )
    semi_major = -mu / (2 * orbit["energy"])
    period = 2 * math.pi * math.sqrt(semi_major**3 / mu)
    target_semi_major = destination["orbit"]["semi_major_axis"]
    hohmann_time = math.pi * math.sqrt(((semi_major + target_semi_major) / 2) ** 3 / mu)
    wait = period if maximum_wait is None else float(maximum_wait)
    if (
        not math.isfinite(wait)
        or wait <= 0
        or not math.isfinite(lead_time)
        or lead_time <= 0
    ):
        raise ValueError("Search wait and lead time must be positive and finite")
    # Planet transfers need a launch-window search rather than one parking orbit.
    if maximum_wait is None and source == system.ancestors(source)[-1]:
        target_period = destination["orbit"]["period"]
        inverse_synodic = abs(1 / period - 1 / target_period)
        wait = min(
            2 / inverse_synodic if inverse_synodic > 1e-12 else period,
            3 * target_period,
        )
    candidates = []
    durations = np.linspace(0.7 * hohmann_time, 1.65 * hohmann_time, duration_samples)
    for departure_ut in np.linspace(
        epoch + lead_time, epoch + lead_time + wait, departure_samples
    ):
        start, start_velocity = propagate(position, velocity, departure_ut - epoch, mu)
        for duration in durations:
            arrival_ut = float(departure_ut + duration)
            finish, finish_velocity = system.state(
                target, arrival_ut, relative_to=source
            )
            for long_way in (False, True):
                try:
                    departure_velocity, arrival_velocity = lambert(
                        start, finish, float(duration), mu, long_way=long_way
                    )
                    vinf = float(np.linalg.norm(arrival_velocity - finish_velocity))
                    cost = np.linalg.norm(
                        departure_velocity - start_velocity
                    ) + capture_delta_v(target_mu, periapsis, vinf)
                    candidates.append(
                        (
                            float(cost),
                            float(departure_ut),
                            arrival_ut,
                            start,
                            start_velocity,
                            long_way,
                        )
                    )
                except ValueError:
                    continue
    if not candidates:
        raise ValueError(
            "No finite zero-revolution Lambert candidate in the search window"
        )
    refined = []
    for _, departure_ut, arrival_ut, start, start_velocity, long_way in sorted(
        candidates, key=lambda item: item[0]
    )[:12]:
        finish, body_velocity = system.state(target, arrival_ut, relative_to=source)
        plane = _unit(np.cross(finish, body_velocity))
        for side in (-1, 1):
            offset = np.zeros(3)
            for _ in range(3):
                departure_velocity, arrival_velocity = lambert(
                    start,
                    finish + offset,
                    arrival_ut - departure_ut,
                    mu,
                    long_way=long_way,
                )
                relative_velocity = arrival_velocity - body_velocity
                vinf = float(np.linalg.norm(relative_velocity))
                impact_parameter = periapsis * math.sqrt(
                    1 + 2 * target_mu / (periapsis * vinf * vinf)
                )
                offset = (
                    side * impact_parameter * _unit(np.cross(relative_velocity, plane))
                )
            departure_velocity, arrival_velocity = lambert(
                start, finish + offset, arrival_ut - departure_ut, mu, long_way=long_way
            )
            vinf = float(np.linalg.norm(arrival_velocity - body_velocity))
            delta_v = departure_velocity - start_velocity
            capture = capture_delta_v(target_mu, periapsis, vinf)
            plan = TransferPlan(
                source,
                target,
                "direct_child",
                departure_ut,
                arrival_ut,
                start.tolist(),
                start_velocity.tolist(),
                delta_v.tolist(),
                vinf,
                capture,
                altitude,
                periapsis,
                offset.tolist(),
                system.route(source, target),
                [
                    "Zero-revolution Lambert transfer in the parent two-body field.",
                    "Gravitational focusing uses an asymptotic B-plane approximation.",
                    "Actual KSP encounter/periapsis and finite-burn errors require live verification.",
                ],
            )
            refined.append(plan)
    return min(refined, key=lambda item: item.estimated_total_delta_v)


def plan_interplanetary_window(
    system,
    source,
    target,
    epoch,
    *,
    parking_altitude=100000.0,
    target_altitude=None,
    departure_samples=48,
    duration_samples=24,
):
    """Estimate a sibling-planet departure window and Oberth ejection budget.

    This is a planning estimate; it does not create a low-orbit ejection vector.
    The live executor must resolve the departure hyperbola and SOI transitions.
    """
    origin, destination = system.bodies[source], system.bodies[target]
    parent = origin["parent"]
    if parent is None or destination["parent"] != parent:
        raise ValueError("Window search requires sibling bodies with a common parent")
    altitude = (
        system.default_target_altitude(target)
        if target_altitude is None
        else target_altitude
    )
    target_radius = destination["equatorial_radius"] + altitude
    if (
        not math.isfinite(altitude)
        or altitude <= destination["atmosphere_depth"] + 1000
        or target_radius >= 0.8 * destination["sphere_of_influence"]
    ):
        raise ValueError(
            "Target orbit must be finite, safely above the atmosphere and inside its SOI"
        )
    parking_radius = origin["equatorial_radius"] + parking_altitude
    if (
        not math.isfinite(parking_altitude)
        or parking_altitude <= origin["atmosphere_depth"] + 1000
        or parking_radius >= origin["sphere_of_influence"]
    ):
        raise ValueError(
            "Departure parking orbit must be finite, safely above the atmosphere and inside its SOI"
        )
    mu = system.bodies[parent]["gravitational_parameter"]
    a1, a2 = origin["orbit"]["semi_major_axis"], destination["orbit"]["semi_major_axis"]
    p1, p2 = origin["orbit"]["period"], destination["orbit"]["period"]
    inverse = abs(1 / p1 - 1 / p2)
    wait = min(2 / inverse if inverse > 1e-12 else p1, 3 * max(p1, p2))
    nominal = math.pi * math.sqrt(((a1 + a2) / 2) ** 3 / mu)
    results = []
    for departure in np.linspace(epoch + 180, epoch + 180 + wait, departure_samples):
        r1, v1 = system.state(source, float(departure), relative_to=parent)
        for duration in np.linspace(0.6 * nominal, 1.7 * nominal, duration_samples):
            arrival = float(departure + duration)
            r2, v2 = system.state(target, arrival, relative_to=parent)
            for long_way in (False, True):
                try:
                    leave, arrive = lambert(
                        r1, r2, float(duration), mu, long_way=long_way
                    )
                except ValueError:
                    continue
                excess = leave - v1
                vinf = float(np.linalg.norm(arrive - v2))
                injection = math.sqrt(
                    float(excess @ excess)
                    + 2 * origin["gravitational_parameter"] / parking_radius
                ) - math.sqrt(origin["gravitational_parameter"] / parking_radius)
                capture = capture_delta_v(
                    destination["gravitational_parameter"], target_radius, vinf
                )
                results.append(
                    dict(
                        source=source,
                        target=target,
                        route=system.route(source, target),
                        departure_ut=float(departure),
                        arrival_ut=arrival,
                        departure_vinf_vector=excess.tolist(),
                        arrival_vinf=vinf,
                        estimated_departure_delta_v=injection,
                        estimated_capture_delta_v=capture,
                        estimated_total_delta_v=injection + capture,
                        target_altitude=altitude,
                        kind="sibling_window_estimate",
                        executable=False,
                        actual_ksp_verified=False,
                        limitations=[
                            "Requires a resolved finite-burn departure hyperbola and actual KSP encounter verification.",
                            "Node execution and planet/moon route coverage are separate from this window estimate.",
                        ],
                    )
                )
    if not results:
        raise ValueError("No Lambert launch window found")
    return min(results, key=lambda item: item["estimated_total_delta_v"])
