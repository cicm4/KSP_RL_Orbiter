"""Explicit maneuver-node preparation against actual KSP patched conics.

Read helpers never change controls. Node creation requires an explicit call and
never removes pre-existing user nodes. The mission CLI owns only its new nodes.
"""

import math
import time

import numpy as np

from .live import rpc_deadline
from .mission import plan_child_transfer, require_delta_v
from .orbital import elements, lambert, propagate
from .system import SystemCatalog


class EncounterUnavailable(RuntimeError):
    """This candidate window has no admissible verified target encounter."""


def usable_liquid_propellant(resources):
    """Count only burnable stock 9:11 LF/Ox mixture, not excess unusable fuel."""
    names = set(resources.names)
    if not {"LiquidFuel", "Oxidizer"} <= names:
        return 0.0
    lf = float(resources.amount("LiquidFuel"))
    ox = float(resources.amount("Oxidizer"))
    density_lf = float(resources.density("LiquidFuel"))
    density_ox = float(resources.density("Oxidizer"))
    mixtures = min(lf / 9, ox / 11)
    return mixtures * (9 * density_lf + 11 * density_ox)


def read_parking_state(connection, *, rpc_timeout=15.0):
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        vessel = sc.active_vessel
        body = vessel.orbit.body
        frame = body.non_rotating_reference_frame
        active = [
            engine
            for engine in vessel.parts.engines
            if engine.active and engine.has_fuel
        ]
        if not active or any(
            not engine.can_shutdown
            or set(engine.propellant_names) != {"LiquidFuel", "Oxidizer"}
            for engine in active
        ):
            raise ValueError(
                "Mission execution requires active, shutdown-capable stock LF/Ox vacuum engines"
            )
        fuel = usable_liquid_propellant(vessel.resources)
        mass = float(vessel.mass)
        return dict(
            ut=float(sc.ut),
            source=body.name,
            r=list(vessel.position(frame)),
            v=list(vessel.velocity(frame)),
            mass=mass,
            dry_mass=mass - fuel,
            thrust=float(vessel.available_thrust),
            isp=float(vessel.specific_impulse),
            fuel_mass=fuel,
            vessel=vessel.name,
            periapsis=float(vessel.orbit.periapsis_altitude),
            apoapsis=float(vessel.orbit.apoapsis_altitude),
            situation=str(vessel.situation),
        )


def plan_live_transfer(
    connection,
    target,
    target_altitude=None,
    *,
    rpc_timeout=15.0,
    reserve=100.0,
    coast_orbits=0,
):
    """Read and plan without controls; later windows coast the measured conic in math only."""
    if (
        isinstance(coast_orbits, bool)
        or not isinstance(coast_orbits, int)
        or coast_orbits < 0
    ):
        raise ValueError("coast_orbits must be a nonnegative integer")
    system = SystemCatalog.load()
    state = read_parking_state(connection, rpc_timeout=rpc_timeout)
    if target not in system.bodies:
        raise ValueError(
            f"Unknown target {target!r}; available: {', '.join(system.bodies)}"
        )
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        for name in (state["source"], target):
            actual = sc.bodies[name]
            saved = system.bodies[name]
            if not math.isclose(
                actual.gravitational_parameter,
                saved["gravitational_parameter"],
                rel_tol=1e-8,
            ) or not math.isclose(
                actual.equatorial_radius, saved["equatorial_radius"], rel_tol=1e-8
            ):
                raise ValueError(
                    "Live system differs from the captured catalog; recapture before flight"
                )
            if name == state["source"]:
                source_mu = float(actual.gravitational_parameter)
    position, velocity, epoch = state["r"], state["v"], state["ut"]
    if coast_orbits:
        orbit = elements(position, velocity, source_mu)
        if not math.isfinite(orbit["energy"]) or orbit["energy"] >= 0:
            raise ValueError(
                "Later planning windows require a finite bound parking orbit"
            )
        semi_major = -source_mu / (2 * orbit["energy"])
        period = 2 * math.pi * math.sqrt(semi_major**3 / source_mu)
        coast_time = coast_orbits * period
        if not math.isfinite(coast_time):
            raise ValueError("Parking-orbit coast duration must be finite")
        position, velocity = propagate(position, velocity, coast_time, source_mu)
        epoch += coast_time
    plan = plan_child_transfer(
        system,
        state["source"],
        target,
        position,
        velocity,
        epoch,
        target_altitude=target_altitude,
    )
    budget = require_delta_v(
        plan,
        mass=state["mass"],
        dry_mass=state["dry_mass"],
        isp=state["isp"],
        reserve=reserve,
    )
    return plan, budget


def set_node_vector(space_center, node, delta_v, frame):
    """Node axes are anti-radial/prograde/normal, not radial/prograde/normal."""
    node_frame = node.orbital_reference_frame
    local = space_center.transform_direction(
        tuple(float(x) for x in delta_v), frame, node_frame
    )
    node.prograde = float(local[1])
    node.normal = float(local[2])
    node.radial = -float(local[0])
    # The node basis can drift against the caller's frame between RPCs. Check
    # the encoded axes in the same node-local basis used for the setters.
    measured = np.asarray(node.burn_vector(node_frame))
    if not np.allclose(measured, local, atol=1e-3, rtol=1e-6):
        raise RuntimeError(
            "KSP node vector disagrees with requested local burn; frame conversion rejected"
        )


def create_node(connection, vessel, ut, delta_v, *, rpc_timeout=15.0):
    with rpc_deadline(rpc_timeout):
        if vessel.control.nodes:
            raise ValueError(
                "Remove or finish existing maneuver nodes before starting an automatic mission"
            )
        node = vessel.control.add_node(float(ut))
        try:
            set_node_vector(
                connection.space_center,
                node,
                delta_v,
                vessel.orbit.body.non_rotating_reference_frame,
            )
        except BaseException:
            node.remove()
            raise
    return node


def encounter_prediction(node, target, *, max_patches=8):
    orbit = node.orbit
    patches = []
    for _ in range(max_patches):
        if orbit is None:
            break
        name = orbit.body.name
        patch = dict(
            body=name,
            periapsis=float(orbit.periapsis_altitude),
            apoapsis=float(orbit.apoapsis_altitude),
            eccentricity=float(orbit.eccentricity),
        )
        patches.append(patch)
        if name == target:
            if (
                not math.isfinite(patch["periapsis"])
                or not math.isfinite(patch["eccentricity"])
                or patch["eccentricity"] < 0
            ):
                raise ValueError(
                    "KSP returned nonfinite or invalid target encounter elements"
                )
            return dict(
                encounter=True,
                target=target,
                periapsis=patch["periapsis"],
                eccentricity=patch["eccentricity"],
                patches=patches,
            )
        orbit = orbit.next_orbit
    return dict(encounter=False, target=target, patches=patches)


def prepare_encounter_node(
    connection, plan, log=None, *, rpc_timeout=15.0, periapsis_tolerance=1000.0
):
    """Create and refine one owned node, requiring an actual predicted encounter."""
    system = SystemCatalog.load()
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        vessel = sc.active_vessel
        if vessel.orbit.body.name != plan.source:
            raise ValueError("Departure body changed after planning")
        if plan.departure_ut < float(sc.ut) + 30:
            raise ValueError(
                "Transfer plan is stale; replan with sufficient burn lead time"
            )
        frame = vessel.orbit.body.non_rotating_reference_frame
    node = None
    candidates = []
    target_position, _ = system.state(
        plan.target, plan.arrival_ut, relative_to=plan.source
    )
    mu = system.bodies[plan.source]["gravitational_parameter"]
    start = np.asarray(plan.departure_position)
    start_velocity = np.asarray(plan.departure_velocity)
    duration = plan.arrival_ut - plan.departure_ut
    original_departure = start_velocity + np.asarray(plan.delta_v)
    # Preserve the selected short/long branch by matching the original velocity.
    branch = min(
        (False, True),
        key=lambda long_way: np.linalg.norm(
            lambert(
                start,
                target_position + plan.aim_offset,
                duration,
                mu,
                long_way=long_way,
            )[0]
            - original_departure
        ),
    )

    def sample(scale):
        departure, _ = lambert(
            start,
            target_position + scale * np.asarray(plan.aim_offset),
            duration,
            mu,
            long_way=branch,
        )
        delta = departure - start_velocity
        with rpc_deadline(rpc_timeout):
            set_node_vector(sc, node, delta, frame)
        # Let KSP's patched-conic solver process the changed node on its frame.
        time.sleep(0.04)
        with rpc_deadline(rpc_timeout):
            prediction = encounter_prediction(node, plan.target)
        if getattr(plan, "kind", "direct_child") == "direct_child":
            intermediates = [
                patch["body"]
                for patch in prediction["patches"]
                if patch["body"] not in (plan.source, plan.target)
            ]
            prediction["admissible"] = not intermediates
            if intermediates:
                prediction.update(
                    rejection_reason="unmodeled_intermediate_body",
                    intermediate_bodies=intermediates,
                )
        if log is not None:
            log.write(
                "encounter_node_trial",
                offset_scale=scale,
                delta_v=delta,
                prediction=prediction,
            )
        if prediction["encounter"] and prediction.get("admissible", True):
            error = prediction["periapsis"] - plan.target_altitude
            candidates.append(
                (abs(error), float(scale), float(error), delta, prediction)
            )
            return error
        return None

    try:
        node = create_node(
            connection, vessel, plan.departure_ut, plan.delta_v, rpc_timeout=rpc_timeout
        )
        sample(1.0)
        if not candidates or candidates[0][0] > periapsis_tolerance:
            for scale in (0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
                sample(scale)
        ordered = sorted(candidates, key=lambda item: item[1])
        brackets = [
            (left, right)
            for left, right in zip(ordered, ordered[1:])
            if left[2] * right[2] <= 0
        ]
        if brackets:
            low, high = min(brackets, key=lambda pair: min(pair[0][0], pair[1][0]))
            lo, hi, lo_error = low[1], high[1], low[2]
            for _ in range(14):
                mid = (lo + hi) / 2
                error = sample(mid)
                if error is None:
                    break
                if abs(error) <= periapsis_tolerance:
                    break
                if error * lo_error <= 0:
                    hi = mid
                else:
                    lo, lo_error = mid, error
        if not candidates:
            raise EncounterUnavailable(
                "No admissible actual KSP target encounter exists for the candidate node; "
                "unmodeled intermediate encounters are excluded"
            )
        error, scale, _, delta, prediction = min(candidates, key=lambda item: item[0])
        if not math.isfinite(error):
            raise ValueError("KSP encounter periapsis error is nonfinite")
        if error > max(5000.0, periapsis_tolerance):
            raise EncounterUnavailable(
                f"KSP encounter periapsis differs from target by {error:.0f} m; refusing an unverified burn"
            )
        with rpc_deadline(rpc_timeout):
            set_node_vector(sc, node, delta, frame)
            actual_delta = float(node.delta_v)
        plan.delta_v = delta.tolist()
        prediction.update(
            offset_scale=scale,
            actual_node_delta_v=actual_delta,
            periapsis_error_m=error,
            verified_by="actual_ksp_patched_conic_node",
        )
        if log is not None:
            log.write("encounter_node_verified", **prediction)
        return node, prediction
    except BaseException:
        if node is not None:
            with rpc_deadline(rpc_timeout):
                node.remove()
        raise


def correct_periapsis_node(
    connection, vessel, target_radius, *, rpc_timeout=15.0, lead_time=180.0
):
    """Small inbound angular-momentum correction after the actual SOI change."""
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        body = vessel.orbit.body
        mu = float(body.gravitational_parameter)
        body_radius = float(body.equatorial_radius)
        now = float(sc.ut)
        frame = body.non_rotating_reference_frame
        r = np.asarray(vessel.position(frame))
        v = np.asarray(vessel.velocity(frame))
        if float(vessel.orbit.time_to_periapsis) < lead_time + 60:
            raise ValueError("Too close to periapsis for a centered correction burn")
    r, v = propagate(r, v, lead_time, mu)
    radius = float(np.linalg.norm(r))
    radial = r / radius
    vr = float(v @ radial)
    tangential = v - vr * radial
    speed = float(np.linalg.norm(tangential))
    if vr >= 0 or target_radius >= radius or target_radius <= body_radius:
        raise ValueError(
            "Periapsis correction requires a safe inbound approach outside the target radius"
        )
    desired = target_radius * math.sqrt(
        (vr * vr + 2 * mu * (1 / target_radius - 1 / radius))
        / (radius * radius - target_radius * target_radius)
    )
    delta = (desired - speed) * tangential / speed
    if np.linalg.norm(delta) < 0.02:
        return None
    return create_node(
        connection, vessel, now + lead_time, delta, rpc_timeout=rpc_timeout
    )
