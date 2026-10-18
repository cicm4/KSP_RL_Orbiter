"""Actual-KSP parking-orbit transfer/capture validation; default is planning only.

--fly explicitly authorizes nodes, maneuver burns, coasting warp, and capture
controls. No headless environment substitutes for any mission stage. Learned
capture and the transparent scripted capture baseline are separate choices.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .capture_env import (
    CaptureConfig,
    build_capture_observation,
    capture_teacher,
    decode_capture_action,
    orbital_state,
    rtn_basis,
)
from .live import (
    FuelMonitor,
    FlightLog,
    StageController,
    fly,
    jsonable,
    load_controller,
    rpc_deadline,
    stage_numbers,
    stop_controls,
    validate_vehicle_mass,
    wait_for_ut,
)
from .maneuvers import execute_node
from .orbital import propagate, rocket_delta_v
from .ownership import FlightOwnership

PARITY = np.array([1.0, -1.0, 1.0])
DEFAULT_PARKING_WINDOWS = 8
PAD_SCENE_TIMEOUT = 120.0


class PhaseLog:
    """Keep ascent telemetry distinguishable in the single mission recording."""

    def __init__(self, log, phase):
        self.log, self.phase = log, phase

    def write(self, event, **data):
        self.log.write(self.phase + "_" + event, phase=self.phase, **data)


def control_part_identity(part):
    # kRPC object IDs are scoped to this connection, not persistent vessel IDs.
    return dict(name=part.name, title=part.title, connection_object=str(part))


def inspect_active_pad(
    connection, config, *, plan=(3, 2), fuel_stages=(2, 1), rpc_timeout=15.0
):
    """Read-only eligibility check for the supported stock OrbiterRL Transfer craft.

    Known craft/control mismatches return ready=False and failures. RPC failures
    raise to the caller. Nothing is launched, loaded, saved, staged or controlled.
    """
    profile_bytes = (
        resources.files("orbiter").joinpath("data/transfer_vehicle.json").read_bytes()
    )
    profile = json.loads(profile_bytes)
    contract = jsonable(asdict(config))
    failures = []
    with rpc_deadline(rpc_timeout):
        sc = connection.space_center
        vessel = sc.active_vessel
        body = vessel.orbit.body
        control = vessel.control
        situation = str(vessel.situation)
        source = str(control.source).rsplit(".", 1)[-1].lower()
        state = str(control.state).rsplit(".", 1)[-1].lower()
        crew = sorted(member.name for member in vessel.crew)
        parts = sorted(part.name for part in vessel.parts.all)
        engines = sorted(
            [
                dict(
                    name=engine.part.name,
                    activation_stage=int(engine.part.stage),
                    decouple_stage=int(engine.part.decouple_stage),
                    propellants=sorted(engine.propellant_names),
                    active=bool(engine.active),
                    has_fuel=bool(engine.has_fuel),
                    can_shutdown=bool(engine.can_shutdown),
                    available_thrust=float(engine.available_thrust),
                    thrust_limit=float(engine.thrust_limit),
                    max_vacuum_thrust=float(engine.max_vacuum_thrust),
                    vacuum_specific_impulse=float(engine.vacuum_specific_impulse),
                )
                for engine in vessel.parts.engines
            ],
            key=lambda item: (
                item["name"],
                item["activation_stage"],
                item["decouple_stage"],
            ),
        )
        decouplers = sorted(
            [
                dict(
                    name=decoupler.part.name,
                    activation_stage=int(decoupler.part.stage),
                    decouple_stage=int(decoupler.part.decouple_stage),
                )
                for decoupler in vessel.parts.decouplers
            ],
            key=lambda item: (
                item["name"],
                item["activation_stage"],
                item["decouple_stage"],
            ),
        )
        mass = float(vessel.mass)
        current_stage = int(control.current_stage)
        latitude = float(vessel.flight().latitude)
        mu, radius = float(body.gravitational_parameter), float(body.equatorial_radius)
        expected_mass = config.payload_mass + sum(
            stage.dry_mass + stage.fuel_mass for stage in config.stages
        )
        data = dict(
            ready=False,
            failures=failures,
            supported_profile=profile["supported_profile"],
            profile_scope=profile["scope"],
            profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
            vessel=vessel.name,
            situation=situation,
            body=body.name,
            mass=mass,
            expected_mass=expected_mass,
            current_stage=current_stage,
            part_names=parts,
            engines=engines,
            decouplers=decouplers,
            crew_names=crew,
            control_state=state,
            control_source=source,
            controlling_part=control_part_identity(vessel.parts.controlling),
            ut=float(sc.ut),
        )
        if situation.rsplit(".", 1)[-1].lower() not in {"prelaunch", "pre_launch"}:
            failures.append(
                "The active vessel must already be on the launchpad in PRELAUNCH."
            )
        if (
            state != "full"
            or source not in {"kerbal", "probe"}
            or source == "kerbal"
            and not crew
        ):
            failures.append(
                "The active vessel needs full control from an onboard Kerbal or powered probe."
            )
        if parts != sorted(profile["parts"]):
            failures.append(
                "The active vessel does not match the supported 20-part OrbiterRL Transfer profile."
            )
        # Equal mass and part names cannot establish which engine will ignite or
        # detach. Bind the actual module/stage groups before acquiring controls.
        expected_engines = sorted(profile["engines"], key=lambda item: item["name"])
        engine_keys = ("name", "activation_stage", "decouple_stage", "propellants")
        if len(engines) != len(expected_engines) or any(
            any(actual[key] != expected[key] for key in engine_keys)
            for actual, expected in zip(engines, expected_engines)
        ):
            failures.append(
                "Engine types, activation stages, decouple groups or propellants differ from the transfer profile."
            )
        if decouplers != profile["decouplers"]:
            failures.append(
                "Decoupler activation stages or decouple groups differ from the transfer profile."
            )
        if any(
            actual["active"]
            or not actual["has_fuel"]
            or not actual["can_shutdown"]
            or not math.isfinite(actual["available_thrust"])
            or actual["available_thrust"] <= 0
            or not math.isfinite(actual["thrust_limit"])
            or abs(actual["thrust_limit"] - 1.0) > 1e-6
            or any(
                not math.isfinite(actual[key])
                or abs(actual[key] / expected[key] - 1.0) > 0.01
                for key in ("max_vacuum_thrust", "vacuum_specific_impulse")
            )
            for actual, expected in zip(engines, expected_engines)
        ):
            failures.append(
                "Prelaunch engines must be inactive, fueled, shutdown-capable and at full rated thrust/ISP."
            )
        if (
            tuple(plan) != tuple(profile["stage_plan"])
            or tuple(fuel_stages) != tuple(profile["fuel_stages"])
            or current_stage != profile["stage_plan"][0] + 1
        ):
            failures.append(
                "The active craft must have activation stages 3,2 and fuel stages 2,1 before launch."
            )
        if not math.isfinite(mass) or mass <= 0 or abs(mass / expected_mass - 1) > 0.1:
            failures.append(
                "The active craft wet mass differs from the ascent model by over 10%."
            )
        if (
            not all(math.isfinite(value) and value > 0 for value in (mu, radius))
            or abs(mu / config.mu - 1) > 0.01
            or abs(radius / config.radius - 1) > 0.01
        ):
            failures.append("The launch body differs from the ascent model.")
        if not math.isfinite(latitude) or abs(latitude) > 1:
            failures.append(
                "The ascent adapter requires an approximately equatorial launchpad."
            )
        if any(
            contract.get(key) != value
            for key, value in profile["vehicle_contract"].items()
        ):
            failures.append(
                "The ascent model does not use the supported transfer craft physics profile."
            )
        if control.nodes:
            failures.append(
                "Finish or remove existing maneuver nodes before starting this mission."
            )
        try:
            monitor = FuelMonitor(vessel, fuel_stages)
            if not all(
                math.isfinite(value) and value > 0 for value in monitor.initial_fuel
            ):
                raise ValueError("Stage propellant must be positive and finite")
            validate_vehicle_mass(config, monitor)
            StageController(
                sc, vessel, plan
            )  # Constructor reads/checks only; never activates a stage.
            data["fuel_stage_masses"] = list(monitor.initial_fuel)
        except ValueError as error:
            failures.append(str(error))
    data["ready"] = not failures
    return jsonable(data)


def check_new_pad_save(ksp_root, name):
    """Conservatively reject an existing name in any local save game."""
    if (
        not name
        or name.lower() in {"persistent", "quicksave"}
        or any(char in name for char in "/\\")
        or name in {".", ".."}
    ):
        raise ValueError(
            "Pad snapshot must have a dedicated plain name, excluding persistent/quicksave"
        )
    if ksp_root is None or not (Path(ksp_root) / "saves").is_dir():
        raise ValueError(
            "--pad-save-name requires the local --ksp-root with a readable saves directory"
        )
    expected = (name + ".sfs").casefold()
    for folder in (Path(ksp_root) / "saves").iterdir():
        if folder.is_dir() and any(
            path.name.casefold() == expected for path in folder.iterdir()
        ):
            raise FileExistsError(
                f"Refusing to overwrite pad snapshot {name!r} in {folder}"
            )


def launch_pad_snapshot(ksp_root, name):
    """Validate one existing OrbiterRL snapshot and identify its active pad craft."""
    if (
        not name
        or name.lower() in {"persistent", "quicksave"}
        or any(char in name for char in "/\\")
        or name in {".", ".."}
    ):
        raise ValueError(
            "Launch pad snapshot must have a dedicated plain name, excluding persistent/quicksave"
        )
    if ksp_root is None:
        raise ValueError("--launch-pad-save requires the local --ksp-root")
    dedicated = Path(ksp_root).resolve() / "saves" / "OrbiterRL"
    snapshot = dedicated / (name + ".sfs")
    if snapshot.resolve().parent != dedicated:
        raise ValueError(
            "Launch pad snapshot must reside only in the dedicated OrbiterRL save"
        )
    if not snapshot.is_file():
        raise FileNotFoundError(f"OrbiterRL pad snapshot does not exist: {snapshot}")
    # kRPC Load resolves a basename in the active save, not an absolute path.
    for folder in dedicated.parent.iterdir():
        if folder.is_dir() and folder != dedicated:
            if any(
                path.name.casefold() == snapshot.name.casefold()
                for path in folder.iterdir()
            ):
                raise ValueError(
                    "Pad snapshot basename is ambiguous in another local save"
                )
    data = snapshot.read_bytes()
    depth, flight_depth, vessel_depth, part_depth = 0, None, None, None
    pending, active_index, current, vessels, saved_ut = None, None, None, [], None
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line == "{":
            depth += 1
            if pending == "FLIGHTSTATE":
                flight_depth = depth
            elif (
                pending == "VESSEL"
                and flight_depth is not None
                and depth == flight_depth + 1
            ):
                vessel_depth, current = depth, {"part_names": [], "crew": []}
            elif (
                pending == "PART"
                and vessel_depth is not None
                and depth == vessel_depth + 1
            ):
                part_depth = depth
            pending = None
        elif line == "}":
            if depth == part_depth:
                part_depth = None
            if depth == vessel_depth:
                vessels.append(current)
                vessel_depth, current = None, None
            if depth == flight_depth:
                break
            depth -= 1
        elif "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if depth == flight_depth and key == "UT":
                saved_ut = float(value)
            elif depth == flight_depth and key == "activeVessel":
                try:
                    active_index = int(value)
                except ValueError as error:
                    raise ValueError(
                        "Pad snapshot has an invalid active vessel index"
                    ) from error
            elif (
                current is not None
                and depth == vessel_depth
                and key in {"name", "sit", "stg", "lat", "lon", "alt"}
            ):
                current[key] = value
            elif current is not None and depth == part_depth:
                if key == "name":
                    current["part_names"].append(value)
                elif key == "crew":
                    current["crew"].append(value)
        else:
            pending = line
    if (
        active_index is None
        or not 0 <= active_index < len(vessels)
        or not vessels[active_index].get("name")
        or vessels[active_index].get("sit", "").replace("_", "").upper() != "PRELAUNCH"
    ):
        raise ValueError("Named snapshot must contain an active pre-launch craft")
    active = vessels[active_index]
    try:
        state = dict(
            ut=float(saved_ut),
            stage=int(active["stg"]),
            latitude=float(active["lat"]),
            longitude=float(active["lon"]),
            altitude=float(active["alt"]),
            crew=sorted(active["crew"]),
            part_names=sorted(active["part_names"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Pad snapshot lacks the active craft state needed for load verification"
        ) from error
    if (
        not all(
            math.isfinite(state[key])
            for key in ("ut", "latitude", "longitude", "altitude")
        )
        or not state["part_names"]
        or state["stage"] < 1
    ):
        raise ValueError("Pad snapshot active craft state must be finite and complete")
    return dict(
        name=name,
        path=str(snapshot),
        sha256=hashlib.sha256(data).hexdigest(),
        vessel_name=vessels[active_index]["name"],
        state=state,
        hash_scope="Validated local file; exact serialized loaded-file identity is not exposed by kRPC.",
    )


def verify_loaded_pad_state(sc, vessel, config, snapshot, *, elapsed_wall):
    """Check practical snapshot consistency; kRPC exposes no persistent vessel ID."""
    expected = snapshot["state"]
    flight = vessel.flight()
    body = vessel.orbit.body
    root_position = np.asarray(
        vessel.parts.root.position(body.non_rotating_reference_frame), dtype=float
    )
    if root_position.shape != (3,) or not np.isfinite(root_position).all():
        raise ValueError("Loaded snapshot root position is invalid")
    radius = float(body.equatorial_radius)
    actual = dict(
        ut=float(sc.ut),
        stage=int(vessel.control.current_stage),
        latitude=float(flight.latitude),
        longitude=float(flight.longitude),
        # SFS pad altitude locates the root transform. kRPC MeanAltitude uses
        # vessel.CoM, several metres lower on this long vertical rocket.
        altitude=math.hypot(*root_position) - radius,
        altitude_reference="root_part_origin",
        mean_altitude=float(flight.mean_altitude),
        crew=sorted(member.name for member in vessel.crew),
        part_names=sorted(part.name for part in vessel.parts.all),
        mu=float(body.gravitational_parameter),
        radius=radius,
    )
    numeric = (
        "ut",
        "latitude",
        "longitude",
        "altitude",
        "mean_altitude",
        "mu",
        "radius",
    )
    if not all(math.isfinite(actual[key]) for key in numeric):
        raise ValueError("Loaded snapshot state is nonfinite")
    longitude_error = (actual["longitude"] - expected["longitude"] + 180) % 360 - 180
    if (
        not expected["ut"] - 1 <= actual["ut"] <= expected["ut"] + elapsed_wall + 5
        or actual["stage"] != expected["stage"]
        or actual["crew"] != expected["crew"]
        or actual["part_names"] != expected["part_names"]
        or abs(actual["latitude"] - expected["latitude"]) > 0.001
        or abs(longitude_error) > 0.001
        or abs(actual["altitude"] - expected["altitude"]) > 5
        or abs(actual["mu"] / config.mu - 1) > 0.01
        or abs(actual["radius"] / config.radius - 1) > 0.01
    ):
        raise ValueError(
            "Loaded snapshot state differs from the validated OrbiterRL pad snapshot"
        )
    if (
        hashlib.sha256(Path(snapshot["path"]).read_bytes()).hexdigest()
        != snapshot["sha256"]
    ):
        raise ValueError("Validated local pad snapshot changed during load")
    return dict(
        snapshot_state_consistent=True,
        loaded_state=actual,
        tolerances=dict(
            ut_before_seconds=1.0,
            ut_after_load_wall_seconds=5.0,
            geographic_degrees=0.001,
            altitude_metres=5.0,
            body_relative=0.01,
        ),
        scope="Local unique basename and loaded pad-state consistency, not cryptographic loaded-file identity.",
    )


def launch_ascent(
    connection,
    config,
    model,
    craft=None,
    *,
    plan,
    fuel_stages,
    log,
    rpc_timeout=15.0,
    wall_timeout=1800.0,
    pad_save_name=None,
    launch_pad_save=None,
    launch_active=False,
    ksp_root=None,
):
    """Start a named craft or explicit pad snapshot, retaining its controlling part."""
    if sum((bool(craft), bool(launch_pad_save), bool(launch_active))) != 1:
        raise ValueError(
            "Select exactly one launch craft, launch pad snapshot or active pad"
        )
    if launch_active and (pad_save_name or ksp_root):
        raise ValueError("Active pad launch accepts no save or KSP-root options")
    if craft and (any(char in craft for char in "/\\") or craft.endswith(".craft")):
        raise ValueError(
            "Launch craft must be a VAB craft name without a path or extension"
        )
    if not plan or fuel_stages is None or len(fuel_stages) != len(config.stages):
        raise ValueError(
            "Launch requires explicit activation and fuel-stage mappings for this model"
        )
    if pad_save_name:
        if launch_pad_save:
            raise ValueError("--pad-save-name cannot accompany --launch-pad-save")
        check_new_pad_save(ksp_root, pad_save_name)
    snapshot = (
        launch_pad_snapshot(ksp_root, launch_pad_save) if launch_pad_save else None
    )
    preflight = (
        inspect_active_pad(
            connection,
            config,
            plan=plan,
            fuel_stages=fuel_stages,
            rpc_timeout=rpc_timeout,
        )
        if launch_active
        else None
    )
    if preflight is not None and not preflight["ready"]:
        raise ValueError(
            "Active pad preflight rejected: " + "; ".join(preflight["failures"])
        )
    expected_name = (
        preflight["vessel"]
        if preflight
        else snapshot["vessel_name"] if snapshot else craft
    )
    sc, vessel, part = connection.space_center, None, None
    deadline = time.monotonic() + wall_timeout
    result = dict(success=False, craft=craft, stage_plan=plan, fuel_stages=fuel_stages)
    try:
        scene_deadline = (
            min(deadline, time.monotonic() + PAD_SCENE_TIMEOUT)
            if snapshot
            else deadline
        )
        if launch_active:
            result["active_pad_preflight"] = preflight
        elif snapshot:
            load_started = time.monotonic()
            with rpc_deadline(min(PAD_SCENE_TIMEOUT, wall_timeout)):
                sc.load(launch_pad_save)
            result["loaded_pad_save"] = snapshot
            log.write("explicit_launch_pad_snapshot_load", **snapshot)
        else:
            with rpc_deadline(min(rpc_timeout, wall_timeout)):
                sc.launch_vessel_from_vab(craft, recover=False)
        while True:
            remaining = min(deadline, scene_deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Named launch craft did not reach the pad before the deadline"
                )
            with rpc_deadline(min(rpc_timeout, remaining)):
                candidate = sc.active_vessel
                if (
                    launch_active
                    and control_part_identity(candidate.parts.controlling)
                    != preflight["controlling_part"]
                ):
                    raise RuntimeError("Active pad changed after preflight")
                if candidate.name == expected_name and any(
                    word in str(candidate.situation).lower()
                    for word in ("pre_launch", "prelaunch")
                ):
                    if snapshot:
                        if int(candidate.control.current_stage) != plan[0] + 1:
                            raise ValueError(
                                "Loaded pad staging differs from the explicit activation plan"
                            )
                        mass = float(candidate.mass)
                        expected_mass = config.payload_mass + sum(
                            stage.dry_mass + stage.fuel_mass for stage in config.stages
                        )
                        if (
                            not math.isfinite(mass)
                            or abs(mass / expected_mass - 1) > 0.1
                        ):
                            raise ValueError(
                                "Loaded pad wet mass differs from the trained craft by over 10%"
                            )
                        result.update(
                            verify_loaded_pad_state(
                                sc,
                                candidate,
                                config,
                                snapshot,
                                elapsed_wall=time.monotonic() - load_started,
                            )
                        )
                    vessel = candidate
                    part = vessel.parts.controlling
                    identity = control_part_identity(part)
                    pad_ut = float(sc.ut)
                    break
            time.sleep(0.05)
        if pad_save_name:
            check_new_pad_save(ksp_root, pad_save_name)
            with rpc_deadline(min(rpc_timeout, deadline - time.monotonic())):
                sc.save(pad_save_name)
            result["pad_save"] = dict(
                name=pad_save_name,
                local_ksp_root=str(Path(ksp_root).resolve()),
                existing_names_checked_in_all_local_saves=True,
            )
        result.update(controlling_part=identity, pad_ut=pad_ut)
        log.write("launch_handoff", **result)
        ascent = fly(
            connection,
            config,
            plan=plan,
            fuel_stages=fuel_stages,
            model=model,
            scripted=False,
            log=PhaseLog(log, "ascent"),
            rpc_timeout=rpc_timeout,
            wall_timeout=max(0.001, deadline - time.monotonic()),
        )
        result["ascent"] = ascent
        if not ascent["success"]:
            raise RuntimeError(
                "Launch policy did not achieve its parking orbit: " + str(ascent)
            )
        with rpc_deadline(rpc_timeout):
            if sc.active_vessel != part.vessel or part.vessel.parts.controlling != part:
                raise RuntimeError(
                    "Parking-orbit handoff lost the launched controlling part"
                )
            result.update(
                success=True,
                parking_ut=float(sc.ut),
                parking_controlling_part=control_part_identity(part),
            )
        return result, part
    finally:
        errors = []
        if part is not None:
            try:
                with rpc_deadline(min(rpc_timeout, 3.0)):
                    vessel = part.vessel
            except Exception as error:
                errors.append("launched_part_vessel: " + str(error))
        errors.extend(
            stop_controls([vessel] if vessel is not None else [], min(rpc_timeout, 3.0))
        )
        result["cleanup_errors"] = errors
        if errors:
            result["success"] = False
        log.write("launch_ascent_end", **result)


def _state(connection):
    from .mission_nodes import read_parking_state

    return read_parking_state(connection)


def reflect(vector):
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Mission vectors must contain three finite coordinates")
    return value * PARITY


def capture_direction_native(r, v, rtn_direction):
    """Map a right-handed capture RTN command back to native kRPC axes."""
    direction = reflect(rtn_basis(r, v) @ np.asarray(rtn_direction, dtype=float))
    norm = float(np.linalg.norm(direction))
    return direction / norm if norm > 0 else direction


def read_actual_orbit(sc, vessel):
    orbit, body = vessel.orbit, vessel.orbit.body
    semi_major = float(orbit.semi_major_axis)
    mu = float(body.gravitational_parameter)
    return dict(
        ut=float(sc.ut),
        body=body.name,
        vessel=vessel.name,
        apoapsis=float(orbit.apoapsis_altitude),
        periapsis=float(orbit.periapsis_altitude),
        eccentricity=float(orbit.eccentricity),
        semi_major_axis=semi_major,
        energy=-mu / (2 * semi_major) if semi_major != 0 else math.nan,
        period=float(orbit.period),
        throttle=float(vessel.control.throttle),
        situation=str(vessel.situation),
    )


def orbit_failures(sample, config):
    if sample.get("body") != config.body_name:
        return ["wrong_body"]
    keys = ("ut", "apoapsis", "periapsis", "eccentricity", "energy", "throttle")
    if any(
        not isinstance(sample.get(key), (float, int)) or not math.isfinite(sample[key])
        for key in keys
    ):
        return ["invalid_telemetry"]
    failures = []
    if sample["energy"] >= 0 or not 0 <= sample["eccentricity"] < 1:
        failures.append("not_bound")
    if not (
        config.target_altitude - config.orbit_tolerance
        <= sample["periapsis"]
        <= sample["apoapsis"]
        <= config.target_altitude + config.orbit_tolerance
    ):
        failures.append("outside_target_band")
    if sample["periapsis"] + config.radius <= config.safe_radius:
        failures.append("unsafe_periapsis")
    if sample["apoapsis"] + config.radius >= config.sphere_of_influence:
        failures.append("outside_soi")
    return failures


def capture_config(
    connection, target, target_altitude, *, metadata=None, rpc_timeout=15.0
):
    base = (
        CaptureConfig.from_dict(metadata["environment"])
        if metadata
        else CaptureConfig()
    )
    with rpc_deadline(rpc_timeout):
        body = connection.space_center.bodies[target]
        row = dict(
            name=body.name,
            gravitational_parameter=float(body.gravitational_parameter),
            equatorial_radius=float(body.equatorial_radius),
            atmosphere_depth=float(body.atmosphere_depth),
            sphere_of_influence=float(body.sphere_of_influence),
        )
    state = _state(connection)
    overrides = asdict(base)
    for name in (
        "body_name",
        "mu",
        "radius",
        "atmosphere_depth",
        "sphere_of_influence",
    ):
        overrides.pop(name)
    overrides.update(
        target_altitude=target_altitude,
        target_altitude_randomization=0.0,
        orbit_tolerance=5000.0,
        randomization=0.0,
        mass=float(state["mass"]),
        dry_mass=float(state["dry_mass"]),
        thrust=float(state["thrust"]),
        isp=float(state["isp"]),
    )
    return CaptureConfig.for_body(row, **overrides)


def resolve_target_altitude(target, requested=None):
    """Use the capture task's nominal safe goal when altitude is omitted."""
    from .system import SystemCatalog

    bodies = SystemCatalog.load().bodies
    if target not in bodies:
        raise ValueError(f"Unknown target {target!r}; available: {', '.join(bodies)}")
    if bodies[target]["parent"] is None:
        raise ValueError(
            "A central-star orbital mission needs a heliocentric goal; planetary capture does not apply"
        )
    overrides = {} if requested is None else {"target_altitude": requested}
    return CaptureConfig.for_body(bodies[target], **overrides).target_altitude


def read_capture_state(
    connection, vessel, config, *, initial_mass, goal_normal, start_ut
):
    raw = _state(connection)
    if raw["source"] != config.body_name or vessel.orbit.body.name != config.body_name:
        raise RuntimeError(
            "Actual KSP vessel is not inside the requested destination SOI"
        )
    r, v = reflect(raw["r"]), reflect(raw["v"])
    frame = vessel.orbit.body.non_rotating_reference_frame
    direction = reflect(vessel.direction(frame))
    state = orbital_state(r, v, config.mu)
    h = state["h"] / max(state["momentum"], 1e-12)
    plane_error = math.degrees(math.acos(float(np.clip(h @ goal_normal, -1.0, 1.0))))
    actual = read_actual_orbit(connection.space_center, vessel)
    values = dict(
        r=r,
        v=v,
        direction=direction,
        goal_normal=goal_normal,
        mass=float(raw["mass"]),
        initial_mass=initial_mass,
        dry_mass=float(raw["dry_mass"]),
        thrust=float(raw["thrust"]),
        isp=float(raw["isp"]),
        time=float(raw["ut"]) - start_ut,
        body=raw["source"],
        plane_error_deg=plane_error,
        orbit_in_band=not orbit_failures(actual, config)
        and plane_error <= config.plane_tolerance_deg,
        actual_orbit=actual,
        **state,
    )
    observation = build_capture_observation(config, values)
    if not np.isfinite(observation).all():
        raise RuntimeError("Capture observation is nonfinite")
    return values, observation


def run_capture(
    connection,
    vessel,
    config,
    *,
    log,
    model=None,
    scripted=False,
    autopilot_prepared=False,
    rpc_timeout=15.0,
    wall_timeout=1200.0,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    if (model is None) == (not scripted):
        raise ValueError(
            "Select exactly one learned model or scripted capture controller"
        )
    sc = connection.space_center
    deadline = clock() + wall_timeout
    touched = [vessel]
    result = dict(
        success=False,
        controller="scripted_capture_teacher" if scripted else "learned_capture_policy",
    )

    def remaining():
        value = deadline - clock()
        if value <= 0:
            raise TimeoutError("Capture wall-clock deadline expired")
        return value

    try:
        with rpc_deadline(min(rpc_timeout, remaining())):
            raw = _state(connection)
            if raw["source"] != config.body_name:
                raise RuntimeError("Capture must start in the actual target SOI")
            r, v = reflect(raw["r"]), reflect(raw["v"])
            goal_normal = np.cross(r, v)
            norm = float(np.linalg.norm(goal_normal))
            if norm < 1e-10:
                raise ValueError("Capture start has no defined orbital plane")
            goal_normal /= norm
            start_ut, initial_mass = float(raw["ut"]), float(raw["mass"])
            ap = vessel.auto_pilot
            vessel.control.throttle = 0.0
            vessel.control.sas = False
            ap.target_roll = (
                math.nan
            )  # kRPC's documented unset sentinel: point the thrust axis only.
            if not autopilot_prepared:
                ap.reference_frame = vessel.orbit.body.non_rotating_reference_frame
                ap.target_direction = tuple(
                    -np.asarray(raw["v"]) / np.linalg.norm(raw["v"])
                )
                ap.engage()
        log.write(
            "capture_start",
            controller=result["controller"],
            config=asdict(config),
            start_ut=start_ut,
            initial_mass=initial_mass,
            frame_parity=PARITY,
            goal_normal_right_handed=goal_normal,
        )
        streak, previous_ut = 0, start_ut
        for step in range(math.ceil(config.max_time / config.control_dt) + 2):
            with rpc_deadline(min(rpc_timeout, remaining())):
                if sc.active_vessel != vessel:
                    raise RuntimeError("Active vessel changed during capture")
                data, observation = read_capture_state(
                    connection,
                    vessel,
                    config,
                    initial_mass=initial_mass,
                    goal_normal=goal_normal,
                    start_ut=start_ut,
                )
            ut = data["actual_orbit"]["ut"]
            if ut < previous_ut - 1e-6:
                raise RuntimeError("KSP time moved backwards during capture")
            previous_ut = ut
            log.write("capture_telemetry", step=step, observation=observation, **data)
            streak = streak + 1 if data["orbit_in_band"] else 0
            if streak >= config.success_hold_steps:
                result.update(
                    success=True,
                    termination_reason="target_orbit",
                    steps=step + 1,
                    final_orbit=data["actual_orbit"],
                )
                break
            if data["radius"] <= config.safe_radius:
                raise RuntimeError("Capture reached the surface/atmosphere clearance")
            if data["radius"] >= config.sphere_of_influence:
                raise RuntimeError("Vessel left the destination SOI")
            if data["time"] >= config.max_time:
                raise TimeoutError("Capture simulation-time limit reached")
            if data["mass"] <= data["dry_mass"] + 1e-6 and not data["orbit_in_band"]:
                raise RuntimeError("Capture exhausted usable engine propellant")
            proxy = SimpleNamespace(
                config=config,
                r=data["r"],
                v=data["v"],
                goal_normal=goal_normal,
                telemetry=lambda: data,
            )
            action = (
                capture_teacher(proxy)
                if scripted
                else model.predict(observation, deterministic=True)[0]
            )
            throttle, direction_rtn = decode_capture_action(config, data, action)
            if data["orbit_in_band"]:
                throttle = 0.0
            with rpc_deadline(min(rpc_timeout, remaining())):
                if np.linalg.norm(direction_rtn) > 0:
                    ap.target_direction = tuple(
                        capture_direction_native(data["r"], data["v"], direction_rtn)
                    )
                error = abs(float(ap.error))
                if not math.isfinite(error):
                    raise RuntimeError("Capture autopilot error is nonfinite")
                if error > 3.0:
                    throttle = 0.0
                vessel.control.throttle = throttle
                command_ut = float(sc.ut)
            log.write(
                "capture_action",
                step=step,
                ut=command_ut,
                controller=result["controller"],
                policy_action=action,
                throttle=throttle,
                direction_rtn=direction_rtn,
                pointing_error_deg=error,
            )
            wait_for_ut(
                sc,
                command_ut + config.control_dt,
                remaining(),
                rpc_timeout=rpc_timeout,
                clock=clock,
                sleeper=sleeper,
            )
        else:
            result.update(termination_reason="step_limit")
        return result
    finally:
        errors = stop_controls(touched, min(rpc_timeout, 3.0))
        result["cleanup_errors"] = errors
        if errors:
            result.update(success=False, termination_reason="cleanup_failed")
        log.write("capture_end", **result)


def incoming_radius_time(r, v, mu, desired_radius, time_to_periapsis):
    r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
    if (
        not math.isfinite(time_to_periapsis)
        or time_to_periapsis <= 0
        or float(r @ v) >= 0
    ):
        raise ValueError("Need a finite incoming coast before periapsis")
    if np.linalg.norm(r) <= desired_radius:
        return 0.0
    end, _ = propagate(r, v, time_to_periapsis, mu)
    if np.linalg.norm(end) > desired_radius:
        raise ValueError("Incoming conic never reaches the requested capture radius")
    low, high = 0.0, time_to_periapsis
    for _ in range(60):
        middle = (low + high) / 2
        point, _ = propagate(r, v, middle, mu)
        if np.linalg.norm(point) > desired_radius:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def bounded_warp(sc, ut, *, rpc_timeout=120.0):
    try:
        with rpc_deadline(rpc_timeout):
            current = float(sc.ut)
            if (
                not math.isfinite(current)
                or not math.isfinite(ut)
                or ut < current - 1e-6
            ):
                raise ValueError("Warp target must be finite and in the future")
            if ut > current:
                sc.warp_to(ut, max_rails_rate=100000.0, max_physics_rate=2.0)
            if float(sc.ut) < ut - 1e-3:
                raise RuntimeError("KSP warp returned before the requested time")
    finally:
        errors = []
        for name in ("rails_warp_factor", "physics_warp_factor"):
            try:
                with rpc_deadline(3):
                    setattr(sc, name, 0)
            except Exception as error:
                errors.append(name + ": " + str(error))
        if errors:
            raise RuntimeError("Warp cleanup failed: " + "; ".join(errors))


def target_entry_ut(sc, vessel, target):
    now, patch = float(sc.ut), vessel.orbit
    for _ in range(8):
        if patch.body.name == target:
            return now
        next_patch = patch.next_orbit
        if next_patch is None:
            break
        entry = float(patch.time_to_soi_change)
        if next_patch.body.name == target:
            if not math.isfinite(entry) or entry < 0:
                raise RuntimeError("Actual KSP encounter has no future SOI entry time")
            return now + entry
        patch = next_patch
    raise RuntimeError("Actual KSP patched conics show no destination encounter")


def coast_to_capture(
    connection, vessel, config, *, log, rpc_timeout=15.0, warp_rpc_timeout=120.0
):
    sc = connection.space_center
    with rpc_deadline(rpc_timeout):
        raw = _state(connection)
        if raw["source"] != config.body_name:
            raise RuntimeError("Cannot coast for capture outside the actual target SOI")
        ttp = float(vessel.orbit.time_to_periapsis)
        dt = incoming_radius_time(
            raw["r"],
            raw["v"],
            config.mu,
            config.arrival_radius_ratio * config.goal_radius,
            ttp,
        )
        _, future_velocity = propagate(raw["r"], raw["v"], dt, config.mu)
        ap = vessel.auto_pilot
        vessel.control.throttle = 0.0
        vessel.control.sas = False
        ap.reference_frame = vessel.orbit.body.non_rotating_reference_frame
        ap.target_roll = math.nan
        ap.target_direction = tuple(-future_velocity / np.linalg.norm(future_velocity))
        ap.engage()
    deadline = time.monotonic() + 60.0
    while True:
        with rpc_deadline(rpc_timeout):
            error, ut = abs(float(ap.error)), float(sc.ut)
        if not math.isfinite(error):
            raise RuntimeError("Pre-capture pointing error is nonfinite")
        if error <= 3.0:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Pre-capture retrograde alignment timed out")
        wait_for_ut(sc, ut + 0.1, 10.0, rpc_timeout=rpc_timeout)
    with rpc_deadline(rpc_timeout):
        raw = _state(connection)
        dt = incoming_radius_time(
            raw["r"],
            raw["v"],
            config.mu,
            config.arrival_radius_ratio * config.goal_radius,
            float(vessel.orbit.time_to_periapsis),
        )
        target_ut = float(raw["ut"]) + dt
    log.write(
        "capture_coast",
        target_ut=target_ut,
        arrival_radius_ratio=config.arrival_radius_ratio,
    )
    bounded_warp(sc, target_ut, rpc_timeout=warp_rpc_timeout)
    with rpc_deadline(rpc_timeout):
        if vessel.orbit.body.name != config.body_name:
            raise RuntimeError("Destination SOI changed while coasting to capture")
    return dict(autopilot_prepared=True, capture_start_ut=target_ut)


def verify_orbital_period(
    connection, vessel, config, *, log, rpc_timeout=15.0, warp_rpc_timeout=120.0
):
    errors = stop_controls([vessel], min(rpc_timeout, 3.0))
    if errors:
        raise RuntimeError("Cannot verify an uncontrolled coast: " + "; ".join(errors))
    sc = connection.space_center
    with rpc_deadline(rpc_timeout):
        first = read_actual_orbit(sc, vessel)
    failures = orbit_failures(first, config)
    period = first["period"]
    if (
        failures
        or not math.isfinite(period)
        or period <= 0
        or abs(first["throttle"]) > 1e-5
    ):
        return dict(
            success=False,
            failures=failures or ["invalid_period_or_throttle"],
            samples=[first],
        )
    probes, previous = [first], first["ut"]
    for index in range(1, 5):
        bounded_warp(sc, first["ut"] + period * index / 4, rpc_timeout=warp_rpc_timeout)
        with rpc_deadline(rpc_timeout):
            if sc.active_vessel != vessel:
                raise RuntimeError(
                    "Active vessel changed during independent period coast"
                )
            sample = read_actual_orbit(sc, vessel)
        if sample["ut"] < previous - 1e-6:
            raise RuntimeError(
                "KSP time moved backwards during independent period coast"
            )
        previous = sample["ut"]
        failures = orbit_failures(sample, config)
        if abs(sample["throttle"]) > 1e-5:
            failures.append("throttle_not_zero")
        probes.append(sample)
        log.write(
            "independent_orbital_period_probe",
            quarter=index,
            failures=failures,
            **sample,
        )
        if failures:
            break
    elapsed = probes[-1]["ut"] - first["ut"]
    return dict(
        success=not failures and elapsed >= period - 1e-3,
        requested_period_s=period,
        observed_coast_s=elapsed,
        failures=failures,
        samples=probes,
        scope="One actual KSP orbital period with four on-rails coast probes, throttle zero and autopilot off.",
    )


def fly_mission(
    connection,
    target,
    target_altitude,
    *,
    log,
    model=None,
    metadata=None,
    scripted=False,
    rpc_timeout=15.0,
    wall_timeout=1800.0,
    warp_rpc_timeout=120.0,
    expected_control_part=None,
):
    from .mission_nodes import (
        EncounterUnavailable,
        correct_periapsis_node,
        plan_live_transfer,
        prepare_encounter_node,
    )
    from .mission import require_delta_v

    if (model is None) == (not scripted):
        raise ValueError(
            "Select exactly one learned model or scripted capture controller before flight"
        )
    sc, vessel = connection.space_center, None
    owned_nodes = []
    result = dict(
        success=False,
        scope="Actual KSP parking orbit to selected destination orbit; ascent proof is separate.",
        end_to_end_from_launch_verified=False,
        handoffs=[],
        window_rejections=[],
    )
    started = time.monotonic()

    def remaining():
        value = wall_timeout - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Mission wall-clock limit reached")
        return value

    def burn(node, body_name):
        owned_nodes.append(node)
        state = _state(connection)
        available = rocket_delta_v(state["mass"], state["dry_mass"], state["isp"])
        execution = execute_node(
            connection,
            vessel,
            node,
            log=log,
            rpc_timeout=rpc_timeout,
            wall_timeout=remaining(),
            expected_body=body_name,
            available_delta_v=available,
            warp_rpc_timeout=min(warp_rpc_timeout, remaining()),
        )
        if not execution["success"]:
            raise RuntimeError("Maneuver execution failed: " + str(execution))
        with rpc_deadline(rpc_timeout):
            node.remove()
        owned_nodes.remove(node)
        return execution

    try:
        with rpc_deadline(min(rpc_timeout, remaining())):
            vessel = sc.active_vessel
            if expected_control_part is not None and (
                expected_control_part.vessel != vessel
                or vessel.parts.controlling != expected_control_part
            ):
                raise RuntimeError(
                    "Transfer handoff differs from the launched controlling part"
                )
        for coast_orbits in range(DEFAULT_PARKING_WINDOWS):
            with rpc_deadline(remaining()):
                plan, budget = plan_live_transfer(
                    connection,
                    target,
                    target_altitude=target_altitude,
                    coast_orbits=coast_orbits,
                    rpc_timeout=min(rpc_timeout, remaining()),
                )
            result.update(plan=plan.to_dict(), budget=budget, coast_orbits=coast_orbits)
            log.write(
                "mission_plan",
                plan=plan.to_dict(),
                budget=budget,
                coast_orbits=coast_orbits,
            )
            try:
                with rpc_deadline(remaining()):
                    node, prediction = prepare_encounter_node(
                        connection, plan, log, rpc_timeout=min(rpc_timeout, remaining())
                    )
            except EncounterUnavailable as error:
                rejected = dict(
                    coast_orbits=coast_orbits,
                    reason=str(error),
                    exception=type(error).__name__,
                )
                result["window_rejections"].append(rejected)
                log.write("mission_window_rejected", **rejected)
                if coast_orbits == DEFAULT_PARKING_WINDOWS - 1:
                    raise EncounterUnavailable(
                        f"No verified encounter in {DEFAULT_PARKING_WINDOWS} parking-orbit windows"
                    ) from error
            else:
                break
        owned_nodes.append(node)
        current = _state(connection)
        result["refined_budget"] = require_delta_v(
            plan,
            mass=current["mass"],
            dry_mass=current["dry_mass"],
            isp=current["isp"],
            reserve=budget["reserve_delta_v"],
        )
        owned_nodes.remove(
            node
        )  # burn() registers this node through execution/cleanup.
        result["patched_conic_prediction"] = prediction
        result["departure"] = burn(node, plan.source)
        with rpc_deadline(rpc_timeout):
            entry = target_entry_ut(sc, vessel, target)
        log.write("target_soi_coast", target=target, target_entry_ut=entry)
        bounded_warp(sc, entry + 2.0, rpc_timeout=min(warp_rpc_timeout, remaining()))
        with rpc_deadline(rpc_timeout):
            actual_body = vessel.orbit.body.name
            handoff_ut = float(sc.ut)
        if actual_body != target:
            raise RuntimeError(f"Actual KSP SOI is {actual_body}; required {target}")
        result["handoffs"].append(
            dict(stage="actual_target_soi", body=actual_body, ut=handoff_ut)
        )
        config = capture_config(
            connection,
            target,
            target_altitude,
            metadata=metadata,
            rpc_timeout=rpc_timeout,
        )
        correction = correct_periapsis_node(connection, vessel, config.goal_radius)
        if correction is not None:
            result["periapsis_correction"] = burn(correction, target)
        coast = coast_to_capture(
            connection,
            vessel,
            config,
            log=log,
            rpc_timeout=rpc_timeout,
            warp_rpc_timeout=min(warp_rpc_timeout, remaining()),
        )
        result["capture"] = run_capture(
            connection,
            vessel,
            config,
            log=log,
            model=model,
            scripted=scripted,
            autopilot_prepared=coast["autopilot_prepared"],
            rpc_timeout=rpc_timeout,
            wall_timeout=remaining(),
        )
        if not result["capture"]["success"]:
            raise RuntimeError(
                "Destination capture controller did not establish the requested orbit"
            )
        result["handoffs"].append(
            dict(stage="actual_target_orbit", **result["capture"]["final_orbit"])
        )
        result["independent_coast"] = verify_orbital_period(
            connection,
            vessel,
            config,
            log=log,
            rpc_timeout=rpc_timeout,
            warp_rpc_timeout=min(warp_rpc_timeout, remaining()),
        )
        result["success"] = result["independent_coast"]["success"]
        if expected_control_part is not None:
            with rpc_deadline(min(rpc_timeout, remaining())):
                if (
                    expected_control_part.vessel != vessel
                    or vessel.parts.controlling != expected_control_part
                ):
                    raise RuntimeError(
                        "Final orbit differs from the launched controlling part"
                    )
                result["final_controlling_part"] = control_part_identity(
                    expected_control_part
                )
        return result
    finally:
        cleanup = stop_controls(
            [vessel] if vessel is not None else [], min(rpc_timeout, 3.0)
        )
        for node in owned_nodes:
            try:
                with rpc_deadline(min(rpc_timeout, 3.0)):
                    node.remove()
            except Exception as error:
                cleanup.append("owned_node_remove: " + str(error))
        for name in ("rails_warp_factor", "physics_warp_factor"):
            try:
                with rpc_deadline(min(rpc_timeout, 3.0)):
                    setattr(sc, name, 0)
            except Exception as error:
                cleanup.append(name + ": " + str(error))
        result["cleanup_errors"] = cleanup
        if cleanup:
            result["success"] = False
        log.write("mission_end", **result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--target-altitude",
        type=float,
        help="Orbit altitude in metres; defaults to max(50 km, atmosphere + 20 km)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fly", action="store_true")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--capture-model", type=Path)
    controls.add_argument("--capture-scripted", action="store_true")
    parser.add_argument(
        "--launch-model",
        type=Path,
        help="Optional learned ascent checkpoint for a fresh single-invocation launch",
    )
    launch_source = parser.add_mutually_exclusive_group()
    launch_source.add_argument(
        "--launch-active",
        action="store_true",
        help="Fly the already-present supported transfer craft on the active launchpad; no load or craft creation",
    )
    launch_source.add_argument(
        "--launch-craft",
        help="Named VAB craft, without .craft; launch uses recover=False",
    )
    launch_source.add_argument(
        "--launch-pad-save",
        help="Explicitly reload an existing crewed pre-launch snapshot from OrbiterRL; requires --ksp-root",
    )
    parser.add_argument(
        "--stage-plan", type=stage_numbers, help="Explicit launch activation sequence"
    )
    parser.add_argument(
        "--fuel-stages",
        type=stage_numbers,
        help="Explicit launch propellant stage mapping",
    )
    parser.add_argument(
        "--pad-save-name",
        help="Optional new dedicated pad snapshot; requires local --ksp-root",
    )
    parser.add_argument(
        "--ksp-root",
        type=Path,
        help="Local KSP installation, used to validate dedicated pad snapshot paths",
    )
    parser.add_argument(
        "--ascent-record",
        type=Path,
        help="Optional separately recorded ascent evidence; never implies attribution by itself",
    )
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--rpc-timeout", type=float, default=15.0)
    parser.add_argument("--warp-rpc-timeout", type=float, default=120.0)
    parser.add_argument("--wall-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.rpc_timeout, args.warp_rpc_timeout, args.wall_timeout)
    ) or (
        args.target_altitude is not None
        and (not math.isfinite(args.target_altitude) or args.target_altitude <= 0)
    ):
        parser.error("Altitude/timeouts must be positive and finite")
    if args.fly and not (args.capture_model or args.capture_scripted):
        parser.error("--fly requires --capture-model or --capture-scripted")
    launch_options = (
        args.launch_model,
        args.launch_craft,
        args.launch_pad_save,
        True if args.launch_active else None,
        args.stage_plan,
        args.fuel_stages,
        args.pad_save_name,
        args.ksp_root,
    )
    if any(value is not None for value in launch_options):
        if (
            not args.fly
            or not args.launch_model
            or not (args.launch_craft or args.launch_pad_save or args.launch_active)
            or args.stage_plan is None
            or args.fuel_stages is None
        ):
            parser.error(
                "Fresh launch requires --fly, --launch-model, a launch source, --stage-plan and --fuel-stages"
            )
        if args.ascent_record:
            parser.error(
                "Fresh launch records its own ascent; --ascent-record is for a separate parking-orbit invocation"
            )
        if args.launch_active and (args.ksp_root or args.pad_save_name):
            parser.error("--launch-active accepts no --ksp-root or --pad-save-name")
        if args.pad_save_name:
            if args.launch_pad_save:
                parser.error("--pad-save-name cannot accompany --launch-pad-save")
            try:
                check_new_pad_save(args.ksp_root, args.pad_save_name)
            except (ValueError, OSError) as error:
                parser.error(str(error))
        if args.launch_pad_save:
            try:
                launch_pad_snapshot(args.ksp_root, args.launch_pad_save)
            except (ValueError, OSError) as error:
                parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    connection, log, owns_summary, ownership = None, None, False, None
    summary = dict(
        success=False,
        source="actual_ksp_krpc",
        mode="fly" if args.fly else "read_only_plan",
        target=args.target,
        target_altitude=args.target_altitude,
        capture_controller=(
            "learned_capture_policy"
            if args.capture_model
            else "scripted_capture_teacher" if args.capture_scripted else None
        ),
        end_to_end_from_launch_verified=False,
    )
    output = args.output_dir / "mission.json"
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(summary, handle)
        owns_summary = True
        if args.fly:
            ownership = FlightOwnership(args.address, args.rpc_port)
            ownership.acquire()
        args.target_altitude = resolve_target_altitude(
            args.target, args.target_altitude
        )
        summary["target_altitude"] = args.target_altitude
        catalog = (
            resources.files("orbiter").joinpath("data/stock_system.json").read_bytes()
        )
        summary["system_catalog_sha256"] = hashlib.sha256(catalog).hexdigest()
        model, metadata = None, None
        if args.capture_model:
            from .train_capture import load_capture_controller

            before = hashlib.sha256(args.capture_model.read_bytes()).hexdigest()
            model, metadata = load_capture_controller(args.capture_model)
            if hashlib.sha256(args.capture_model.read_bytes()).hexdigest() != before:
                raise RuntimeError("Capture checkpoint changed while loading")
            summary.update(
                capture_model_sha256=before,
                capture_model=str(args.capture_model.resolve()),
                capture_metadata_sha256=hashlib.sha256(
                    args.capture_model.with_suffix(".metadata.json").read_bytes()
                ).hexdigest(),
                capture_model_task=metadata["task"],
                capture_model_num_timesteps=int(model.num_timesteps),
            )
        launch_model, launch_config = None, None
        if args.launch_model:
            launch_before = hashlib.sha256(args.launch_model.read_bytes()).hexdigest()
            launch_model, launch_config = load_controller(args.launch_model)
            if (
                hashlib.sha256(args.launch_model.read_bytes()).hexdigest()
                != launch_before
            ):
                raise RuntimeError("Launch checkpoint changed while loading")
            summary.update(
                launch_model=str(args.launch_model.resolve()),
                launch_model_sha256=launch_before,
                launch_num_timesteps=int(launch_model.num_timesteps),
                launch_config=asdict(launch_config),
                launch_craft=args.launch_craft,
                launch_pad_save=args.launch_pad_save,
                launch_active=args.launch_active,
                stage_plan=args.stage_plan,
                fuel_stages=args.fuel_stages,
            )
        if args.ascent_record:
            summary.update(
                separate_ascent_record=str(args.ascent_record.resolve()),
                separate_ascent_record_sha256=hashlib.sha256(
                    args.ascent_record.read_bytes()
                ).hexdigest(),
            )
        import krpc

        with rpc_deadline(args.rpc_timeout):
            connection = krpc.connect(
                name=(
                    "Orbiter live mission"
                    if args.fly
                    else "Read-only Orbiter mission plan"
                ),
                address=args.address,
                rpc_port=args.rpc_port,
                stream_port=None,
                use_pregenerated_stubs=False,
            )
        log = FlightLog(args.output_dir / "mission.jsonl")
        log.write("mission_invocation", **summary)
        if args.fly:
            expected_part, operation_started = None, time.monotonic()
            if args.launch_model:
                summary["launch"], expected_part = launch_ascent(
                    connection,
                    launch_config,
                    launch_model,
                    args.launch_craft,
                    plan=args.stage_plan,
                    fuel_stages=args.fuel_stages,
                    log=log,
                    rpc_timeout=args.rpc_timeout,
                    wall_timeout=args.wall_timeout,
                    pad_save_name=args.pad_save_name,
                    launch_pad_save=args.launch_pad_save,
                    launch_active=args.launch_active,
                    ksp_root=args.ksp_root,
                )
                if not summary["launch"]["success"]:
                    raise RuntimeError("Launch controls could not be released safely")
            summary["mission"] = fly_mission(
                connection,
                args.target,
                args.target_altitude,
                log=log,
                model=model,
                metadata=metadata,
                scripted=args.capture_scripted,
                rpc_timeout=args.rpc_timeout,
                wall_timeout=max(
                    0.001, args.wall_timeout - (time.monotonic() - operation_started)
                ),
                warp_rpc_timeout=args.warp_rpc_timeout,
                expected_control_part=expected_part,
            )
            summary["success"] = summary["mission"]["success"]
            summary["end_to_end_from_launch_verified"] = bool(
                args.launch_model
                and summary["success"]
                and summary["launch"]["success"]
                and summary["mission"]["independent_coast"]["success"]
            )
        else:
            from .mission_nodes import plan_live_transfer

            with rpc_deadline(max(args.rpc_timeout, 120.0)):
                plan, budget = plan_live_transfer(
                    connection, args.target, target_altitude=args.target_altitude
                )
            summary.update(
                plan=plan.to_dict(),
                budget=budget,
                planning_complete=True,
                note="Read-only candidate and budget; actual KSP encounter/transfer/capture have not been executed by this invocation.",
            )
        return (
            0
            if (summary["success"] if args.fly else summary.get("planning_complete"))
            else 1
        )
    except Exception as error:
        summary.update(success=False, error=f"{type(error).__name__}: {error}")
        return 1
    finally:
        try:
            if log is not None:
                try:
                    log.close()
                except Exception as error:
                    summary["log_close_error"] = f"{type(error).__name__}: {error}"
            if connection is not None:
                try:
                    with rpc_deadline(min(args.rpc_timeout, 3.0)):
                        connection.close()
                except Exception as error:
                    summary["connection_close_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
            if owns_summary:
                output.write_text(
                    json.dumps(jsonable(summary), indent=2, allow_nan=False) + "\n"
                )
            print(json.dumps(jsonable(summary), indent=2, allow_nan=False), flush=True)
        finally:
            if ownership is not None:
                ownership.release()


if __name__ == "__main__":
    raise SystemExit(main())
