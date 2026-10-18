"""Bind curated mission claims to immutable model and actual-game recordings."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "validation/missions/manifest.json").read_text())


def artifact_path(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT), f"Artifact escapes repository: {relative}"
    assert relative in MANIFEST["artifacts"], f"Unhashed mission artifact: {relative}"
    return path


def read_json(relative):
    return json.loads(artifact_path(relative).read_text())


def recording(relative):
    payload = artifact_path(relative).read_bytes()
    if relative.endswith(".gz"):
        payload = gzip.decompress(payload)
    return [json.loads(line) for line in payload.decode().splitlines()]


def one_event(rows, event):
    selected = [row for row in rows if row["event"] == event]
    assert len(selected) == 1, f"Expected exactly one {event}, found {len(selected)}"
    return selected[0]


def event_payload(event):
    return {
        key: value for key, value in event.items() if key not in {"event", "wall_time"}
    }


def assert_selected_window(rows, mission):
    plans = [row for row in rows if row["event"] == "mission_plan"]
    rejections = [row for row in rows if row["event"] == "mission_window_rejected"]
    assert len(plans) == len(rejections) + 1
    assert plans[-1]["plan"] == mission["plan"]
    assert plans[-1]["budget"] == mission["budget"]
    assert [event_payload(row) for row in rejections] == mission.get(
        "window_rejections", []
    )
    selected_window = mission.get("coast_orbits", 0)
    assert selected_window == len(rejections) < 8
    assert plans[-1].get("coast_orbits", 0) == selected_window
    for index, rejection in enumerate(rejections):
        assert rejection["coast_orbits"] == index
        assert plans[index]["coast_orbits"] == index
        assert rejection["exception"] == "EncounterUnavailable" and rejection["reason"]
        assert (
            rows.index(plans[index])
            < rows.index(rejection)
            < rows.index(plans[index + 1])
        )
    verified = one_event(rows, "encounter_node_verified")
    assert rows.index(plans[-1]) < rows.index(verified)
    burns = [row for row in rows if row["event"] == "maneuver_start"]
    assert burns and all(rows.index(row) > rows.index(verified) for row in burns)


def no_cleanup_errors(value):
    if isinstance(value, dict):
        if "cleanup_errors" in value:
            assert value["cleanup_errors"] == []
        for child in value.values():
            no_cleanup_errors(child)
    elif isinstance(value, list):
        for child in value:
            no_cleanup_errors(child)


def assert_orbit(sample, config, target):
    assert sample["body"] == target == config["body_name"]
    for name in (
        "ut",
        "periapsis",
        "apoapsis",
        "eccentricity",
        "energy",
        "period",
        "throttle",
    ):
        assert math.isfinite(sample[name]), (name, sample[name])
    assert sample["energy"] < 0 and 0 <= sample["eccentricity"] < 1
    low = config["target_altitude"] - config["orbit_tolerance"]
    high = config["target_altitude"] + config["orbit_tolerance"]
    assert low <= sample["periapsis"] <= sample["apoapsis"] <= high
    safe_altitude = max(
        config["surface_clearance"],
        config["atmosphere_depth"] + config["atmosphere_clearance"],
    )
    assert sample["periapsis"] > safe_altitude
    assert sample["apoapsis"] + config["radius"] < config["sphere_of_influence"]
    assert sample["throttle"] == 0.0
    assert "orbiting" in sample["situation"].lower()


def assert_full_period(coast, rows, config, target):
    assert coast["success"] is True and coast["failures"] == []
    assert len(coast["samples"]) == 5
    assert coast["requested_period_s"] > 0
    assert coast["observed_coast_s"] >= coast["requested_period_s"] - 1e-3
    samples = coast["samples"]
    assert coast["observed_coast_s"] == pytest.approx(
        samples[-1]["ut"] - samples[0]["ut"]
    )
    assert samples[0]["period"] == coast["requested_period_s"]
    for sample in samples:
        assert_orbit(sample, config, target)
    assert all(right["ut"] > left["ut"] for left, right in zip(samples, samples[1:]))
    probes = [row for row in rows if row["event"] == "independent_orbital_period_probe"]
    assert [row["quarter"] for row in probes] == [1, 2, 3, 4]
    for index, (probe, sample) in enumerate(zip(probes, samples[1:]), start=1):
        assert probe["failures"] == []
        assert {key: probe[key] for key in sample} == sample
        assert (
            sample["ut"]
            >= samples[0]["ut"] + coast["requested_period_s"] * index / 4 - 1e-3
        )


def test_both_requested_destination_flights_are_curated():
    assert {flight["target"] for flight in MANIFEST["flights"]} == {"Mun", "Minmus"}, (
        "Evidence gate remains incomplete until both actual learned destination flights are archived"
    )
    assert len({flight["id"] for flight in MANIFEST["flights"]}) == len(
        MANIFEST["flights"]
    )
    historical = {flight["id"]: flight for flight in MANIFEST["flights"]}
    assert historical["mun_learned"]["target"] == "Mun"
    assert historical["mun_learned"]["end_to_end_from_launch"] is False
    assert historical["minmus_learned"]["target"] == "Minmus"
    assert historical["minmus_learned"]["end_to_end_from_launch"] is True


def test_actual_saved_pad_reload_uses_root_altitude_and_restores_validated_minmus():
    result = read_json(MANIFEST["pad_reload_check"])
    assert result["source"] == "actual_ksp_krpc" and result["success"] is True
    check = result["pad_check"]
    assert check["snapshot_state_consistent"] is True
    actual, expected = check["loaded_state"], result["snapshot"]["state"]
    assert actual["altitude_reference"] == "root_part_origin"
    assert abs(actual["altitude"] - expected["altitude"]) < 0.5
    assert abs(actual["mean_altitude"] - expected["altitude"]) > 5
    for key in ("stage", "crew", "part_names"):
        assert actual[key] == expected[key]
    assert result["cleanup_errors"] == []
    assert result["restored_nodes"] == 0
    assert result["restored_rails_warp_factor"] == 0
    assert result["restored_physics_warp_factor"] == 0
    restored = result["restored_orbit"]
    assert restored["body"] == "Minmus" and restored["throttle"] == 0
    assert 45000 <= restored["periapsis"] <= restored["apoapsis"] <= 55000


@pytest.mark.parametrize("relative,expected", MANIFEST["artifacts"].items())
def test_mission_artifact_size_and_hash(relative, expected):
    payload = artifact_path(relative).read_bytes()
    assert len(payload) == expected["bytes"]
    assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    if relative.endswith(".gz"):
        assert (
            hashlib.sha256(gzip.decompress(payload)).hexdigest()
            == expected["uncompressed_sha256"]
        )


@pytest.mark.parametrize(
    "model", MANIFEST["models"].values(), ids=MANIFEST["models"].keys()
)
def test_model_and_sidecar_are_the_curated_immutable_artifacts(model):
    for path_key, hash_key in (
        ("checkpoint", "sha256"),
        ("metadata", "metadata_sha256"),
    ):
        payload = artifact_path(model[path_key]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == model[hash_key]
        assert model[hash_key] == MANIFEST["artifacts"][model[path_key]]["sha256"]
    assert (
        Path(model["metadata"]).name
        == Path(model["checkpoint"]).with_suffix(".metadata.json").name
    )


@pytest.mark.parametrize("flight", MANIFEST["flights"], ids=lambda row: row["id"])
def test_learned_mission_invocation_handoffs_and_full_period_proof(flight):
    summary, rows = read_json(flight["summary"]), recording(flight["recording"])
    assert summary["source"] == "actual_ksp_krpc" and summary["mode"] == "fly"
    assert summary["success"] is True and summary["mission"]["success"] is True
    assert summary["target"] == flight["target"]
    assert summary["capture_controller"] == "learned_capture_policy"
    assert (
        summary["end_to_end_from_launch_verified"] is flight["end_to_end_from_launch"]
    )
    model = MANIFEST["models"]["capture"]
    metadata = read_json(model["metadata"])
    assert summary["capture_model_sha256"] == model["sha256"]
    assert summary["capture_metadata_sha256"] == model["metadata_sha256"]
    assert summary["capture_model_num_timesteps"] == metadata["num_timesteps"]
    assert summary["capture_model_task"] == metadata["task"] == "orbital_capture_v2"
    assert (
        summary["system_catalog_sha256"]
        == MANIFEST["artifacts"]["orbiter/data/stock_system.json"]["sha256"]
    )
    invocation = one_event(rows, "mission_invocation")
    for key in (
        "source",
        "mode",
        "target",
        "target_altitude",
        "capture_controller",
        "capture_model_sha256",
        "capture_metadata_sha256",
        "capture_model_task",
        "capture_model_num_timesteps",
        "system_catalog_sha256",
    ):
        assert invocation[key] == summary[key]
    mission = summary["mission"]
    assert event_payload(one_event(rows, "mission_end")) == mission
    assert_selected_window(rows, mission)
    assert mission["plan"]["target"] == flight["target"]
    assert (
        mission["plan"]["actual_ksp_verified"] is False
    )  # Candidate alone is not proof.
    prediction = mission["patched_conic_prediction"]
    assert prediction["encounter"] is True and prediction["target"] == flight["target"]
    assert prediction["verified_by"] == "actual_ksp_patched_conic_node"
    assert event_payload(one_event(rows, "encounter_node_verified")) == prediction
    assert mission["departure"]["success"] is True
    handoffs = mission["handoffs"]
    assert [row["stage"] for row in handoffs] == [
        "actual_target_soi",
        "actual_target_orbit",
    ]
    assert all(row["body"] == flight["target"] for row in handoffs)
    assert handoffs[0]["ut"] < handoffs[1]["ut"]
    capture = mission["capture"]
    assert (
        capture["success"] is True and capture["controller"] == "learned_capture_policy"
    )
    assert event_payload(one_event(rows, "capture_end")) == capture
    start = one_event(rows, "capture_start")
    assert start["controller"] == "learned_capture_policy"
    assert start["config"]["target_altitude"] == summary["target_altitude"]
    assert_orbit(capture["final_orbit"], start["config"], flight["target"])
    assert_full_period(
        mission["independent_coast"], rows, start["config"], flight["target"]
    )
    actions = [row for row in rows if row["event"] == "capture_action"]
    assert actions and all(
        row["controller"] == "learned_capture_policy" for row in actions
    )
    assert not any("scripted_capture" in str(row.get("controller", "")) for row in rows)
    no_cleanup_errors(summary)
    for row in rows:
        no_cleanup_errors(row)
        if row["event"].endswith("controls_released"):
            assert row["errors"] == []


@pytest.mark.parametrize("flight", MANIFEST["flights"], ids=lambda row: row["id"])
def test_launch_attribution_is_bound_to_actual_same_invocation_or_separate_record(
    flight,
):
    summary, rows = read_json(flight["summary"]), recording(flight["recording"])
    if flight["end_to_end_from_launch"] is True:
        assert summary["end_to_end_from_launch_verified"] is True
        launch, mission = summary["launch"], summary["mission"]
        assert launch["success"] is True and launch["ascent"]["success"] is True
        assert (
            launch["controlling_part"]
            == launch["parking_controlling_part"]
            == mission["final_controlling_part"]
        )
        assert summary["launch_model_sha256"] == MANIFEST["models"]["ascent"]["sha256"]
        ascent_metadata = read_json(MANIFEST["models"]["ascent"]["metadata"])
        assert summary["launch_num_timesteps"] == ascent_metadata["num_timesteps"]
        invocation = one_event(rows, "mission_invocation")
        assert (
            invocation["launch_model_sha256"] == summary["launch_model_sha256"]
        )
        assert invocation["launch_num_timesteps"] == summary["launch_num_timesteps"]
        start = one_event(rows, "ascent_flight_start")
        assert start["controller"] == ascent_metadata["algorithm"].upper()
        assert start["config"] == summary["launch_config"]
        assert start["stage_plan"] == summary["stage_plan"] == launch["stage_plan"]
        assert start["fuel_stages"] == summary["fuel_stages"] == launch["fuel_stages"]
        assert one_event(rows, "launch_handoff")["controlling_part"] == launch[
            "controlling_part"
        ]
        assert one_event(rows, "ascent_flight_end")["success"] is True
        assert event_payload(one_event(rows, "launch_ascent_end")) == launch
        assert mission["independent_coast"]["success"] is True
        assert "separate_ascent_record_sha256" not in summary
    else:
        assert flight["id"] == "mun_learned"
        assert flight["target"] == "Mun"
        assert flight["end_to_end_from_launch"] is False
        assert summary["end_to_end_from_launch_verified"] is False
        ascent_path = MANIFEST["ascent_recording"]
        raw_hash = MANIFEST["artifacts"][ascent_path]["uncompressed_sha256"]
        assert summary["separate_ascent_record_sha256"] == raw_hash
        assert (
            one_event(rows, "mission_invocation")["separate_ascent_record_sha256"]
            == raw_hash
        )
        ascent = recording(ascent_path)
        assert (
            one_event(ascent, "controller_checkpoint")["sha256"]
            == MANIFEST["models"]["ascent"]["sha256"]
        )
        assert one_event(ascent, "flight_end")["success"] is True
        assert one_event(ascent, "controls_released")["errors"] == []
        assert "launch" not in summary


PANEL_FLIGHTS = {
    "mun_panel_launch": ("Mun", 50000.0),
    "minmus_panel_launch": ("Minmus", 60000.0),
}
NATIVE_FLIGHTS = {"mun_native_launch": ("Mun", 50000.0)}
CONTROL_FLIGHTS = {**PANEL_FLIGHTS, **NATIVE_FLIGHTS}


def test_native_run_is_a_distinct_actual_launch():
    flights = {flight["id"]: flight for flight in MANIFEST["flights"]}
    native = flights["mun_native_launch"]
    assert native["frontend"] == "native_ksp_toolbar"
    native_job = read_json(native["panel_request"])["job_id"]
    assert all(
        native_job != read_json(flights[identifier]["panel_request"])["job_id"]
        for identifier in PANEL_FLIGHTS
    )
    frontend = read_json(MANIFEST["native_frontend_validation"])
    assert frontend["job_id"] == native_job
    assert frontend["frontend"] == native["frontend"]
    assert frontend["cold_reload_source_sha256"] == hashlib.sha256(
        (ROOT / "orbiter/ksp_autopilot.cs").read_bytes()
    ).hexdigest()
    assert frontend["launch_dll_sha256"] != frontend["cold_reload_dll_sha256"]
    assert any(
        "[OrbiterAutopilot] Launch-to-orbit mission verified" in line
        for line in frontend["native_log"]
    )
    observed = frontend["cold_reload_observation"]
    assert observed["heading"] == "Orbit verified"
    assert observed["last_status_age_s"] == 0
    assert not any(observed[key] for key in ("run_enabled", "stop_enabled", "error_visible"))


def test_both_panel_destinations_have_distinct_full_launch_evidence():
    flights = {flight["id"]: flight for flight in MANIFEST["flights"]}
    assert PANEL_FLIGHTS.keys() <= flights.keys(), (
        "Both actual Run-button launch-to-destination flights must remain archived"
    )
    paths, jobs = [], []
    for identifier, (target, altitude) in PANEL_FLIGHTS.items():
        flight = flights[identifier]
        assert flight["target"] == target
        assert flight["end_to_end_from_launch"] is True
        paths.extend(
            flight[key]
            for key in ("summary", "recording", "panel_request", "panel_status")
        )
        request = read_json(flight["panel_request"])
        assert request["validated_payload"] == dict(
            target=target, target_altitude_m=altitude
        )
        jobs.append(request["job_id"])
    assert len(paths) == len(set(paths)) == 8
    assert len(jobs) == len(set(jobs)) == 2
    failure_jobs = {
        read_json(row["panel_request"])["job_id"] for row in MANIFEST["failed_attempts"]
    }
    assert not set(jobs).intersection(failure_jobs)


def worker_option(argv, flag):
    assert argv.count(flag) == 1, f"Expected one worker option {flag}"
    index = argv.index(flag)
    assert index + 1 < len(argv)
    return argv[index + 1]


@pytest.mark.parametrize(
    "flight",
    [
        flight
        for flight in MANIFEST["flights"]
        if flight["id"] in CONTROL_FLIGHTS
        or "panel_request" in flight
        or "panel_status" in flight
    ],
    ids=lambda row: row["id"],
)
def test_control_request_status_and_active_pad_launch_are_one_actual_invocation(flight):
    target, altitude = CONTROL_FLIGHTS[flight["id"]]
    summary, rows = read_json(flight["summary"]), recording(flight["recording"])
    request = read_json(flight["panel_request"])
    status = read_json(flight["panel_status"])
    assert flight["target"] == summary["target"] == target
    assert flight["end_to_end_from_launch"] is True
    assert summary["end_to_end_from_launch_verified"] is True
    assert request["validated_payload"] == dict(
        target=target, target_altitude_m=altitude
    )
    assert summary["target_altitude"] == altitude
    assert request["mode"] == "active_pad" and summary["launch_active"] is True
    assert summary["launch_craft"] is None and summary["launch_pad_save"] is None
    assert "loaded_pad_save" not in summary["launch"]
    assert "separate_ascent_record_sha256" not in summary
    invocation = one_event(rows, "mission_invocation")
    assert invocation["launch_active"] is True
    assert invocation["target_altitude"] == altitude

    preflight = request["preflight"]
    handoff = summary["launch"]["active_pad_preflight"]
    profile_path = "orbiter/data/transfer_vehicle.json"
    profile = read_json(profile_path)
    for check in (preflight, handoff):
        assert check["ready"] is True and check["failures"] == []
        assert (
            check["profile_sha256"] == MANIFEST["artifacts"][profile_path]["sha256"]
        )
        assert check["supported_profile"] == profile["supported_profile"]
        assert check["part_names"] == sorted(profile["parts"])
        assert check["body"] == "Kerbin" and check["current_stage"] == 4
        assert check["control_state"] == "full"
        assert check["control_source"] == "kerbal" and check["crew_names"]
        assert check["mass"] == check["expected_mass"] == profile["wet_mass"]
    for key in ("vessel", "part_names", "crew_names", "fuel_stage_masses"):
        assert preflight[key] == handoff[key]
    # Remote object numbers are scoped to a connection. The three launch/parking/
    # final identities are checked above within the worker's single connection.
    for key in ("name", "title"):
        assert preflight["controlling_part"][key] == handoff["controlling_part"][key]
    assert handoff["controlling_part"] == summary["launch"]["controlling_part"]
    assert preflight["ut"] <= handoff["ut"] <= summary["launch"]["pad_ut"]

    argv = request["worker_argv"]
    assert argv[1:3] == ["-m", "orbiter.mission_live"]
    assert argv.count("--fly") == argv.count("--launch-active") == 1
    assert not set(argv).intersection(
        {"--launch-craft", "--launch-pad-save", "--capture-scripted", "--ascent-record"}
    )
    assert worker_option(argv, "--target") == target
    assert float(worker_option(argv, "--target-altitude")) == altitude
    assert worker_option(argv, "--stage-plan") == "3,2"
    assert worker_option(argv, "--fuel-stages") == "2,1"
    assert worker_option(argv, "--address") == "127.0.0.1"
    for flag, summary_key, model_key in (
        ("--launch-model", "launch_model", "ascent"),
        ("--capture-model", "capture_model", "capture"),
    ):
        original_path = worker_option(argv, flag)
        assert original_path == summary[summary_key]
        assert Path(original_path).name == Path(
            MANIFEST["models"][model_key]["checkpoint"]
        ).name

    assert status["state"] == status["phase"] == "succeeded"
    assert status["active"] is False and status["error"] is None
    final = status["mission"]
    assert final["id"] == request["job_id"]
    assert final["target"] == target and final["target_altitude_m"] == altitude
    assert final["started_at"] == request["accepted_at"] and final["finished_at"]
    assert final["returncode"] == 0 and final["result"] == summary
    assert final["output_dir"] == worker_option(argv, "--output-dir")
    assert Path(final["output_dir"]).name == request["job_id"]
    assert final["control_cleanup"] == dict(
        verified=True, source="mission_end", errors=[]
    )
    assert status["last_event"] == one_event(rows, "mission_end")


def test_failed_panel_attempt_preserves_successful_ascent_and_unflown_departure():
    failure = next(
        row
        for row in MANIFEST["failed_attempts"]
        if row["id"] == "minmus_panel_pre_fix"
    )
    assert failure["id"] not in {row["id"] for row in MANIFEST["flights"]}
    summary, rows = read_json(failure["summary"]), recording(failure["recording"])
    request, status = read_json(failure["panel_request"]), read_json(
        failure["panel_status"]
    )
    assert summary["source"] == "actual_ksp_krpc" and summary["mode"] == "fly"
    assert summary["success"] is False
    assert summary["end_to_end_from_launch_verified"] is False
    assert summary["target"] == failure["target"] == "Minmus"
    assert summary["target_altitude"] == failure["target_altitude"] == 60000.0
    assert summary["launch_active"] is True
    assert "requested inertial burn; frame conversion rejected" in summary["error"]
    assert summary["launch"]["success"] is True
    assert summary["launch"]["ascent"]["success"] is True
    assert one_event(rows, "ascent_flight_end")["success"] is True
    assert event_payload(one_event(rows, "launch_ascent_end")) == summary["launch"]
    assert (
        summary["launch"]["controlling_part"]
        == summary["launch"]["parking_controlling_part"]
    )
    invocation = one_event(rows, "mission_invocation")
    for key, model_key in (
        ("launch_model_sha256", "ascent"),
        ("capture_model_sha256", "capture"),
    ):
        assert summary[key] == invocation[key] == MANIFEST["models"][model_key]["sha256"]
    assert request["mode"] == "active_pad"
    assert request["validated_payload"] == dict(
        target="Minmus", target_altitude_m=60000.0
    )
    assert request["preflight"]["ready"] is True
    assert request["preflight"]["failures"] == []
    argv = request["worker_argv"]
    assert argv.count("--launch-active") == 1
    assert worker_option(argv, "--target") == "Minmus"
    assert float(worker_option(argv, "--target-altitude")) == 60000.0

    # The exception prevented assignment of the component result to the summary;
    # its finally block still records the failed, cleaned-up transfer component.
    assert "mission" not in summary
    ended = one_event(rows, "mission_end")
    assert ended["success"] is False and ended["handoffs"] == []
    assert ended["plan"] == one_event(rows, "mission_plan")["plan"]
    assert ended["budget"] == one_event(rows, "mission_plan")["budget"]
    assert all(
        name not in ended for name in ("departure", "capture", "independent_coast")
    )
    assert not any(
        row["event"].startswith(("maneuver_", "capture_", "independent_orbital_"))
        or row["event"] == "encounter_node_verified"
        for row in rows
    )
    no_cleanup_errors(summary)
    for row in rows:
        no_cleanup_errors(row)
        if row["event"].endswith("controls_released"):
            assert row["errors"] == []
    assert status["state"] == status["phase"] == "failed"
    assert status["active"] is False and status["error"] == summary["error"]
    final = status["mission"]
    assert final["id"] == request["job_id"]
    assert final["result"] == summary and final["returncode"] == 1
    assert final["started_at"] == request["accepted_at"] and final["finished_at"]
    assert final["output_dir"] == worker_option(argv, "--output-dir")
    assert Path(final["output_dir"]).name == request["job_id"]
    assert final["control_cleanup"] == dict(
        verified=True, source="mission_end", errors=[]
    )
    assert status["last_event"] == ended


def test_node_frame_diagnostic_records_time_skew_without_an_axis_encoding_error():
    diagnostic = read_json(MANIFEST["node_frame_diagnostic"])
    assert diagnostic["source"] == "actual_ksp_krpc"
    failure = next(
        row
        for row in MANIFEST["failed_attempts"]
        if row["id"] == "minmus_panel_pre_fix"
    )
    failed_rows = recording(failure["recording"])
    assert diagnostic["plan"] == one_event(failed_rows, "mission_plan")["plan"]
    samples = diagnostic["samples"]
    setters = [row for row in samples if "stage" in row]
    assert [row["stage"] for row in setters] == ["prograde", "normal", "radial"]
    assert setters[-1]["ut"] > setters[0]["ut"]
    local = diagnostic["local"]
    components = setters[-1]["components"]  # Node API order: prograde, normal, radial.
    assert [-components[2], components[0], components[1]] == pytest.approx(
        local, abs=1e-12
    )
    reads = [row for row in samples if "delay" in row]
    assert [row["delay"] for row in reads] == [0, 0.04, 0.1, 0.3]
    assert all(right["ut"] > left["ut"] for left, right in zip(reads, reads[1:]))
    requested = diagnostic["plan"]["delta_v"]
    errors = []
    magnitude = math.dist(requested, [0.0, 0.0, 0.0])
    for row in reads:
        difference = [actual - desired for actual, desired in zip(row["burn"], requested)]
        assert row["difference"] == pytest.approx(difference, abs=1e-12)
        assert math.dist(row["burn"], [0.0, 0.0, 0.0]) == pytest.approx(
            magnitude, abs=1e-8
        )
        error = math.dist(row["burn"], requested)
        errors.append(error)
        assert math.dist(row["local_now"], local) == pytest.approx(error, abs=1e-8)
    assert all(right > left for left, right in zip(errors, errors[1:]))
    assert 0.002 < errors[0] < 0.003  # m/s, not a degree-scale axis/sign error.
    assert not all(
        abs(actual - desired) <= 1e-3 + 1e-6 * abs(desired)
        for actual, desired in zip(reads[0]["burn"], requested)
    )  # This correct local encoding failed the original time-skewed guard.


def test_scripted_arrival_baseline_is_never_attributed_to_learned_full_mission():
    baseline = MANIFEST["baseline"]
    summary, rows = read_json(baseline["summary"]), recording(baseline["recording"])
    assert (
        summary["success"] is True
        and summary["end_to_end_from_launch_verified"] is False
    )
    assert summary["controller"] == "scripted"
    assert summary["capture"]["controller"] == "scripted_capture_teacher"
    assert summary["initial_state"]["source"] == summary["target"] == "Mun"
    assert "mission" not in summary and "capture_model_sha256" not in summary
    assert one_event(rows, "capture_start")["controller"] == "scripted_capture_teacher"
    assert one_event(rows, "capture_end")["controller"] == "scripted_capture_teacher"
    actions = [row for row in rows if row["event"] == "capture_action"]
    assert actions and all(
        row["controller"] == "scripted_capture_teacher" for row in actions
    )
    assert not any(row["event"] == "mission_invocation" for row in rows)
    config = one_event(rows, "capture_start")["config"]
    assert_full_period(summary["independent_coast"], rows, config, "Mun")
    no_cleanup_errors(summary)


def test_capture_training_provenance_preserves_zero_new_phase_steps_and_ppo_ancestry():
    training = read_json(MANIFEST["training"])
    capture, ascent = training["capture"], training["ascent"]
    model = MANIFEST["models"]["capture"]
    metadata = read_json(model["metadata"])
    assert (
        capture["checkpoint"] == model["checkpoint"]
        and capture["sha256"] == model["sha256"]
    )
    assert capture["environment"] == metadata["environment"]
    assert capture["selected_new_phase_ppo_steps"] == metadata["num_timesteps"] == 0
    assert (
        capture["completed_new_phase_ppo_experiment_steps"]
        == capture["arguments"]["steps"]
        > 0
    )
    initialization = capture["initialization"]
    assert (
        initialization["source_num_timesteps"] > 0
        and initialization["optimizer_and_step_count_reset"] is True
    )
    assert (
        len(initialization["sha256"]) == 64
        and initialization["sha256"] != model["sha256"]
    )
    assert capture["arguments"]["initialize"] == initialization["checkpoint"]
    assert (
        capture["arguments"]["dagger_rounds"] > 0
        and capture["arguments"]["bc_episodes"] > 0
    )
    assert (
        capture["independent_nominal"]["sha256"]
        == capture["arrival_stress"]["sha256"]
        == model["sha256"]
    )
    assert capture["independent_nominal"]["num_timesteps"] == 0
    assert ascent["checkpoint"] == MANIFEST["models"]["ascent"]["checkpoint"]
    assert ascent["sha256"] == MANIFEST["models"]["ascent"]["sha256"]
    assert (
        ascent["num_timesteps"]
        == read_json(MANIFEST["models"]["ascent"]["metadata"])["num_timesteps"]
    )


def test_vacuum_diagnostic_is_tied_to_scripted_recording_and_marks_fit_scope():
    diagnostic = read_json(MANIFEST["vacuum_diagnostic"])
    source = diagnostic["source"]
    assert source == MANIFEST["baseline"]["recording"]
    assert diagnostic["source_sha256"] == MANIFEST["artifacts"][source]["sha256"]
    assert diagnostic["summary"]["offset_fitted_on_same_flight"] is True
    assert diagnostic["summary"]["independent_latency_validation"] is False


def experimental_stock_records():
    summary = read_json(MANIFEST["training"])["experimental_stock_capture"]
    raw = {}
    for name, path in summary["artifacts"].items():
        raw[name] = json.loads(gzip.decompress(artifact_path(path).read_bytes()))
    return summary, raw["nominal_full"], raw["stress_full"]


def assert_capture_evaluation_summary(compact, raw, bodies, seeds):
    assert raw["task"] == "orbital_capture_v2" and raw["kind"] == "policy"
    assert set(raw["per_body"]) == set(compact["per_body"]) == set(bodies)
    successes = 0
    episodes = 0
    for body in bodies:
        row, summary = raw["per_body"][body], compact["per_body"][body]
        results = row["results"]
        assert [item["seed"] for item in results] == seeds
        assert len({item["seed"] for item in results}) == len(seeds)
        assert all(item["final_info"]["body"] == body for item in results)
        observed_successes = sum(item["success"] for item in results)
        failures = dict(
            Counter(
                item["termination_reason"] for item in results if not item["success"]
            )
        )
        assert row["successes"] == observed_successes
        assert row["episodes"] == len(results)
        assert row["failure_counts"] == failures
        assert row["fuel_exhaustions"] == sum(
            item["termination_reason"] == "out_of_fuel" for item in results
        )
        assert row["reference_budget_shortfalls"] == sum(
            bool(item["reference_capture_budget"]["reference_budget_shortfall"])
            for item in results
        )
        assert summary == {key: row[key] for key in summary}
        successes += observed_successes
        episodes += len(results)
    assert compact["successes"] == raw["successes"] == successes
    assert compact["episodes"] == raw["episodes"] == episodes
    assert compact["success_ci95"] == raw["success_ci95"]


def test_experimental_stock_model_is_trained_on_all_bodies_without_ppo_or_zero_shot_claim():
    summary, nominal, _ = experimental_stock_records()
    model = MANIFEST["models"]["experimental_stock_capture"]
    metadata = read_json(model["metadata"])
    provenance = nominal["provenance"]
    stock_bodies = {
        name
        for name, body in metadata["body_catalog"].items()
        if body["parent"] is not None
    }
    assert len(stock_bodies) == 16
    assert (
        set(summary["trained_bodies"])
        == set(metadata["train_bodies"])
        == set(provenance["train_bodies"])
        == stock_bodies
    )
    assert (
        summary["heldout_bodies"]
        == metadata["heldout_bodies"]
        == provenance["heldout_bodies"]
        == []
    )
    assert (
        summary["zero_shot_body_claim"]
        is provenance["zero_shot_body_evaluation"]
        is False
    )
    assert summary["checkpoint"] == model["checkpoint"]
    assert (
        summary["checkpoint_sha256"]
        == provenance["checkpoint_sha256"]
        == model["sha256"]
    )
    assert (
        summary["metadata_sha256"]
        == provenance["metadata_sha256"]
        == model["metadata_sha256"]
    )
    assert (
        summary["selected_num_timesteps"]
        == metadata["num_timesteps"]
        == provenance["checkpoint_num_timesteps"]
        == 0
    )
    assert (
        summary["prior_ppo_ancestry_steps"]
        == summary["source_initialization"]["source_num_timesteps"]
        == 0
    )
    assert all(row["num_timesteps"] == 0 for row in summary["prior_ancestry"])
    assert summary["prior_ancestry"] == provenance["prior_ancestry"]
    assert summary["source_initialization"] == provenance["source_initialization"]
    assert (
        summary["bounded_ppo_steps_completed"] == summary["arguments"]["steps"] == 65536
    )
    assert (
        summary["selected_dagger_round_before_ppo"]
        == provenance["selected_dagger_round_before_ppo"]
        == 1
    )
    assert (
        summary["completed_dagger_rounds"]
        == provenance["completed_dagger_rounds"]
        == 12
    )
    assert "BC" in summary["training_method"] and "DAgger" in summary["training_method"]
    assert "no PPO ancestry" in summary["training_method"]
    assert set(summary["source_initialization"]["source_train_bodies"]).issubset(
        stock_bodies
    )
    assert set(summary["source_initialization"]["source_heldout_bodies"]).issubset(
        stock_bodies
    )


def test_experimental_stock_nominal_claim_matches_all_raw_episodes_and_fresh_seeds():
    summary, nominal, _ = experimental_stock_records()
    seeds = summary["evaluation_arguments"]["nominal_seeds"]
    assert len(seeds) == len(set(seeds)) == 32
    assert not set(seeds).intersection(summary["fixed_validation_seeds"])
    assert seeds == nominal["provenance"]["independent_nominal_seeds"]
    assert_capture_evaluation_summary(
        summary["nominal"], nominal["evaluation"], summary["trained_bodies"], seeds
    )
    assert (
        summary["nominal"]["successes"] == 511 and summary["nominal"]["episodes"] == 512
    )
    assert summary["nominal"]["per_body"]["Jool"]["failure_counts"] == {"time_limit": 1}
    assert summary["nominal"]["per_body"]["Jool"]["successes"] == 31


def test_experimental_stock_stress_reuses_seeds_and_preserves_gilly_jool_failures():
    summary, nominal, stress = experimental_stock_records()
    assert stress["checkpoint_sha256"] == summary["checkpoint_sha256"]
    cases = stress["evaluation"]["cases"]
    assert set(cases) == set(summary["stress"])
    assert len(cases) == 6
    seeds = summary["evaluation_arguments"]["stress_seeds"]
    assert len(seeds) == len(set(seeds)) == 8
    assert seeds == nominal["provenance"]["independent_stress_seeds"]
    assert not set(seeds).intersection(summary["fixed_validation_seeds"])
    assert not set(seeds).intersection(summary["evaluation_arguments"]["nominal_seeds"])
    assert "same eight seeds" in summary["stress_sampling_note"]
    assert "correlated" in summary["stress_sampling_note"]
    for name, case in cases.items():
        assert_capture_evaluation_summary(
            summary["stress"][name], case, summary["trained_bodies"], seeds
        )
        assert case["episodes"] == 16 * 8
    assert cases["target_clearance_1.4x"]["per_body"]["Gilly"]["failure_counts"] == {
        "time_limit": 8
    }
    assert cases["arrival_vinf_0.2_circular_speed"]["per_body"]["Jool"][
        "failure_counts"
    ] == {"time_limit": 2}
    assert cases["arrival_vinf_0.7_circular_speed"]["per_body"]["Jool"][
        "failure_counts"
    ] == {"out_of_fuel": 1}
    assert cases["craft_mass_4125_thrust_0.8_isp_0.9"]["per_body"]["Jool"][
        "failure_counts"
    ] == {"time_limit": 3, "out_of_fuel": 1}


def test_experimental_stock_checkpoint_is_separate_from_actual_ksp_evidence():
    summary, _, _ = experimental_stock_records()
    experimental_hash = summary["checkpoint_sha256"]
    assert experimental_hash != MANIFEST["models"]["capture"]["sha256"]
    assert "no transfer, launch or KSP execution claim" in summary["scope"]
    for flight in MANIFEST["flights"]:
        actual = read_json(flight["summary"])
        invocation = one_event(recording(flight["recording"]), "mission_invocation")
        assert (
            actual["capture_model_sha256"]
            == invocation["capture_model_sha256"]
            != experimental_hash
        )
    baseline = read_json(MANIFEST["baseline"]["summary"])
    assert "capture_model_sha256" not in baseline
