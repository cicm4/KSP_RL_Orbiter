"""Panel security and single-controller lifecycle tests; no actual KSP calls."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import http.client
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from orbiter import control


def test_inspector_busy_lease_rejects_before_models_and_rpc(configured, monkeypatch):
    import krpc
    from orbiter import live
    from orbiter.ownership import FlightOwnership

    config = configured
    monkeypatch.setattr(
        live,
        "load_controller",
        lambda *args: pytest.fail("Busy inspector loaded model"),
    )
    monkeypatch.setattr(
        krpc, "connect", lambda **kwargs: pytest.fail("Busy inspector contacted RPC")
    )
    with FlightOwnership(rpc_port=config.rpc_port):
        result = control.inspect_active(config)
    assert not result["ready"] and "FlightOwnershipBusy" in result["error"]


@pytest.mark.parametrize("outcome", ["success", "close_error", "interrupt"])
def test_inspector_holds_lease_through_model_rpc_and_close(
    configured, monkeypatch, outcome
):
    import krpc
    from orbiter import live, mission_live
    from orbiter.ownership import flight_busy

    config = configured
    observed = []

    def load(*args):
        observed.append(("model", flight_busy(rpc_port=config.rpc_port)))
        return object(), object()

    def inspect(*args, **kwargs):
        observed.append(("inspect", flight_busy(rpc_port=config.rpc_port)))
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return dict(ready=True, failures=[])

    def close():
        observed.append(("close", flight_busy(rpc_port=config.rpc_port)))
        if outcome == "close_error":
            raise RuntimeError("test inspector close")

    monkeypatch.setattr(live, "load_controller", load)
    monkeypatch.setattr(mission_live, "inspect_active_pad", inspect)
    monkeypatch.setattr(krpc, "connect", lambda **kwargs: SimpleNamespace(close=close))
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            control.inspect_active(config)
    else:
        result = control.inspect_active(config)
        assert result["ready"] == (outcome == "success")
    assert observed == [("model", True), ("inspect", True), ("close", True)]
    assert not flight_busy(rpc_port=config.rpc_port)


class Process:
    pid = 12345

    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake flight", timeout)
        return self.returncode


@pytest.fixture
def configured(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    launch, capture = models / "ascent.zip", models / "capture.zip"
    for path in (launch, capture):
        path.write_bytes(b"fake model")
        path.with_suffix(".metadata.json").write_text("{}")
    # Reserve a test-only endpoint for the fixture's lifetime. Generic fake
    # panels must never probe or acquire the real KSP lease on port 50000.
    with ExitStack() as sockets:
        while True:
            endpoint = sockets.enter_context(socket.socket())
            endpoint.bind(("127.0.0.1", 0))
            port = endpoint.getsockname()[1]
            if port != 50000:
                break
        yield control.ControlConfig(
            tmp_path, launch, capture, tmp_path / "runs", rpc_port=port
        )


@pytest.fixture
def panel(configured):
    processes, commands, signals, inspections = [], [], [], []

    def inspect(config):
        inspections.append(config)
        return dict(ready=True, vessel="OrbiterRL Transfer", failures=[], body="Kerbin")

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        process = Process()
        processes.append(process)
        return process

    def send(process, signum):
        signals.append(signum)

    value = control.MissionControl(
        configured, inspector=inspect, popen=popen, signal_worker=send, monitor=False
    )
    value.testing = SimpleNamespace(
        processes=processes, commands=commands, signals=signals, inspections=inspections
    )
    yield value
    for process in processes:
        process.returncode = -int(signal.SIGINT)
    value.close(cleanup_timeout=0)


def payload(target="Mun", altitude=50000):
    return dict(target=target, target_altitude_m=altitude)


@pytest.mark.parametrize(
    "data",
    [
        payload("Duna"),
        payload(altitude=39999),
        payload(altitude=60001),
        payload(altitude=True),
        payload(altitude="50000"),
        payload(altitude=float("nan")),
        payload(altitude=float("inf")),
        payload(altitude=10**1000),
        {**payload(), "launch_model": "/tmp/custom.zip"},
        {},
        [],
    ],
)
def test_only_supported_destinations_and_trained_goal_range_can_launch(panel, data):
    with pytest.raises(control.ControlError):
        panel.run(data)
    assert not panel.testing.commands
    assert not panel.testing.inspections


def test_worker_uses_explicit_models_active_ship_and_fixed_stages(panel, configured):
    result = panel.run(payload("Minmus", 60000))
    assert result["active"] and not result["can_run"]
    command, options = panel.testing.commands[0]
    assert command[:5] == [
        sys.executable,
        "-m",
        "orbiter.mission_live",
        "--fly",
        "--launch-active",
    ]
    for flag, value in [
        ("--launch-model", str(configured.launch_model)),
        ("--capture-model", str(configured.capture_model)),
        ("--stage-plan", "3,2"),
        ("--fuel-stages", "2,1"),
        ("--target", "Minmus"),
        ("--target-altitude", "60000.0"),
        ("--wall-timeout", "2400"),
    ]:
        assert command[command.index(flag) + 1] == value
    assert "--launch-pad-save" not in command and "--launch-craft" not in command
    assert options["start_new_session"] is True
    assert options["shell"] is False
    assert options["stdin"] == subprocess.DEVNULL
    assert options["cwd"] == configured.project_root
    assert Path(result["mission"]["output_dir"]).parent == configured.run_root
    accepted = json.loads(
        (Path(result["mission"]["output_dir"]) / "panel_request.json").read_text()
    )
    assert accepted["worker_argv"] == command
    assert accepted["validated_payload"] == payload("Minmus", 60000)
    assert accepted["preflight"]["ready"] and accepted["mode"] == "active_pad"
    assert accepted["job_id"] == result["mission"]["id"]


def test_two_concurrent_runs_create_only_one_flight_worker(panel):
    def attempt():
        try:
            panel.run(payload())
            return 202
        except control.ControlError as error:
            return error.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == [202, 409]
    assert len(panel.testing.commands) == len(panel.testing.inspections) == 1


def test_status_caches_readonly_inspection_and_never_inspects_during_flight(panel):
    assert panel.status()["can_run"]
    assert panel.status()["can_run"]
    assert len(panel.testing.inspections) == 1
    panel.run(payload())
    assert len(panel.testing.inspections) == 2  # Run refreshes the active ship.
    panel.preflight_at = -float("inf")
    for _ in range(3):
        assert panel.status()["active"]
    assert len(panel.testing.inspections) == 2


def test_failed_preflight_or_missing_model_prevents_worker(panel, configured):
    panel.inspector = lambda _: dict(
        ready=False, failures=["Load the supported craft on the launchpad"]
    )
    with pytest.raises(control.ControlError, match="supported craft") as error:
        panel.run(payload())
    assert error.value.status == 409
    configured.capture_model.with_suffix(".metadata.json").unlink()
    panel.inspector = lambda _: pytest.fail(
        "Missing models must fail before RPC inspection"
    )
    with pytest.raises(control.ControlError, match="model file is missing"):
        panel.run(payload())
    assert not panel.testing.commands


def test_rpc_inspection_error_is_displayable_and_cannot_launch(panel):
    def unavailable(_):
        raise TimeoutError("KSP is not responding")

    panel.inspector = unavailable
    assert "KSP is not responding" in panel.status()["preflight"]["error"]
    with pytest.raises(control.ControlError, match="KSP is not responding"):
        panel.run(payload())
    assert not panel.testing.commands


def test_stop_interrupts_once_and_waits_for_worker_cleanup(panel):
    started = panel.run(payload())
    assert panel.stop()["state"] == "stopping"
    assert panel.stop()["state"] == "stopping"
    assert panel.testing.signals == [signal.SIGINT]
    assert panel.status()["active"]  # No second controller while cleanup runs.
    (Path(started["mission"]["output_dir"]) / "mission.jsonl").write_text(
        '{"event":"launch_handoff"}\n{"event":"ascent_action"}\n'
        '{"event":"launch_ascent_end","cleanup_errors":[]}\n'
    )
    panel.testing.processes[0].returncode = -2
    result = panel.status()
    assert result["state"] == result["phase"] == "stopped"
    assert not result["active"]
    assert panel.output_handle is None


def test_server_close_stops_and_reaps_worker_before_returning(panel):
    started = panel.run(payload())
    (Path(started["mission"]["output_dir"]) / "mission.jsonl").write_text(
        '{"event":"mission_invocation"}\n'
    )

    def released(process, signum):
        panel.testing.signals.append(signum)
        process.returncode = -2

    panel.signal_worker = released
    panel.close(cleanup_timeout=0)
    assert panel.testing.signals == [signal.SIGINT]
    assert panel.worker is None and panel.closed
    with pytest.raises(control.ControlError, match="shutting down"):
        panel.run(payload())


def test_incomplete_shutdown_preserves_worker_without_forced_termination(panel):
    panel.run(payload())
    with pytest.raises(control.ShutdownIncomplete, match="remains active"):
        panel.close(cleanup_timeout=0)
    assert panel.testing.signals == [signal.SIGINT]
    assert panel.worker is panel.testing.processes[0]
    assert panel.status()["state"] == "stopping"
    assert "Shutdown incomplete" in panel.status()["error"]


@pytest.mark.parametrize(
    "records",
    [
        [],
        [dict(event="launch_handoff"), dict(event="ascent_action")],
        [
            dict(event="launch_handoff"),
            dict(event="mission_end", cleanup_errors=["throttle release failed"]),
        ],
        [
            dict(event="launch_handoff"),
            dict(event="launch_ascent_end", cleanup_errors=[]),
            dict(event="capture_action"),
        ],
    ],
)
def test_stop_exit_never_claims_released_controls_without_ordered_proof(panel, records):
    started = panel.run(payload())
    directory = Path(started["mission"]["output_dir"])
    (directory / "mission.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    panel.stop()
    panel.testing.processes[0].returncode = -2
    result = panel.status()
    assert result["state"] == "failed"
    assert "release was not verified" in result["error"]


def test_record_progress_and_success_come_from_worker_artifacts(panel):
    started = panel.run(payload())
    directory = Path(started["mission"]["output_dir"])
    records = [
        dict(event="ascent_telemetry", altitude=81000, apoapsis=100000),
        dict(event="capture_telemetry", altitude=50001, apoapsis=50003),
        dict(event="independent_orbital_period_probe", periapsis=49999),
    ]
    (directory / "mission.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + '\n{"event":'
    )
    state = panel.status()
    assert state["phase"] == "orbit_verification"
    assert state["telemetry"]["periapsis"] == 49999
    summary = dict(
        success=True,
        end_to_end_from_launch_verified=True,
        launch=dict(success=True),
        mission=dict(independent_coast=dict(success=True)),
    )
    (directory / "mission.json").write_text(json.dumps(summary))
    panel.testing.processes[0].returncode = 0
    state = panel.status()
    assert state["state"] == state["phase"] == "succeeded"
    assert state["mission"]["result"] == summary
    assert state["mission"]["returncode"] == 0


def test_nonzero_exit_cannot_report_success_even_with_success_record(panel):
    started = panel.run(payload())
    (Path(started["mission"]["output_dir"]) / "mission.json").write_text(
        '{"success":true}'
    )
    panel.testing.processes[0].returncode = 1
    state = panel.status()
    assert state["state"] == "failed"
    assert state["error"]


@pytest.mark.parametrize(
    "summary",
    [
        dict(success=True),
        dict(success=True, end_to_end_from_launch_verified=True),
        dict(
            success=True,
            end_to_end_from_launch_verified=True,
            launch=dict(success=True),
            mission=dict(independent_coast=dict(success=False)),
        ),
        dict(
            success=True,
            end_to_end_from_launch_verified=False,
            launch=dict(success=True),
            mission=dict(independent_coast=dict(success=True)),
        ),
    ],
)
def test_incomplete_success_summary_cannot_claim_verified_orbit(panel, summary):
    started = panel.run(payload())
    (Path(started["mission"]["output_dir"]) / "mission.json").write_text(
        json.dumps(summary)
    )
    panel.testing.processes[0].returncode = 0
    result = panel.status()
    assert result["state"] == "failed" and "verification" in result["error"]


@pytest.mark.parametrize("contents", ["null", "[]", "{invalid"])
def test_malformed_worker_summary_is_a_displayable_failure(panel, contents):
    started = panel.run(payload())
    (Path(started["mission"]["output_dir"]) / "mission.json").write_text(contents)
    panel.testing.processes[0].returncode = 0
    result = panel.status()
    assert result["state"] == "failed"
    assert "missing or invalid" in result["error"]


def test_worker_spawn_failure_closes_file_and_reports_error(panel):
    def cannot_spawn(*args, **kwargs):
        raise OSError("Python executable is unavailable")

    panel.popen = cannot_spawn
    with pytest.raises(control.ControlError, match="Python executable"):
        panel.run(payload())
    assert panel.output_handle is None
    assert panel.worker is None
    assert panel.status()["state"] == "failed"


@pytest.fixture
def server(panel):
    value = control.ControlServer(("127.0.0.1", 0), panel)
    thread = threading.Thread(target=value.serve_forever, daemon=True)
    thread.start()
    yield value
    value.shutdown()
    for process in panel.testing.processes:
        process.returncode = -2
    value.server_close()
    thread.join(timeout=2)


def request(server, method, path, body=None, *, headers=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        return response.status, dict(response.getheaders()), data
    finally:
        connection.close()


def authenticated_headers(server):
    return {
        "Origin": f"http://127.0.0.1:{server.server_address[1]}",
        "X-Orbiter-CSRF": server.controller.csrf_token,
        "Content-Type": "application/json",
    }


def test_browser_session_and_status_are_json_without_cross_origin_permission(server):
    code, headers, body = request(server, "GET", "/api/session")
    result = json.loads(body)
    assert code == 200
    assert result["application"] == "orbiter-autopilot"
    assert result["protocol_version"] == 1
    assert result["server_instance"] == server.controller.server_instance
    assert result["project_root"] == str(server.controller.config.project_root)
    assert result["rpc_port"] == server.controller.config.rpc_port
    assert result["destinations"] == ["Mun", "Minmus"]
    assert result["altitude"]["minimum_m"] == 40000
    assert result["csrf_token"] == server.controller.csrf_token
    assert headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers
    code, _, body = request(server, "GET", "/api/status")
    assert code == 200 and json.loads(body)["can_run"]


@pytest.mark.parametrize("changed", ["job", "server"])
def test_native_stale_stop_cannot_interrupt_a_replacement_flight(server, changed):
    started = server.controller.run(payload())
    identity = dict(
        expected_job_id=started["mission"]["id"],
        server_instance=server.controller.server_instance,
    )
    identity["expected_job_id" if changed == "job" else "server_instance"] = "stale"
    code, _, body = request(
        server,
        "POST",
        "/api/stop",
        json.dumps(identity),
        headers=authenticated_headers(server),
    )
    assert code == 409 and "changed" in json.loads(body)["error"]
    assert server.controller.testing.signals == []
    assert server.controller.status()["state"] == "running"


def test_native_matching_job_stop_keeps_control_ownership_until_cleanup(server):
    started = server.controller.run(payload())
    identity = dict(
        expected_job_id=started["mission"]["id"],
        server_instance=server.controller.server_instance,
    )
    code, _, body = request(
        server,
        "POST",
        "/api/stop",
        json.dumps(identity),
        headers=authenticated_headers(server),
    )
    result = json.loads(body)
    assert code == 202 and result["active"] and result["state"] == "stopping"
    assert server.controller.testing.signals == [signal.SIGINT]


def test_orphan_flight_blocks_cached_readiness_and_new_inspection(panel, monkeypatch):
    assert panel.status()["can_run"]
    count = len(panel.testing.inspections)
    monkeypatch.setattr(control, "flight_busy", lambda **kwargs: True)
    result = panel.status()
    assert not result["can_run"] and not result["preflight"]["ready"]
    assert "owns" in result["preflight"]["error"]
    with pytest.raises(control.ControlError, match="owns"):
        panel.run(payload())
    assert len(panel.testing.inspections) == count
    assert not panel.testing.commands


@pytest.mark.parametrize(
    "change",
    [
        "missing_token",
        "wrong_token",
        "remote_origin",
        "missing_origin",
        "remote_host",
        "null_origin",
    ],
)
def test_http_refuses_cross_site_control_requests(server, change):
    headers = authenticated_headers(server)
    if change == "missing_token":
        del headers["X-Orbiter-CSRF"]
    elif change == "wrong_token":
        headers["X-Orbiter-CSRF"] = "incorrect"
    elif change == "remote_origin":
        headers["Origin"] = "https://example.com"
    elif change == "missing_origin":
        del headers["Origin"]
    elif change == "remote_host":
        headers["Host"] = "example.com"
    else:
        headers["Origin"] = "null"
    code, _, body = request(
        server, "POST", "/api/run", json.dumps(payload()), headers=headers
    )
    assert code == 403 and json.loads(body)["error"]
    assert not server.controller.testing.commands


def test_http_run_and_stop_use_the_authenticated_single_controller(server):
    headers = authenticated_headers(server)
    code, _, body = request(
        server, "POST", "/api/run", json.dumps(payload()), headers=headers
    )
    assert code == 202 and json.loads(body)["state"] == "running"
    code, _, body = request(
        server, "POST", "/api/run", json.dumps(payload()), headers=headers
    )
    assert code == 409 and "already" in json.loads(body)["error"]
    code, _, body = request(server, "POST", "/api/stop", "{}", headers=headers)
    assert code == 202 and json.loads(body)["state"] == "stopping"


def test_shutdown_timeout_keeps_http_status_available(server, monkeypatch):
    controller = server.controller
    controller.run(payload())
    original_close = controller.close
    monkeypatch.setattr(
        controller, "close", lambda **kwargs: original_close(cleanup_timeout=0)
    )
    with pytest.raises(control.ShutdownIncomplete):
        server.server_close()
    code, _, body = request(server, "GET", "/api/status")
    result = json.loads(body)
    assert code == 200 and result["active"] and result["state"] == "stopping"
    assert controller.testing.signals == [signal.SIGINT]


@pytest.mark.parametrize(
    "body,content_type,expected",
    [
        ('{"target":"Mun","target_altitude_m":NaN}', "application/json", 400),
        ("not json", "application/json", 400),
        ("{}", "text/plain", 415),
        (" " * 4097, "application/json", 413),
    ],
)
def test_http_rejects_malformed_or_oversized_requests(
    server, body, content_type, expected
):
    headers = authenticated_headers(server)
    headers["Content-Type"] = content_type
    code, _, data = request(server, "POST", "/api/run", body, headers=headers)
    assert code == expected and json.loads(data)["error"]
    assert not server.controller.testing.commands


def test_server_rejects_nonloopback_bind(panel):
    with pytest.raises(ValueError, match="127.0.0.1"):
        control.ControlServer(("0.0.0.0", 0), panel)


def test_preflight_subprocess_has_a_bounded_deadline_and_no_flight_flags(
    configured, monkeypatch
):
    called = {}

    def run(command, **kwargs):
        called.update(command=command, **kwargs)
        return SimpleNamespace(
            stdout='{"ready":false,"failures":["KSP is closed"]}', returncode=1
        )

    monkeypatch.setattr(control.subprocess, "run", run)
    result = control.preflight_subprocess(configured)
    assert not result["ready"]
    assert called["timeout"] == 18
    assert "--inspect-active" in called["command"]
    assert "--fly" not in called["command"]
    assert called["cwd"] == configured.project_root


def test_crashed_preflight_cannot_report_readiness(configured, monkeypatch):
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"ready":true}', returncode=-2),
    )
    result = control.preflight_subprocess(configured)
    assert not result["ready"] and "completing normally" in result["error"]


def test_help_needs_neither_torch_nor_krpc():
    script = """
import sys
from orbiter.control import parse_args
try:
    parse_args(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch','krpc','stable_baselines3','sb3_contrib'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "--no-browser" in result.stdout and "--run-root" in result.stdout


def test_model_and_record_defaults_resolve_under_explicit_project_root(tmp_path):
    args = control.parse_args(["--project-root", str(tmp_path), "--no-browser"])
    assert args.launch_model == tmp_path / "models/transfer_ascent_ppo.zip"
    assert args.capture_model == tmp_path / "models/moon_capture.zip"
    assert args.run_root == tmp_path / "runs/control"
