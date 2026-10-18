"""Local launchpad mission panel; one explicit Run starts one KSP controller.

The HTTP server uses only the standard library. Read-only KSP inspection runs
in a bounded subprocess because the RPC deadline requires a main thread.
Inspection is excluded while the flight worker owns the connection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser

from .ownership import FlightOwnership, flight_busy

DESTINATIONS = ("Mun", "Minmus")
MIN_ALTITUDE_M = 40_000.0
MAX_ALTITUDE_M = 60_000.0
DEFAULT_ALTITUDE_M = 50_000.0
MAX_REQUEST_BYTES = 4096
PREFLIGHT_CACHE_SECONDS = 5.0
PREFLIGHT_WALL_TIMEOUT = 18.0


class ControlError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class ShutdownIncomplete(RuntimeError):
    """A worker still owns the controls; keep the panel and worker alive."""


@dataclass(frozen=True)
class ControlConfig:
    project_root: Path
    launch_model: Path
    capture_model: Path
    run_root: Path
    python: str = sys.executable
    rpc_port: int = 50000
    rpc_timeout: float = 5.0

    def model_errors(self):
        errors = []
        for label, model in (
            ("Ascent", self.launch_model),
            ("Capture", self.capture_model),
        ):
            for path in (model, model.with_suffix(".metadata.json")):
                if not path.is_file():
                    errors.append(f"{label} model file is missing: {path}")
        return errors


def validate_mission(payload):
    if not isinstance(payload, dict) or set(payload) != {"target", "target_altitude_m"}:
        raise ControlError("Run requires only target and target_altitude_m")
    target, altitude = payload["target"], payload["target_altitude_m"]
    if target not in DESTINATIONS:
        raise ControlError("Choose Mun or Minmus")
    if (
        isinstance(altitude, bool)
        or not isinstance(altitude, (int, float))
        or not MIN_ALTITUDE_M <= altitude <= MAX_ALTITUDE_M
        or not math.isfinite(altitude)
    ):
        raise ControlError("Orbit altitude must be between 40000 and 60000 metres")
    return target, float(altitude)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def inspect_active(config):
    """Read-only helper, called only on a subprocess's main thread."""
    errors = config.model_errors()
    if errors:
        return dict(ready=False, failures=errors, error="; ".join(errors))
    connection, result, ownership = None, None, None
    try:
        ownership = FlightOwnership(rpc_port=config.rpc_port)
        ownership.acquire()
        from .live import load_controller, rpc_deadline
        from .mission_live import inspect_active_pad
        import krpc

        model, launch_config = load_controller(config.launch_model)
        del model
        with rpc_deadline(config.rpc_timeout):
            connection = krpc.connect(
                name="Orbiter panel read-only launchpad inspection",
                address="127.0.0.1",
                rpc_port=config.rpc_port,
                stream_port=None,
                use_pregenerated_stubs=False,
            )
        result = inspect_active_pad(
            connection,
            launch_config,
            plan=(3, 2),
            fuel_stages=(2, 1),
            rpc_timeout=config.rpc_timeout,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        result = dict(ready=False, failures=[message], error=message)
    finally:
        try:
            if connection is not None:
                try:
                    with rpc_deadline(min(config.rpc_timeout, 3.0)):
                        connection.close()
                except Exception as error:
                    message = f"Inspection connection cleanup failed: {error}"
                    result = dict(result or {}, ready=False)
                    result["failures"] = list(result.get("failures", [])) + [message]
                    result["error"] = message
        finally:
            if ownership is not None:
                ownership.release()
    return _jsonable(result)


def preflight_subprocess(config):
    command = [
        config.python,
        "-m",
        "orbiter.control",
        "--inspect-active",
        "--project-root",
        str(config.project_root),
        "--launch-model",
        str(config.launch_model),
        "--capture-model",
        str(config.capture_model),
        "--run-root",
        str(config.run_root),
        "--rpc-port",
        str(config.rpc_port),
        "--rpc-timeout",
        str(config.rpc_timeout),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=config.project_root,
            capture_output=True,
            text=True,
            timeout=PREFLIGHT_WALL_TIMEOUT,
            check=False,
        )
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or not isinstance(result.get("ready"), bool):
            raise ValueError("Inspection did not return a readiness result")
        if completed.returncode != 0 and result["ready"]:
            raise ValueError("Inspection exited before completing normally")
        return result
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        message = f"Cannot inspect the active launchpad vessel: {error}"
        return dict(ready=False, failures=[message], error=message)


def _signal_group(process, signum):
    # Flight workers have a dedicated session; never signal the panel's group.
    os.killpg(process.pid, signum)


def _read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _recent_events(path):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            length = handle.tell()
            handle.seek(max(0, length - 262_144))
            if length > 262_144:
                handle.readline()
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A concurrently written final line may be incomplete.
        if isinstance(row, dict):
            records.append(row)
    return records


def _phase(event, previous):
    if event.startswith("ascent_") or event == "launch_handoff":
        return "ascent"
    if event in {"launch_ascent_end", "mission_plan", "mission_window_rejected"}:
        return "transfer_planning"
    if event.startswith("maneuver_"):
        return "maneuver"
    if event == "target_soi_coast":
        return "transfer_coast"
    if event.startswith("capture_"):
        return "capture_coast" if event == "capture_coast" else "capture"
    if event == "independent_orbital_period_probe":
        return "orbit_verification"
    return previous


def _control_release_evidence(path):
    """Require a finalizer after the last possible control operation."""
    invoked = handoff = unknown = False
    last_activity = last_release = -1
    source, errors = None, []
    fields = {
        "ascent_controls_released": "errors",
        "launch_ascent_end": "cleanup_errors",
        "mission_end": "cleanup_errors",
    }
    try:
        with path.open() as handle:
            for index, line in enumerate(handle):
                try:
                    record = json.loads(line)
                    event = record["event"]
                except (ValueError, KeyError, TypeError):
                    unknown, last_activity = True, index
                    continue
                invoked |= event == "mission_invocation"
                handoff |= event == "launch_handoff"
                if event in fields:
                    value = record.get(fields[event])
                    errors = (
                        value
                        if isinstance(value, list)
                        else ["Cleanup result is missing"]
                    )
                    if not errors:
                        last_release, source = index, event
                    else:
                        last_activity = index
                elif event != "mission_invocation":
                    last_activity = index
    except OSError:
        return dict(
            verified=False, source=None, errors=["Flight cleanup record is missing"]
        )
    verified = last_release >= last_activity and last_release >= 0
    if not handoff and invoked and not unknown and last_activity < 0:
        verified, source = True, "interrupted_before_control_handoff"
    return dict(verified=verified, source=source, errors=errors if not verified else [])


def _complete_success(result):
    if not isinstance(result, dict):
        return False
    launch, mission = result.get("launch"), result.get("mission")
    return bool(
        result.get("success") is True
        and result.get("end_to_end_from_launch_verified") is True
        and isinstance(launch, dict)
        and launch.get("success") is True
        and isinstance(mission, dict)
        and isinstance(mission.get("independent_coast"), dict)
        and mission["independent_coast"].get("success") is True
        and not launch.get("cleanup_errors")
        and not mission.get("cleanup_errors")
    )


class MissionControl:
    def __init__(
        self,
        config,
        *,
        inspector=preflight_subprocess,
        popen=subprocess.Popen,
        signal_worker=_signal_group,
        clock=time.monotonic,
        monitor=True,
    ):
        self.config = config
        self.inspector, self.popen = inspector, popen
        self.signal_worker, self.clock = signal_worker, clock
        self.monitor = monitor
        self.lock = threading.RLock()
        self.csrf_token = secrets.token_urlsafe(32)
        self.server_instance = secrets.token_hex(16)
        self.worker = self.output_handle = self.monitor_thread = None
        self.mission = None
        self.state, self.phase = "idle", "idle"
        self.error = self.telemetry = self.last_event = None
        self.preflight = dict(
            ready=False, failures=["Vessel has not been inspected yet"]
        )
        self.preflight_at = -math.inf
        self.closed = False

    def session(self):
        errors = self.config.model_errors()
        return dict(
            application="orbiter-autopilot",
            protocol_version=1,
            server_instance=self.server_instance,
            project_root=str(self.config.project_root),
            rpc_port=self.config.rpc_port,
            launch_model=str(self.config.launch_model),
            capture_model=str(self.config.capture_model),
            csrf_token=self.csrf_token,
            destinations=list(DESTINATIONS),
            altitude=dict(
                minimum_m=MIN_ALTITUDE_M,
                maximum_m=MAX_ALTITUDE_M,
                default_m=DEFAULT_ALTITUDE_M,
            ),
            models_ready=not errors,
            model_errors=errors,
        )

    def _refresh_preflight_locked(self, *, force=False):
        if self.worker is not None or self.closed:
            return
        if flight_busy(rpc_port=self.config.rpc_port):
            message = "Another Orbiter flight worker owns this KSP connection; wait for its controls to be released."
            self.preflight = dict(ready=False, failures=[message], error=message)
            self.preflight_at = self.clock()
            return
        if not force and self.clock() - self.preflight_at < PREFLIGHT_CACHE_SECONDS:
            return
        errors = self.config.model_errors()
        try:
            result = (
                dict(ready=False, failures=errors, error="; ".join(errors))
                if errors
                else self.inspector(self.config)
            )
            if not isinstance(result, dict) or not isinstance(
                result.get("ready"), bool
            ):
                raise ValueError("Inspection did not return a readiness result")
            self.preflight = result
        except Exception as error:
            message = f"Cannot inspect the active vessel: {error}"
            self.preflight = dict(ready=False, failures=[message], error=message)
        self.preflight_at = self.clock()

    def _read_progress_locked(self):
        if self.mission is None:
            return
        directory = Path(self.mission["output_dir"])
        for record in _recent_events(directory / "mission.jsonl"):
            event = str(record.get("event", ""))
            self.phase = _phase(event, self.phase)
            self.last_event = record
            if "telemetry" in event or event == "independent_orbital_period_probe":
                self.telemetry = record
        if self.state in {"succeeded", "failed", "stopped", "stopping"}:
            self.phase = self.state

    def _finish_locked(self, process, returncode):
        if self.worker is not process:
            return
        self._read_progress_locked()
        result = _read_json(Path(self.mission["output_dir"]) / "mission.json")
        stopped = self.state == "stopping"
        success = returncode == 0 and _complete_success(result)
        cleanup = _control_release_evidence(
            Path(self.mission["output_dir"]) / "mission.jsonl"
        )
        self.state = (
            "stopped"
            if stopped and cleanup["verified"]
            else "succeeded"
            if success and not stopped
            else "failed"
        )
        self.phase = self.state
        self.mission.update(
            returncode=returncode,
            result=result,
            control_cleanup=cleanup,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        if stopped and not cleanup["verified"]:
            self.error = "Worker exited, but flight-control release was not verified; inspect KSP."
            if cleanup["errors"]:
                self.error += " " + "; ".join(map(str, cleanup["errors"]))
        elif success or stopped:
            self.error = None
        else:
            self.error = (result.get("error") if result else None) or (
                "Mission summary is missing or invalid; orbit verification is unconfirmed"
                if result is None
                else f"Mission worker exited with status {returncode} without complete launch-to-orbit verification"
            )
        if self.output_handle is not None:
            self.output_handle.close()
            self.output_handle = None
        self.worker = None
        self.preflight_at = -math.inf

    def _watch_worker(self, process):
        returncode = process.wait()
        with self.lock:
            self._finish_locked(process, returncode)

    def _poll_locked(self):
        if self.worker is not None:
            returncode = self.worker.poll()
            if returncode is not None:
                self._finish_locked(self.worker, returncode)

    def _snapshot_locked(self):
        errors = self.config.model_errors()
        return _jsonable(
            dict(
                server_instance=self.server_instance,
                state=self.state,
                active=self.worker is not None,
                can_run=(
                    not self.closed
                    and self.worker is None
                    and not errors
                    and self.preflight.get("ready") is True
                ),
                preflight=self.preflight,
                phase=self.phase,
                telemetry=self.telemetry,
                last_event=self.last_event,
                error=self.error,
                mission=self.mission,
                models_ready=not errors,
                model_errors=errors,
            )
        )

    def status(self):
        with self.lock:
            self._poll_locked()
            self._read_progress_locked()
            self._refresh_preflight_locked()
            return self._snapshot_locked()

    def run(self, payload):
        target, altitude = validate_mission(payload)
        requested_at = self.clock()
        with self.lock:
            self._poll_locked()
            if self.closed:
                raise ControlError("The mission panel is shutting down", 503)
            if self.worker is not None:
                raise ControlError("A mission already owns the flight controls", 409)
            errors = self.config.model_errors()
            if errors:
                raise ControlError("; ".join(errors), 409)
            # An inspection already in progress when Run was clicked is fresh
            # enough; do not queue a second 18-second read after it completes.
            self._refresh_preflight_locked(force=self.preflight_at < requested_at)
            if self.preflight.get("ready") is not True:
                failures = self.preflight.get("failures", [])
                raise ControlError(
                    "Active vessel is not ready: " + "; ".join(map(str, failures)), 409
                )
            mission_id = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + secrets.token_hex(4)
            )
            directory = self.config.run_root / mission_id
            directory.mkdir(parents=True, exist_ok=False)
            command = [
                self.config.python,
                "-m",
                "orbiter.mission_live",
                "--fly",
                "--launch-active",
                "--launch-model",
                str(self.config.launch_model),
                "--capture-model",
                str(self.config.capture_model),
                "--stage-plan",
                "3,2",
                "--fuel-stages",
                "2,1",
                "--target",
                target,
                "--target-altitude",
                str(altitude),
                "--output-dir",
                str(directory),
                "--address",
                "127.0.0.1",
                "--rpc-port",
                str(self.config.rpc_port),
                "--rpc-timeout",
                str(self.config.rpc_timeout),
                "--wall-timeout",
                "2400",
            ]
            self.mission = dict(
                id=mission_id,
                target=target,
                target_altitude_m=altitude,
                output_dir=str(directory),
                started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=None,
                returncode=None,
                result=None,
            )
            self.error = self.telemetry = self.last_event = None
            self.state, self.phase = "running", "preparing"
            try:
                request = dict(
                    validated_payload=dict(target=target, target_altitude_m=altitude),
                    mode="active_pad",
                    accepted_at=self.mission["started_at"],
                    preflight=self.preflight,
                    worker_argv=command,
                    job_id=mission_id,
                )
                with (directory / "panel_request.json").open("x") as handle:
                    json.dump(_jsonable(request), handle, indent=2, allow_nan=False)
                    handle.write("\n")
                self.output_handle = (directory / "console.log").open("x")
                self.worker = self.popen(
                    command,
                    cwd=self.config.project_root,
                    stdin=subprocess.DEVNULL,
                    stdout=self.output_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    shell=False,
                )
            except Exception as error:
                if self.output_handle is not None:
                    self.output_handle.close()
                self.output_handle = None
                self.state = self.phase = "failed"
                self.error = f"Could not start mission worker: {error}"
                self.mission["finished_at"] = datetime.now(timezone.utc).isoformat()
                raise ControlError(self.error, 500) from error
            if self.monitor:
                self.monitor_thread = threading.Thread(
                    target=self._watch_worker,
                    args=(self.worker,),
                    name="orbiter-mission-worker",
                    daemon=True,
                )
                self.monitor_thread.start()
            return self._snapshot_locked()

    def stop(self, *, expected_job_id=None, server_instance=None):
        with self.lock:
            self._poll_locked()
            if expected_job_id is not None or server_instance is not None:
                if (
                    not isinstance(expected_job_id, str)
                    or not expected_job_id
                    or not isinstance(server_instance, str)
                    or server_instance != self.server_instance
                    or self.mission is None
                    or self.mission["id"] != expected_job_id
                ):
                    raise ControlError(
                        "The flight or server changed; refresh status before stopping",
                        409,
                    )
            if self.worker is not None and self.state != "stopping":
                self.state, self.phase = "stopping", "stopping"
                try:
                    self.signal_worker(self.worker, signal.SIGINT)
                except ProcessLookupError:
                    self._poll_locked()
                except OSError as error:
                    self.state = "running"
                    self.error = f"Could not request worker cleanup: {error}"
                    raise ControlError(self.error, 500) from error
            return self._snapshot_locked()

    def close(self, *, cleanup_timeout=30.0):
        with self.lock:
            self.closed = True
            try:
                self.stop()
            except ControlError:
                pass  # Preserve ownership and allow the existing worker to finish.
            process = self.worker
        if process is None:
            return
        try:
            process.wait(timeout=cleanup_timeout)
        except subprocess.TimeoutExpired as error:
            with self.lock:
                self.error = (
                    "Shutdown incomplete: the mission worker remains active while "
                    "flight-control cleanup finishes. The panel will stay available."
                )
            raise ShutdownIncomplete(self.error) from error
        finally:
            with self.lock:
                returncode = process.poll()
                if returncode is not None:
                    self._finish_locked(process, returncode)
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, controller):
        if address[0] != "127.0.0.1":
            raise ValueError("The mission panel may bind only to 127.0.0.1")
        self.controller = controller
        super().__init__(address, ControlHandler)

    def server_close(self):
        self.controller.close()
        super().server_close()


class ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _origin(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update({"127.0.0.1", "localhost"})
        host = self.headers.get("Host", "").lower()
        if host not in allowed:
            raise ControlError("This panel accepts only loopback Host headers", 403)
        expected = "http://" + host
        origin = self.headers.get("Origin")
        if origin is not None and origin != expected:
            raise ControlError("Cross-origin requests are not allowed", 403)
        return expected

    def _send(self, status, value, *, html=False):
        body = value if html else json.dumps(_jsonable(value), allow_nan=False).encode()
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8" if html else "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            self._origin()
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                page = resources.files("orbiter").joinpath("control.html").read_bytes()
                self._send(200, page, html=True)
            elif path == "/api/session":
                self._send(200, self.server.controller.session())
            elif path == "/api/status":
                self._send(200, self.server.controller.status())
            else:
                raise ControlError("Not found", 404)
        except ControlError as error:
            self._send(error.status, dict(error=str(error)))
        except Exception as error:
            self._send(500, dict(error=f"Panel request failed: {error}"))

    def do_POST(self):
        try:
            expected = self._origin()
            if self.headers.get("Origin") != expected:
                raise ControlError("A same-origin request is required", 403)
            provided = self.headers.get("X-Orbiter-CSRF", "")
            if not secrets.compare_digest(provided, self.server.controller.csrf_token):
                raise ControlError(
                    "Missing or invalid session token; reload the panel", 403
                )
            if self.headers.get_content_type() != "application/json":
                raise ControlError("Requests must contain JSON", 415)
            if self.headers.get("Transfer-Encoding"):
                raise ControlError("Chunked request bodies are not supported")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as error:
                raise ControlError("A valid Content-Length is required") from error
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ControlError("Request body exceeds the supported size", 413)
            self.connection.settimeout(5.0)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ControlError("Incomplete request body")
            try:
                payload = json.loads(
                    raw,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(value)
                    ),
                )
            except (ValueError, UnicodeError) as error:
                raise ControlError("Request body is not valid finite JSON") from error
            path = urlsplit(self.path).path
            if path == "/api/run":
                self._send(202, self.server.controller.run(payload))
            elif path == "/api/stop":
                if payload == {}:
                    result = self.server.controller.stop()
                elif isinstance(payload, dict) and set(payload) == {
                    "expected_job_id",
                    "server_instance",
                }:
                    if not all(
                        isinstance(value, str) and value for value in payload.values()
                    ):
                        raise ControlError("Stop identities must be nonempty strings")
                    result = self.server.controller.stop(**payload)
                else:
                    raise ControlError(
                        "Stop expects an empty object or both flight and server identities"
                    )
                self._send(202 if result["active"] else 200, result)
            else:
                raise ControlError("Not found", 404)
        except ControlError as error:
            self._send(error.status, dict(error=str(error)))
        except Exception as error:
            self._send(500, dict(error=f"Panel request failed: {error}"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project directory containing models/ and runs/",
    )
    parser.add_argument(
        "--launch-model",
        type=Path,
        help="Ascent checkpoint; defaults to PROJECT/models/transfer_ascent_ppo.zip",
    )
    parser.add_argument(
        "--capture-model",
        type=Path,
        help="Capture checkpoint; defaults to PROJECT/models/moon_capture.zip",
    )
    parser.add_argument(
        "--run-root", type=Path, help="Flight records; defaults to PROJECT/runs/control"
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port on 127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=50000)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Append backend output to this file, independent of the launching game",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the local URL without opening a browser",
    )
    parser.add_argument("--inspect-active", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (1 <= args.port <= 65535 and 1 <= args.rpc_port <= 65535):
        parser.error("Ports must be between 1 and 65535")
    if not math.isfinite(args.rpc_timeout) or args.rpc_timeout <= 0:
        parser.error("RPC timeout must be positive and finite")
    args.project_root = args.project_root.resolve()
    if not args.project_root.is_dir():
        parser.error("Project root must be an existing directory")
    for key, default in (
        ("launch_model", "models/transfer_ascent_ppo.zip"),
        ("capture_model", "models/moon_capture.zip"),
        ("run_root", "runs/control"),
    ):
        value = getattr(args, key)
        path = Path(default) if value is None else value
        setattr(
            args,
            key,
            (args.project_root / path).resolve()
            if not path.is_absolute()
            else path.resolve(),
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("a", buffering=1) as handle:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
    config = ControlConfig(
        args.project_root,
        args.launch_model,
        args.capture_model,
        args.run_root,
        rpc_port=args.rpc_port,
        rpc_timeout=args.rpc_timeout,
    )
    if args.inspect_active:
        result = inspect_active(config)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0 if result.get("ready") else 1
    controller = MissionControl(config)
    server = ControlServer(("127.0.0.1", args.port), controller)
    server.timeout = 0.25
    stopping = threading.Event()
    previous_handlers = {}

    def stop_server(signum, frame):
        stopping.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop_server)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Orbiter mission panel: {url}", flush=True)
    server_closed = False
    try:
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception as error:
                print(f"Open {url} in your browser: {error}", file=sys.stderr)
        shutdown_pending = False
        while True:
            if stopping.is_set() or shutdown_pending:
                with controller.lock:
                    controller._poll_locked()
                    active = controller.worker is not None
                if stopping.is_set() or not active:
                    try:
                        server.server_close()
                        server_closed = True
                        break
                    except ShutdownIncomplete as error:
                        print(str(error), file=sys.stderr, flush=True)
                        shutdown_pending = True
                        stopping.clear()
            server.handle_request()
    finally:
        while not server_closed:
            try:
                server.server_close()
                server_closed = True
            except ShutdownIncomplete as error:
                print(str(error), file=sys.stderr, flush=True)
                while True:
                    with controller.lock:
                        controller._poll_locked()
                        active = controller.worker is not None
                    if not active:
                        break
                    server.handle_request()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
