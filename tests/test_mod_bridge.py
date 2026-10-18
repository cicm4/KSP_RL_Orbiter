"""Exercise the real C# bridge against delayed HTTP, without KSP or physics."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import pytest


@pytest.fixture(scope="module", params=["portable", "ksp"])
def bridge_executable(tmp_path_factory, request):
    compiler, runtime = shutil.which("mcs"), shutil.which("mono")
    if not compiler or not runtime:
        pytest.skip("Mono compiler/runtime required for the headless native bridge test")
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path_factory.mktemp("mod-bridge")
    output = folder / "bridge.exe"
    references = ["-r:System.Web.Extensions"]
    if request.param == "ksp":
        managed = Path.home() / "Library/Application Support/Steam/steamapps/common/Kerbal Space Program/KSP.app/Contents/Resources/Data/Managed"
        if not (managed / "Assembly-CSharp.dll").is_file():
            pytest.skip("Local KSP assembly required to exercise its actual embedded JSON decoder")
        # Never add the game BCL to MONO_PATH: the standalone test must retain
        # the host runtime's mscorlib while loading only this managed decoder.
        for name in ("Assembly-CSharp.dll", "UnityEngine.dll", "UnityEngine.CoreModule.dll"):
            shutil.copy2(managed / name, folder / name)
        references += ["-define:KSP_JSON", f"-r:{folder / 'Assembly-CSharp.dll'}"]
    completed = subprocess.run(
        [compiler, "-target:exe", *references, f"-out:{output}",
         str(root / "orbiter/ksp_autopilot.cs"),
         str(root / "tests/mod_bridge_stubs.cs"),
         str(root / "tests/mod_bridge_harness.cs")],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return runtime, output


def _parse(bridge_executable, tmp_path, payload):
    runtime, executable = bridge_executable
    wire = tmp_path / "status.json"
    wire.write_text(json.dumps(payload))
    return subprocess.run([runtime, str(executable), "PARSE", str(wire)],
                          capture_output=True, text=True, timeout=10)


def _active_payload():
    # Real native-Run ascent reply: null fields, unknown nested arrays/objects,
    # preflight/telemetry and a later mission.id. Only output_dir is anonymized.
    return json.loads(Path(__file__).with_name("mod_bridge_active_status.json").read_text())


def test_real_active_wire_keeps_job_identity(bridge_executable, tmp_path):
    reply = _parse(bridge_executable, tmp_path, _active_payload())
    assert reply.returncode == 0, reply.stderr
    assert "JOB=20261018T225815Z-8a22be3f" in reply.stdout
    assert "VERIFIED=False" in reply.stdout


@pytest.mark.parametrize("mutation", ["missing_job", "numeric_job", "unsafe_job", "string_active", "string_altitude", "wrong_activity", "string_proof", "numeric_proof", "string_cleanup", "missing_cleanup_errors"])
def test_protocol_rejects_untyped_identity_and_proof(bridge_executable, tmp_path, mutation):
    payload = _active_payload()
    mission = payload["mission"]
    if mutation == "missing_job":
        del mission["id"]
    elif mutation == "numeric_job":
        mission["id"] = 123
    elif mutation == "unsafe_job":
        mission["id"] = 'job"injection'
    elif mutation == "string_active":
        payload["active"] = "true"
    elif mutation == "string_altitude":
        mission["target_altitude_m"] = "50000"
    elif mutation == "wrong_activity":
        payload["state"] = "succeeded"
    elif mutation in ("string_proof", "numeric_proof"):
        mission["result"] = {"success": "true" if mutation == "string_proof" else 1}
    elif mutation == "string_cleanup":
        mission["control_cleanup"] = {"verified": "true", "errors": []}
    else:
        mission["control_cleanup"] = {"verified": True}
    reply = _parse(bridge_executable, tmp_path, payload)
    assert reply.returncode == 2, reply.stdout + reply.stderr
    assert "invalid" in reply.stderr.lower()


@pytest.mark.parametrize("complete", [True, False])
def test_terminal_orbit_message_requires_complete_proof(bridge_executable, tmp_path, complete):
    payload = _active_payload()
    payload.update(state="succeeded", active=False)
    payload["mission"]["result"] = {
        "success": True, "end_to_end_from_launch_verified": True,
        "launch": {"success": True},
        "mission": {"independent_coast": {"success": complete}, "unrelated": {"success": True}},
    }
    reply = _parse(bridge_executable, tmp_path, payload)
    assert reply.returncode == 0, reply.stderr
    assert f"VERIFIED={complete}" in reply.stdout
    expected = "Orbit verified. A full unpowered orbit completed." if complete else "complete orbit proof is missing"
    assert expected in reply.stdout


@pytest.mark.parametrize("action", ["STOP", "SCENE"])
@pytest.mark.parametrize("delay_stage", ["session", "accepted_run"])
def test_stop_cancels_before_send_and_reconciles_after_accept(bridge_executable, tmp_path, action, delay_stage):
    runtime, executable = bridge_executable
    session_waiting, release_session = threading.Event(), threading.Event()
    delay_next = threading.Event()
    posts = []
    status = dict(state="idle", phase="idle", active=False, can_run=True,
                  preflight=dict(ready=True, failures=[]), mission=None)
    root = str(tmp_path.resolve())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/session":
                if delay_next.is_set():
                    delay_next.clear()
                    session_waiting.set()
                    assert release_session.wait(6), "Test did not release delayed handshake"
                self.reply(dict(application="orbiter-autopilot", protocol_version=1,
                                server_instance="bridge-test-instance", csrf_token="bridge-test-token",
                                project_root=root, rpc_port=50000,
                                launch_model=root + "/models/transfer_ascent_ppo.zip",
                                capture_model=root + "/models/moon_capture.zip"))
            elif self.path == "/api/status":
                self.reply(status)
            else:
                self.reply(dict(error="Not found"), 404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append((self.path, payload))
            assert self.headers["Origin"] == f"http://127.0.0.1:{self.server.server_address[1]}"
            assert self.headers["X-Orbiter-CSRF"] == "bridge-test-token"
            if self.path == "/api/run":
                status.update(state="running", phase="preparing", active=True, can_run=False,
                              mission=dict(id="regression-job", target="Mun", target_altitude_m=50000))
                if delay_stage == "accepted_run":
                    session_waiting.set()
                    assert release_session.wait(6), "Test did not release accepted Run response"
                    # Acceptance happened, but the response is ambiguous. The
                    # actual client must reconcile status and never retry Run.
                    self.reply(dict(error="Accepted job but response failed"), 500)
                else:
                    self.reply(status, 202)
            elif self.path == "/api/stop":
                status.update(state="stopped", phase="stopped", active=False,
                              mission=dict(id="regression-job", target="Mun", target_altitude_m=50000,
                                           control_cleanup=dict(verified=True, errors=[])))
                self.reply(status, 202)
            else:
                self.reply(dict(error="Not found"), 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    process = subprocess.Popen(
        [runtime, str(executable), str(server.server_address[1]), root],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    lines = queue.Queue()
    reader = threading.Thread(target=lambda: [lines.put(line.strip()) for line in process.stdout], daemon=True)
    reader.start()

    def expect(value):
        deadline = time.monotonic() + 10
        observed = []
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            observed.append(line)
            if line == value:
                return
        raise AssertionError(f"Expected {value!r}; stdout was {observed!r}; process={process.poll()}")

    try:
        expect("READY")
        if delay_stage == "session":
            delay_next.set()
        process.stdin.write("RUN\n")
        process.stdin.flush()
        expect("RUN_QUEUED")
        assert session_waiting.wait(5), "Run did not reach the delayed HTTP boundary"
        process.stdin.write(action + "\n")
        process.stdin.flush()
        expect("STOP_QUEUED")
        release_session.set()
        process.wait(timeout=10)
        errors = process.stderr.read()
        if delay_stage == "session":
            assert not posts, f"Stop was queued before send, but bridge still posted: {posts!r}"
        else:
            assert [path for path, _ in posts] == ["/api/run", "/api/stop"], posts
            assert posts[1][1] == dict(expected_job_id="regression-job", server_instance="bridge-test-instance")
        assert process.returncode == 0, errors
    finally:
        release_session.set()
        if process.poll() is None:
            process.terminate()  # The fixture child has no KSP or real controls.
            process.wait(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
