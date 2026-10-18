"""Endpoint ownership uses real OS locks and subprocesses, never game RPCs."""

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from orbiter.ownership import FlightOwnership, FlightOwnershipBusy, flight_busy


@pytest.fixture
def endpoint():
    # Never contend with a user's actual KSP endpoint during tests.
    return f"ownership-test-{uuid4().hex}.invalid", 50000


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for owned test subprocess")


def subprocess_env():
    return dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))


def test_lock_is_exclusive_and_probe_preserves_inode_and_contents(endpoint):
    address, port = endpoint
    ownership = FlightOwnership(address, port)
    assert not flight_busy(address, port)
    ownership.path.write_bytes(b"existing lock contents")
    inode = ownership.path.stat().st_ino
    with ownership:
        assert flight_busy(address, port)
        with pytest.raises(FlightOwnershipBusy):
            FlightOwnership(address, port).acquire()
        if os.name != "nt":
            assert ownership.path.read_bytes() == b"existing lock contents"
        assert ownership.path.stat().st_ino == inode
        if os.name != "nt":
            assert stat.S_IMODE(ownership.path.stat().st_mode) == 0o600
    assert not flight_busy(address, port)
    assert ownership.path.exists() and ownership.path.stat().st_ino == inode
    assert ownership.path.read_bytes() == b"existing lock contents"
    ownership.release()


@pytest.mark.parametrize("outcome", ["success", "error", "interrupt", "close_error"])
def test_main_holds_lock_during_rpc_and_cleanup_including_interrupts(
    endpoint, tmp_path, monkeypatch, outcome
):
    import krpc
    from orbiter import mission_live
    from types import SimpleNamespace

    address, port = endpoint
    observed = []

    def connect(**kwargs):
        observed.append(("connect", flight_busy(address, port)))
        return SimpleNamespace(close=close)

    def close():
        observed.append(("close", flight_busy(address, port)))
        if outcome == "close_error":
            raise RuntimeError("test socket cleanup error")

    def fly(*args, **kwargs):
        observed.append(("fly", flight_busy(address, port)))
        if outcome == "error":
            raise RuntimeError("test flight error")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return {"success": True}

    monkeypatch.setattr(krpc, "connect", connect)
    monkeypatch.setattr(mission_live, "fly_mission", fly)
    args = [
        "--fly",
        "--target",
        "Mun",
        "--capture-scripted",
        "--address",
        address,
        "--rpc-port",
        str(port),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            mission_live.main(args)
    else:
        assert mission_live.main(args) == (1 if outcome == "error" else 0)
    assert observed == [("connect", True), ("fly", True), ("close", True)]
    assert (tmp_path / "run/mission.json").is_file()
    assert not flight_busy(address, port)


def test_same_endpoint_is_shared_across_project_directories(endpoint, tmp_path):
    address, port = endpoint
    other = tmp_path / "another-project"
    other.mkdir()
    with FlightOwnership(address, port):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from orbiter.ownership import flight_busy; import sys; print(flight_busy(sys.argv[1], int(sys.argv[2])))",
                address,
                str(port),
            ],
            cwd=other,
            env=subprocess_env(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert result.returncode == 0 and result.stdout.strip() == "True"


def test_worker_owns_lock_after_launcher_exits(endpoint, tmp_path):
    address, port = endpoint
    ready, release = tmp_path / "ready", tmp_path / "release"
    worker = """
import sys,time
from pathlib import Path
from orbiter.ownership import FlightOwnership
with FlightOwnership(sys.argv[1], int(sys.argv[2])):
    Path(sys.argv[3]).write_text('ready')
    deadline=time.monotonic()+15
    while not Path(sys.argv[4]).exists() and time.monotonic()<deadline:
        time.sleep(.02)
"""
    launcher = """
import subprocess,sys
worker=subprocess.Popen([sys.executable,'-c',sys.argv[1],*sys.argv[2:]], stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True,start_new_session=True)
print(worker.pid,flush=True)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            launcher,
            worker,
            address,
            str(port),
            str(ready),
            str(release),
        ],
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    pid = int(result.stdout.strip())
    try:
        wait_until(ready.exists)
        assert flight_busy(address, port), "Launcher exit released its worker's lock"
        with pytest.raises(FlightOwnershipBusy):
            FlightOwnership(address, port).acquire()
        release.write_text("release")
        wait_until(lambda: not flight_busy(address, port))
    finally:
        release.touch()
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_os_releases_lock_when_worker_exits(endpoint):
    address, port = endpoint
    worker = """
from orbiter.ownership import FlightOwnership
import sys,time
ownership=FlightOwnership(sys.argv[1],int(sys.argv[2])).acquire()
print('ready',flush=True)
time.sleep(10)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", worker, address, str(port)],
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        assert flight_busy(address, port)
    finally:
        process.terminate()
        process.communicate(timeout=5)
    assert not flight_busy(address, port)


def test_loopback_aliases_and_independent_ports():
    assert (
        FlightOwnership("localhost", 50000).path
        == FlightOwnership("127.0.0.1", 50000).path
    )
    assert (
        FlightOwnership("::1", 50000).path == FlightOwnership("127.0.0.1", 50000).path
    )
    assert (
        FlightOwnership("localhost", 50001).path
        != FlightOwnership("localhost", 50000).path
    )


@pytest.mark.parametrize(
    "address,port",
    [("", 50000), ("host", 0), ("host", 65536), ("host", 5.5), ("host", True)],
)
def test_invalid_endpoint_is_rejected(address, port):
    with pytest.raises(ValueError):
        FlightOwnership(address, port)


def test_main_rejects_second_flight_before_model_loading(
    endpoint, tmp_path, monkeypatch
):
    from orbiter import mission_live

    address, port = endpoint
    model = tmp_path / "model.zip"
    model.write_bytes(b"fake")
    monkeypatch.setattr(
        mission_live,
        "load_controller",
        lambda *args: pytest.fail("Busy worker loaded a model"),
    )
    args = [
        "--fly",
        "--target",
        "Mun",
        "--capture-scripted",
        "--launch-active",
        "--launch-model",
        str(model),
        "--stage-plan",
        "3,2",
        "--fuel-stages",
        "2,1",
        "--address",
        address,
        "--rpc-port",
        str(port),
        "--output-dir",
        str(tmp_path / "run"),
    ]
    with FlightOwnership(address, port):
        assert mission_live.main(args) == 1
        summary = json.loads((tmp_path / "run/mission.json").read_text())
        assert "FlightOwnershipBusy" in summary["error"]
        assert flight_busy(address, port)
    assert not flight_busy(address, port)


@pytest.mark.parametrize("write_fails", [False, True])
def test_main_retains_ownership_through_failure_summary_then_releases(
    endpoint, tmp_path, monkeypatch, write_fails
):
    from orbiter import mission_live

    address, port = endpoint
    model = tmp_path / "model.zip"
    model.write_bytes(b"fake")
    output = tmp_path / "run/mission.json"
    original_write = Path.write_text
    observed = []

    def fail_loading(*args):
        assert flight_busy(address, port)
        raise RuntimeError("test model load failed")

    def checked_write(path, *args, **kwargs):
        if path == output:
            observed.append(flight_busy(address, port))
            if write_fails:
                raise OSError("test summary disk error")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(mission_live, "load_controller", fail_loading)
    monkeypatch.setattr(Path, "write_text", checked_write)
    args = [
        "--fly",
        "--target",
        "Mun",
        "--capture-scripted",
        "--launch-active",
        "--launch-model",
        str(model),
        "--stage-plan",
        "3,2",
        "--fuel-stages",
        "2,1",
        "--address",
        address,
        "--rpc-port",
        str(port),
        "--output-dir",
        str(output.parent),
    ]
    if write_fails:
        with pytest.raises(OSError, match="summary disk"):
            mission_live.main(args)
    else:
        assert mission_live.main(args) == 1
    assert observed == [True]
    assert not flight_busy(address, port)
