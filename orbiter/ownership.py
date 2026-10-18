"""Process-owned, nonblocking control lease for one user's kRPC endpoint.

The flight worker holds the descriptor itself, so a panel/backend crash cannot
release a running flight's lease. Lock files are never deleted or rewritten:
replacing the inode would let a second process lock a different file.
"""

from __future__ import annotations

import errno
import getpass
import hashlib
import ipaddress
import os
from pathlib import Path
import stat
import tempfile


class FlightOwnershipBusy(RuntimeError):
    """Another process already owns flight controls for this endpoint."""


def _endpoint(address, rpc_port):
    if not isinstance(address, str) or not address.strip():
        raise ValueError("Flight ownership needs a nonempty endpoint address")
    if type(rpc_port) is not int or not 1 <= rpc_port <= 65535:
        raise ValueError("Flight ownership RPC port must be an integer in 1..65535")
    host = address.strip().lower().rstrip(".")
    if not host:
        raise ValueError("Flight ownership needs a nonempty endpoint address")
    try:
        parsed = ipaddress.ip_address(host)
        host = "loopback" if parsed.is_loopback else parsed.compressed
    except ValueError:
        if host == "localhost":
            host = "loopback"
    return host, rpc_port


class FlightOwnership:
    """Hold one endpoint's OS lock until release, context exit or process exit.

    acquire() is nonblocking and raises FlightOwnershipBusy on contention. Paths
    depend on the OS user and endpoint, never a checkout or current directory.
    Unix uses the system /tmp directory even when launchers have different
    TMPDIR values; Windows uses the user's normal temporary directory.
    """

    def __init__(self, address="127.0.0.1", rpc_port=50000):
        host, port = _endpoint(address, rpc_port)
        self.address, self.rpc_port = address, port
        user = (
            str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser().casefold()
        )
        digest = hashlib.sha256(f"{user}\0{host}\0{port}".encode()).hexdigest()
        directory = Path("/tmp") if os.name == "posix" else Path(tempfile.gettempdir())
        self.path = directory / f"orbiter-flight-{digest}.lock"
        self._fd = None

    def acquire(self):
        if self._fd is not None:
            return self
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("Flight ownership lock must be a regular file")
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise PermissionError("Flight ownership lock belongs to another user")
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            os.set_inheritable(fd, False)
            try:
                if os.name == "nt":
                    import msvcrt

                    # Windows permits a byte-range lock beyond EOF. No marker
                    # byte or metadata write is needed, even for an empty file.
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise FlightOwnershipBusy(
                        f"Another flight worker owns {self.address}:{self.rpc_port}"
                    ) from error
                raise
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def release(self):
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def flight_busy(address="127.0.0.1", rpc_port=50000):
    """Check contention without changing lock contents or contacting KSP.

    This is a momentary UI/preflight probe. The worker must acquire its own
    FlightOwnership lease before models or RPCs to close the probe/start race.
    Unexpected file/permission errors propagate rather than reporting ready.
    """
    ownership = FlightOwnership(address, rpc_port)
    try:
        ownership.acquire()
    except FlightOwnershipBusy:
        return True
    else:
        ownership.release()
        return False
