"""The only generated-code execution boundary: a pinned, constrained container."""

from __future__ import annotations

import json
import os
import re
import selectors
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass

from challenge_cases import CodeCheck, strict_equal

MAX_OUTPUT_BYTES = 65536
MAX_SOURCE_BYTES = 131072

# This program is passed to Python INSIDE Docker, never executed on the host.
# Expected answers are deliberately not sent to the candidate's process.
WORKER = r'''
import contextlib
import copy
import io
import json
import sys

payload = json.load(sys.stdin)
namespace = {}
outputs = []
unchanged = []
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(payload["source"], "<candidate>", "exec"), namespace)
    for args in payload["inputs"]:
        before = copy.deepcopy(args)
        outputs.append(namespace["solve"](*args))
        unchanged.append(args == before)
print(json.dumps({"outputs": outputs, "unchanged": unchanged}, allow_nan=False))
'''


@dataclass(frozen=True)
class SandboxGrade:
    passed: bool
    state: str
    failed_checks: tuple[int, ...] = ()
    evidence: str = ""
    elapsed_seconds: float = 0.0


def docker_argv(command: str) -> tuple[str, ...]:
    words = tuple(shlex.split(command))
    if words not in (("docker",), ("sudo", "-n", "docker")):
        raise ValueError("--docker-command must be exactly 'docker' or 'sudo -n docker'")
    return words


def validate_image(image: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", image):
        raise ValueError("code tasks require --sandbox-image with an exact image@sha256:<64 lowercase hex> pin")


def sandbox_command(command: tuple[str, ...], image: str, name: str) -> list[str]:
    docker_argv(shlex.join(command))
    validate_image(image)
    if not re.fullmatch(r"challenge-code-[0-9a-f]{32}", name):
        raise ValueError("invalid sandbox container name")
    return [
        *command, "run", "--rm", "--pull", "never", "--name", name,
        "--network", "none", "--read-only", "--user", "65534:65534",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
        "--pids-limit", "64", "--oom-score-adj", "800",
        "--ulimit", "nofile=64:64", "--ulimit", "fsize=65536:65536",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
        "--workdir", "/tmp", "--entrypoint", "python3", "-i", image,
        "-I", "-c", WORKER,
    ]


def check_sandbox(command: tuple[str, ...], image: str) -> None:
    """Fail before inference if the supplied immutable image is unavailable."""
    docker_argv(shlex.join(command))
    validate_image(image)
    result = subprocess.run(
        [*command, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode:
        raise ValueError(f"pinned sandbox image unavailable: {result.stderr[:2000]}")


def grade_code(
    source: str, checks: tuple[CodeCheck, ...], *, command: tuple[str, ...],
    image: str, timeout: float = 8.0, require_unchanged: bool = False,
) -> SandboxGrade:
    if not 0 < timeout <= 10:
        raise ValueError("sandbox timeout must be >0 and <=10 seconds")
    name = f"challenge-code-{uuid.uuid4().hex}"
    argv = sandbox_command(command, image, name)
    if len(source.encode()) > MAX_SOURCE_BYTES:
        return SandboxGrade(False, "source_too_large")
    payload = json.dumps({"source": source, "inputs": [c.args for c in checks]}).encode()
    started = time.monotonic()
    raw = bytearray()
    state = "completed"
    returncode = None
    cleanup_failed = False
    try:
        # stdin is a bounded fixture file, never a candidate-controlled host path.
        with tempfile.TemporaryFile() as incoming:
            incoming.write(payload)
            incoming.seek(0)
            process = subprocess.Popen(argv, stdin=incoming, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert process.stdout is not None
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    deadline = started + timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            state = "timeout"
                            break
                        chunk = os.read(process.stdout.fileno(), 8192)
                        if not chunk:
                            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                            break
                        raw.extend(chunk)
                        if len(raw) > MAX_OUTPUT_BYTES:
                            state = "output_limit"
                            break
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)
                process.stdout.close()
    except subprocess.TimeoutExpired:
        state = "timeout"
    except OSError as exc:
        state = "sandbox_unavailable"
        raw.extend(str(exc).encode()[:2000])
    finally:
        # Killing a docker CLI does NOT kill its container. Always remove this
        # unique invocation by name, including on deadline and daemon failures.
        try:
            cleanup = subprocess.run(
                [*command, "rm", "-f", name], stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, timeout=10, check=False,
            )
            if cleanup.returncode and b"No such container" not in cleanup.stderr:
                cleanup_failed = True
        except (OSError, subprocess.TimeoutExpired):
            cleanup_failed = True
    elapsed = time.monotonic() - started
    evidence = raw[:MAX_OUTPUT_BYTES].decode(errors="replace")
    if cleanup_failed:
        return SandboxGrade(False, "sandbox_cleanup_failed", evidence=evidence, elapsed_seconds=elapsed)
    if state != "completed" or returncode != 0:
        return SandboxGrade(False, state if state != "completed" else "execution_error", evidence=evidence, elapsed_seconds=elapsed)
    try:
        observed = json.loads(raw)
        outputs = observed["outputs"]
        unchanged = observed["unchanged"]
        if not isinstance(outputs, list) or not isinstance(unchanged, list) or len(outputs) != len(checks) or len(unchanged) != len(checks):
            raise ValueError("wrong result shape")
    except (ValueError, KeyError, TypeError, RecursionError):
        return SandboxGrade(False, "invalid_output", evidence=evidence, elapsed_seconds=elapsed)
    failed = tuple(
        i for i, (value, check) in enumerate(zip(outputs, checks, strict=True))
        if not strict_equal(value, check.expected) or (require_unchanged and unchanged[i] is not True)
    )
    return SandboxGrade(not failed, "graded", failed, evidence, elapsed)
