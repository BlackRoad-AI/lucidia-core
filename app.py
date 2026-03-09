"""Simple coding portal for executing Python snippets."""

from __future__ import annotations

import ast
import io
import math
import os
import re
import secrets
import subprocess
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

import werkzeug
from flask import Flask, jsonify, render_template, request, session

if not hasattr(werkzeug, "__version__"):
    try:  # pragma: no cover - compatibility shim
        from werkzeug import __about__ as _werkzeug_about
    except ImportError:  # pragma: no cover - fallback when metadata missing
        werkzeug.__version__ = "0"
    else:  # pragma: no cover - executed when metadata available
        werkzeug.__version__ = getattr(_werkzeug_about, "__version__", "0")
        del _werkzeug_about

try:
    import sympy as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "lucidia-dev-secret")

# Expose a restricted set of safe built-in functions to executed code.
SAFE_BUILTINS: dict[str, object] = {
    "print": print,
    "abs": abs,
    "round": round,
    "pow": pow,
    "enumerate": enumerate,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
}

# Preserve variables and functions defined per user session.
SESSION_STATES: dict[str, dict[str, object]] = {}
STATE_LOCK = threading.RLock()


_WORKSPACE_ENV = os.environ.get("LUCIDIA_WORKSPACE")
WORKSPACE_ROOT = (
    Path(_WORKSPACE_ENV).resolve()
    if _WORKSPACE_ENV
    else Path(__file__).resolve().parent.parent
)


# Only allow installing a curated set of packages.
ALLOWLISTED_PACKAGES = {
    "itsdangerous": "itsdangerous==2.2.0",
}
_ALLOWLIST_LOOKUP = {
    **{name: spec for name, spec in ALLOWLISTED_PACKAGES.items()},
    **{spec.lower(): spec for spec in ALLOWLISTED_PACKAGES.values()},
}


class CodeValidationError(ValueError):
    """Raised when submitted code contains unsafe operations."""


def _get_session_id() -> str:
    """Return a stable session identifier for the current client."""
    session_id = session.get("lucidia_session_id")
    if session_id is None:
        session_id = secrets.token_hex(16)
        session["lucidia_session_id"] = session_id
    return session_id


def _load_session_state(session_id: str) -> dict[str, object]:
    """Return a shallow copy of the stored session state."""
    with STATE_LOCK:
        state = SESSION_STATES.setdefault(session_id, {})
        return dict(state)


def _persist_session_state(session_id: str, new_state: dict[str, object]) -> None:
    """Persist sanitized execution locals for the session."""
    sanitized = {k: v for k, v in new_state.items() if k != "__builtins__"}
    with STATE_LOCK:
        SESSION_STATES[session_id] = sanitized


def reset_all_session_state() -> None:
    """Helper for tests to reset every stored session."""
    with STATE_LOCK:
        SESSION_STATES.clear()


def _path_contains_symlink(path: Path, root: Path) -> bool:
    """Check whether ``path`` traverses any symbolic links beneath ``root``."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate(code: str) -> None:
    """Lightweight static analysis to reject obviously unsafe code."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise CodeValidationError("import statements are not allowed")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"):
                raise CodeValidationError("attribute access restricted to math module")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in SAFE_BUILTINS:
                    raise CodeValidationError(f"call to '{func.id}' is not permitted")
            elif isinstance(func, ast.Attribute):
                if not (isinstance(func.value, ast.Name) and func.value.id == "math"):
                    raise CodeValidationError("only calls to math functions are permitted")


# --- routes ----------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.post("/run")
def run_code():
    """Execute user-supplied Python code and return the output."""
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")

    try:
        _validate(code)
    except CodeValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    session_id = _get_session_id()
    session_state = _load_session_state(session_id)
    exec_locals: dict[str, object] = {"math": math}
    exec_locals.update({k: v for k, v in session_state.items() if k != "__builtins__"})
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            exec(code, {"__builtins__": SAFE_BUILTINS}, exec_locals)
        output = stdout.getvalue()
    except Exception as exc:  # noqa: BLE001 - broad for user feedback
        output = f"Error: {exc}"
    finally:
        _persist_session_state(session_id, exec_locals)
    return jsonify({"output": output})


@app.post("/math")
def evaluate_math():
    """Evaluate a mathematical expression and optionally report its derivative."""
    if sp is None:
        return jsonify({"error": "sympy is not installed"}), 500
    data = request.get_json(silent=True) or {}
    expr = data.get("expression")
    if not expr:
        return jsonify({"error": "missing expression"}), 400
    curious = data.get("curious")
    try:
        sym_expr = sp.sympify(expr)
    except sp.SympifyError as exc:
        return jsonify({"error": str(exc)}), 400
    response: dict[str, str] = {"result": str(sym_expr)}
    if curious:
        symbols = list(sym_expr.free_symbols)
        if symbols:
            response["derivative"] = str(sp.diff(sym_expr, symbols[0]))
    return jsonify(response)


@app.post("/install")
def install_package():
    """Install an allowlisted Python package via ``pip`` within the environment."""
    data = request.get_json(silent=True) or {}
    package = data.get("package")
    if not package:
        return jsonify({"error": "missing package"}), 400

    if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:==[A-Za-z0-9_.-]+)?", package):
        return jsonify({"error": "invalid package spec"}), 400
    normalized = package.strip().lower()
    spec = _ALLOWLIST_LOOKUP.get(normalized)
    if spec is None:
        return jsonify({"error": "package not allowed"}), 403

    pip_env = os.environ | {"PIP_NO_INPUT": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            spec,
        ],
        capture_output=True,
        text=True,
        env=pip_env,
    )
    return (
        jsonify({"code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}),
        200 if proc.returncode == 0 else 500,
    )


@app.post("/git/clean")
def git_clean():
    """Reset and remove untracked files from a git repository."""
    data = request.get_json(silent=True) or {}
    raw_path = data.get("path")
    repo_path = Path(raw_path) if raw_path else WORKSPACE_ROOT
    if not repo_path.is_absolute():
        repo_path = WORKSPACE_ROOT / repo_path

    if _path_contains_symlink(repo_path, WORKSPACE_ROOT):
        return jsonify({"error": "path not allowed"}), 400

    try:
        resolved_repo = repo_path.resolve(strict=True)
    except (OSError, RuntimeError):
        return jsonify({"error": "invalid path"}), 400

    try:
        resolved_repo.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return jsonify({"error": "path not allowed"}), 400

    if _path_contains_symlink(resolved_repo, WORKSPACE_ROOT):
        return jsonify({"error": "path not allowed"}), 400

    if not resolved_repo.is_dir():
        return jsonify({"error": "invalid path"}), 400

    git_dir = resolved_repo / ".git"
    if not git_dir.is_dir():
        return jsonify({"error": "not a git repo"}), 400

    reset = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=resolved_repo,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["git", "clean", "-ffdx"],
        cwd=resolved_repo,
        capture_output=True,
        text=True,
    )
    output = reset.stdout + reset.stderr + clean.stdout + clean.stderr
    code = reset.returncode or clean.returncode
    return (
        jsonify({"code": code, "output": output}),
        200 if code == 0 else 500,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
