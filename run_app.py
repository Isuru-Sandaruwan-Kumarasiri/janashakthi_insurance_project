"""
run_app.py — Unified local launcher for Janashakthi Life Insurance Proposal App.

Launches:
1. FastAPI Backend (http://localhost:8000)
2. Streamlit Frontend (http://localhost:8501)

Automatically handles process orchestration, logs aggregation, and clean shutdown.
"""

import sys
import os
import subprocess
import threading
import time
import signal
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
VENV_SCRIPTS = VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
PYTHON_EXE = str(VENV_SCRIPTS / ("python.exe" if os.name == "nt" else "python"))
STREAMLIT_EXE = str(VENV_SCRIPTS / ("streamlit.exe" if os.name == "nt" else "streamlit"))

# Global list of running processes
processes = []
shutting_down = False


def log_stream(stream, prefix, color_code):
    """Read a process stream line-by-line and print with prefix."""
    global shutting_down
    try:
        for line in iter(stream.readline, ""):
            if shutting_down:
                break
            if line:
                # Add terminal colors for readability
                # 36 = Cyan (Backend), 35 = Magenta (Frontend)
                colored_prefix = f"\033[{color_code}m{prefix}\033[0m"
                print(f"{colored_prefix} {line.strip()}", flush=True)
    except Exception:
        pass


def start_process(command, name, color_code):
    """Start a background process and log its output in a separate thread."""
    global shutting_down
    if shutting_down:
        return None

    # Run with stdout/stderr piped
    p = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(ROOT_DIR)
    )
    processes.append((p, name))

    # Spawn threads to consume output streams without blocking
    t_out = threading.Thread(target=log_stream, args=(p.stdout, f"[{name}]", color_code), daemon=True)
    t_err = threading.Thread(target=log_stream, args=(p.stderr, f"[{name} ERR]", color_code), daemon=True)
    t_out.start()
    t_err.start()

    return p


def shutdown(signum=None, frame=None):
    """Gracefully terminate all running subprocesses."""
    global shutting_down
    if shutting_down:
        return
    shutting_down = True
    print("\n\033[31m[Launcher] Shutting down all processes...\033[0m", flush=True)

    for p, name in processes:
        if p.poll() is None:
            print(f"[Launcher] Terminating {name} (PID: {p.pid})...", flush=True)
            p.terminate()

    # Wait a moment for processes to exit, then kill if still alive
    time.sleep(2)
    for p, name in processes:
        if p.poll() is None:
            print(f"[Launcher] Killing unresponsive {name} (PID: {p.pid})...", flush=True)
            p.kill()

    print("\033[32m[Launcher] All processes cleaned up. Exiting.\033[0m")
    sys.exit(0)


def main():
    # Register termination signals
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\033[1;36m" + "=" * 60)
    print("      Janashakthi Life Insurance Proposal Launcher")
    print("=" * 60 + "\033[0m")

    # Verify virtual environment exists
    if not VENV_DIR.exists():
        print(f"\033[31m[Error] Virtual environment not found at {VENV_DIR}\033[0m")
        print("Please create a virtual environment first: python -m venv .venv")
        sys.exit(1)

    # 1. Start FastAPI Backend (port 8000)
    print("[Launcher] Starting Backend (FastAPI on port 8000)...", flush=True)
    backend_cmd = [PYTHON_EXE, "backend/main.py"]
    backend_proc = start_process(backend_cmd, "BACKEND", "36")  # 36 = Cyan

    # Wait a moment to let backend start up
    time.sleep(2)

    # 2. Start Streamlit Frontend (port 8501)
    print("[Launcher] Starting Frontend (Streamlit on port 8501)...", flush=True)
    frontend_cmd = [STREAMLIT_EXE, "run", "frontend/form.py", "--server.port", "8501"]
    frontend_proc = start_process(frontend_cmd, "FRONTEND", "35")  # 35 = Magenta

    print("\n\033[32m[Launcher] Services are running!")
    print("  -> Backend:  http://localhost:8000")
    print("  -> Frontend: http://localhost:8501")
    print("Press Ctrl+C to stop both services.\033[0m\n", flush=True)

    # Keep main thread alive and monitor processes
    try:
        while True:
            time.sleep(1)
            # Check if any process has exited unexpectedly
            for p, name in processes:
                exit_code = p.poll()
                if exit_code is not None:
                    print(f"\033[31m[Launcher] Process {name} exited with code {exit_code}\033[0m", flush=True)
                    shutdown()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
