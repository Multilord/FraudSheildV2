"""
Start all four MCP servers as separate processes.
Run: python start_mcp.py
Stop with Ctrl+C -- all servers will terminate together.
"""

import subprocess
import sys
import os
import signal

SERVERS = [
    ("Scoring  (8001)", "mcp/scoring_server.py"),
    ("Case     (8002)", "mcp/case_server.py"),
    ("Network  (8003)", "mcp/network_server.py"),
    ("Insights (8004)", "mcp/insights_server.py"),
]

processes = []


def shutdown(signum=None, frame=None):
    print("\n[start_mcp] Stopping all MCP servers...")
    for proc in processes:
        proc.terminate()
    for proc in processes:
        proc.wait()
    print("[start_mcp] All servers stopped.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    backend_dir = os.path.dirname(os.path.abspath(__file__))

    print("[start_mcp] Starting MCP servers...\n")
    for name, script in SERVERS:
        path = os.path.join(backend_dir, script)
        proc = subprocess.Popen(
            [sys.executable, path],
            cwd=backend_dir,
        )
        processes.append(proc)
        print(f"  [OK] {name}  (pid {proc.pid})")

    print("\n[start_mcp] All servers running. Press Ctrl+C to stop.\n")

    # Wait for all processes
    for proc in processes:
        proc.wait()
