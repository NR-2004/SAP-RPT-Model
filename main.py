"""
Entry point.
  python main.py                   start the API (opens http://127.0.0.1:8000/docs)
  python main.py evaluate [rows]   accuracy check on hold-out POs
  python main.py config-check      check .env, service key and training CSV
"""

import json
import os
import socket
import sys
import threading
import webbrowser

from app.api import app
from app.config import validate_config
from app.evaluation import evaluate


def run_server():
    problems = validate_config()
    if problems:
        sys.exit("\n  Missing required inputs:\n  - " + "\n  - ".join(problems) + "\n")

    import uvicorn

    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    docs = f"http://127.0.0.1:{port}/docs"
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(f"\n  Port {port} is already in use - the server is probably already running "
                     f"in another terminal.\n  Use that one ({docs}), stop it with CTRL+C, or set PORT=8001 in .env.\n")
    print(f"\n  Open in browser: {docs}\n  Stop with CTRL+C\n", flush=True)
    threading.Timer(2, webbrowser.open, [docs]).start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "config-check":
        problems = validate_config()
        print(json.dumps({"ready": not problems, "missing": problems}, indent=2))
    elif command == "evaluate":
        print(json.dumps(evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 256), indent=2))
    else:
        run_server()
