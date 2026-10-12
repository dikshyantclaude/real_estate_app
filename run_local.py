"""Serve the built app locally. Run: python run_local.py"""
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

root = Path(__file__).resolve().parent / "dist"
handler = partial(SimpleHTTPRequestHandler, directory=str(root))
server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)
print("District Atlas: http://localhost:8000")
print("Press Ctrl+C to stop.")
try:
    server.serve_forever()
except KeyboardInterrupt:
    server.server_close()
