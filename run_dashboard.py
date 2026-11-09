#!/usr/bin/env python3
"""
Chhattisgarh Rice Methane & Groundwater MRV Web Dashboard Launcher
-------------------------------------------------------------------
Launches the standalone interactive Web GIS Dashboard (index.html)
in your default browser and starts a lightweight local HTTP server.

Usage:
  python run_dashboard.py
"""

import os
import sys
import webbrowser
import http.server
import socketserver
import socket


def find_free_port(preferred_port=8000, max_tries=10):
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return preferred_port


def main():
    print("=" * 76)
    print("🌾 CHHATTISGARH RICE METHANE, YIELD & GROUNDWATER MRV DASHBOARD")
    print("=" * 76)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    index_file = os.path.join(root_dir, "index.html")
    if not os.path.exists(index_file):
        print(f"[ERROR] index.html not found in {root_dir}")
        sys.exit(1)

    port = find_free_port(8000)
    url = f"http://localhost:{port}/index.html"

    print(f"\n[INFO] Starting local Web GIS HTTP server...")
    print(f"[INFO] Serving workspace directory: {root_dir}")
    print(f"[INFO] Dashboard URL: {url}")
    print(f"[INFO] Opening default web browser...\n")

    webbrowser.open(url)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress noisy logging for static assets
            pass

    try:
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            print("=" * 76)
            print(f"✨ Web GIS Dashboard live at: {url}")
            print("✨ Features: 146 Blocks Choropleth, 10m Field Parcels, Policy Sliders")
            print("✨ Press Ctrl+C in this terminal to shut down the server.")
            print("=" * 76 + "\n")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down Web GIS Dashboard server. Goodbye!")


if __name__ == "__main__":
    main()
