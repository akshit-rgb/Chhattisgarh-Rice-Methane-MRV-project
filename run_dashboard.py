#!/usr/bin/env python3
"""
Dashboard Launcher for Chhattisgarh Rice Methane & Groundwater MRV
-------------------------------------------------------------------
Automatically launches the dashboard and opens it in your default browser.

Usage:
  python run_dashboard.py
"""

import os
import sys
import subprocess
import webbrowser
import time
import http.server
import socketserver

def find_streamlit():
    # 1. Check current python environment
    try:
        import streamlit
        return [sys.executable, "-m", "streamlit", "run", "app.py"]
    except ImportError:
        pass

    # 2. Check miniconda or other conda environments
    conda_paths = [
        r"C:\Users\akshi\miniconda3\Scripts\streamlit.exe",
        r"C:\ProgramData\miniconda3\Scripts\streamlit.exe",
        r"C:\Users\akshi\anaconda3\Scripts\streamlit.exe"
    ]
    for cp in conda_paths:
        if os.path.exists(cp):
            return [cp, "run", "app.py"]

    # 3. Check system PATH
    try:
        res = subprocess.run(["where", "streamlit"], capture_output=True, text=True, shell=True)
        if res.returncode == 0 and res.stdout.strip():
            first_path = res.stdout.strip().splitlines()[0]
            return [first_path, "run", "app.py"]
    except Exception:
        pass

    return None

def launch_server():
    print("=" * 70)
    print("🚀 CHHATTISGARH RICE METHANE & GROUNDWATER MRV DASHBOARD LAUNCHER")
    print("=" * 70)

    st_cmd = find_streamlit()
    if st_cmd:
        print(f"\n[INFO] Found Streamlit: {' '.join(st_cmd)}")
        print("[INFO] Launching interactive Python Streamlit Dashboard...")
        print("[INFO] Your browser will open automatically at http://localhost:8501\n")
        subprocess.run(st_cmd)
    else:
        port = 8000
        print(f"\n[INFO] Launching high-performance Web GIS Dashboard on port {port}...")
        url = f"http://localhost:{port}/index.html"
        print(f"[INFO] Opening browser at {url} ...\n")
        webbrowser.open(url)
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"Serving at {url} (Press Ctrl+C to stop)...")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.")

if __name__ == "__main__":
    launch_server()
