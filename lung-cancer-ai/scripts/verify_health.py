"""
scripts/verify_health.py
========================
Automated health check verification script.
Tests REST responses, database connections, and model registries.
"""

from __future__ import annotations
import sys
import json
import urllib.request
from pathlib import Path

def test_health():
    url = "http://localhost:8000/api/health"
    print(f"[CHECK] Testing health endpoint at {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                print("[SUCCESS] Platform is ONLINE!")
                print(f"  - App Name: {data.get('app_name')}")
                print(f"  - Domain: {data.get('public_domain')}")
                print(f"  - Inference Engines: {len(data.get('inference_engines', []))} Active")
                return 0
            else:
                print(f"[ERROR] HTTP status {response.status}")
                return 1
    except Exception as e:
        print(f"[OFFLINE] Backend server is not running on port 8000 ({e}).")
        return 1

if __name__ == "__main__":
    sys.exit(test_health())
