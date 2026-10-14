import sys
import subprocess
import time
import json
import urllib.request
import os

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ext_path = r"D:\webtoon translator\extension"
profile_dir = r"D:\webtoon translator\.chrome_debug_profile"

try:
    with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=5) as resp:
        targets = json.loads(resp.read().decode("utf-8"))
        print(f"Connected to existing Chrome! Found {len(targets)} targets:")
        for t in targets:
            print(f" - [{t.get('type')}] {t.get('title')} -> {t.get('url')}")
            print(f"   ws: {t.get('webSocketDebuggerUrl')}")
except Exception as e:
    print("Could not connect to existing Chrome:", e)
