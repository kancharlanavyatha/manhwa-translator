import urllib.request
import json

try:
    with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=5) as resp:
        targets = json.loads(resp.read().decode("utf-8"))
        for t in targets:
            print(t.get("type"), t.get("title"), t.get("url"))
except Exception as e:
    print("Error:", e)
