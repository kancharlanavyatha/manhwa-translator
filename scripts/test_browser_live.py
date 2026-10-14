import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import websockets

sys.stdout.reconfigure(encoding='utf-8')

profile_dir = r"D:\webtoon translator\.chrome_debug_profile"
if os.path.exists(profile_dir):
    try:
        shutil.rmtree(profile_dir)
    except Exception:
        pass
os.makedirs(profile_dir, exist_ok=True)

chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ext_path = r"D:\webtoon translator\extension"
url = "http://127.0.0.1:8000/demo/index.html"

cmd = [
    chrome_exe,
    "--remote-debugging-port=9222",
    f"--load-extension={ext_path}",
    f"--user-data-dir={profile_dir}",
    "--no-first-run",
    "--no-default-browser-check",
    url,
]

print("Launching Chrome...")
proc = subprocess.Popen(cmd)

async def run_cdp():
    await asyncio.sleep(3)
    target = None
    for attempt in range(10):
        try:
            with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=5) as resp:
                targets = json.loads(resp.read().decode("utf-8"))
                for t in targets:
                    if "demo/index.html" in t.get("url", ""):
                        target = t
                        break
                if target:
                    break
        except Exception:
            pass
        await asyncio.sleep(1)

    if not target:
        print("Could not find demo page target in Chrome.")
        return

    ws_url = target.get("webSocketDebuggerUrl")
    print("Connecting to CDP target WebSocket:", ws_url)

    async with websockets.connect(ws_url) as ws:
        # Enable Log, Console, Runtime, Page
        await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        await ws.send(json.dumps({"id": 2, "method": "Log.enable"}))
        await ws.send(json.dumps({"id": 3, "method": "Page.enable"}))

        print("Listening for browser events for 10 seconds...")
        start_t = time.time()
        while time.time() - start_t < 10:
            try:
                msg_str = await asyncio.wait_for(ws.recv(), timeout=2.0)
                msg = json.loads(msg_str)
                method = msg.get("method", "")
                params = msg.get("params", {})

                if method == "Runtime.consoleAPICalled":
                    args = params.get("args", [])
                    txt = " ".join([str(a.get("value", a.get("description", ""))) for a in args])
                    print(f"[BROWSER CONSOLE {params.get('type')}]: {txt}")
                elif method == "Runtime.exceptionThrown":
                    exc = params.get("exceptionDetails", {})
                    print(f"[BROWSER EXCEPTION]: {exc.get('text')} - {exc.get('exception', {}).get('description')}")
                elif method == "Log.entryAdded":
                    entry = params.get("entry", {})
                    print(f"[BROWSER LOG {entry.get('level')}]: {entry.get('text')}")
            except asyncio.TimeoutError:
                pass

        # Now evaluate page state
        eval_script = """
        ({
            title: document.title,
            pillText: document.getElementById('wt-widget-count')?.textContent || 'no widget',
            panel1Src: document.getElementById('panel-img-1')?.src?.substring(0, 50),
            panel2Src: document.getElementById('panel-img-2')?.src?.substring(0, 50),
            extInjected: !!window.__webtoonTranslatorInjected
        })
        """
        await ws.send(json.dumps({
            "id": 100,
            "method": "Runtime.evaluate",
            "params": {"expression": eval_script, "returnByValue": True}
        }))

        resp_str = await ws.recv()
        resp = json.loads(resp_str)
        print("[PAGE STATE]:", json.dumps(resp.get("result", {}).get("result", {}).get("value"), indent=2))

asyncio.run(run_cdp())
proc.terminate()
print("Test completed.")
