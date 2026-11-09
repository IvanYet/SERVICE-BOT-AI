"""Loopback-only browser UI for live Receptionist vision tuning."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse


DEFAULTS = {
    "seat_confidence": 0.75,
    "seat_min_width": 70,
    "seat_min_height": 90,
    "seat_min_area": 7000,
    "seat_aspect_min": 0.45,
    "seat_aspect_max": 1.8,
    "seat_confirm_s": 1.0,
    "seat_hold_s": 1.5,
    "seat_keep_s": 4.0,
    "seat_overlap_min": 0.20,
    "seat_fallback_overlap": 0.55,
    "reid_recover_threshold": 0.60,
    "reid_track_threshold": 0.60,
    "reid_margin": 0.06,
    "reid_min_hits": 3,
}

RANGES = {
    "seat_confidence": (0.45, 0.99, 0.01),
    "seat_min_width": (20, 400, 1),
    "seat_min_height": (20, 400, 1),
    "seat_min_area": (500, 100000, 100),
    "seat_aspect_min": (0.1, 1.5, 0.01),
    "seat_aspect_max": (0.5, 3.0, 0.01),
    "seat_confirm_s": (0.2, 5.0, 0.1),
    "seat_hold_s": (0.0, 5.0, 0.1),
    "seat_keep_s": (1.0, 15.0, 0.5),
    "seat_overlap_min": (0.05, 0.8, 0.01),
    "seat_fallback_overlap": (0.3, 0.95, 0.01),
    "reid_recover_threshold": (0.3, 0.99, 0.01),
    "reid_track_threshold": (0.3, 0.99, 0.01),
    "reid_margin": (0.0, 0.3, 0.01),
    "reid_min_hits": (2, 10, 1),
}

LABELS = {
    "seat_confidence": "Seat detector confidence",
    "seat_min_width": "Minimum seat width (px)",
    "seat_min_height": "Minimum seat height (px)",
    "seat_min_area": "Minimum seat area (px²)",
    "seat_aspect_min": "Minimum width / height",
    "seat_aspect_max": "Maximum width / height",
    "seat_confirm_s": "Evidence time to confirm (s)",
    "seat_hold_s": "Keep occupied through gaps (s)",
    "seat_keep_s": "Retain seat track (s)",
    "seat_overlap_min": "Minimum person / seat overlap",
    "seat_fallback_overlap": "Pose fallback overlap",
    "reid_recover_threshold": "ReID recovery similarity",
    "reid_track_threshold": "Bound track similarity",
    "reid_margin": "ReID best-match margin",
    "reid_min_hits": "Repeated ReID matches",
}


def _page():
    fields = "".join(
        f'<label>{LABELS[key]} <input type="number" name="{key}" '
        f'min="{low}" max="{high}" step="{step}" required></label>'
        for key, (low, high, step) in RANGES.items()
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Receptionist Vision Tuning</title>
<style>
body{{font:16px system-ui,sans-serif;margin:20px auto;max-width:1100px;padding:0 16px;background:#111827;color:#e5e7eb}}
h1{{font-size:1.5rem}} .layout{{display:grid;grid-template-columns:minmax(300px,1fr) minmax(360px,1.5fr);gap:20px}}
form,.panel{{background:#1f2937;padding:16px;border-radius:12px}} label{{display:grid;grid-template-columns:1fr 110px;gap:10px;align-items:center;margin:10px 0;font-size:.9rem}}
input{{width:100%;box-sizing:border-box;padding:7px;border-radius:5px;border:1px solid #4b5563;background:#111827;color:white}}
button{{padding:10px 16px;border:0;border-radius:7px;background:#22c55e;color:#052e16;font-weight:700;cursor:pointer}}
img{{width:100%;background:#030712;border-radius:8px;min-height:240px;object-fit:contain}} pre{{white-space:pre-wrap;max-height:230px;overflow:auto}}
.hint{{color:#9ca3af;font-size:.9rem}} #message{{min-height:1.4em;color:#86efac}}
@media(max-width:800px){{.layout{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Receptionist Vision Tuning</h1>
<p class="hint">Loopback only. Changes apply immediately and are saved to the project tuning JSON.</p>
<div class="layout"><form id="form">{fields}<button>Apply and save</button><div id="message"></div></form>
<section class="panel"><h2>Live camera and detections</h2><img src="/preview.mjpg" alt="Waiting for camera preview">
<h3>Current detector output</h3><pre id="status">Waiting for vision data…</pre></section></div>
<script>
const form=document.querySelector('#form'), message=document.querySelector('#message');
async function refresh(){{
 const r=await fetch('/api/config'); const d=await r.json();
 for(const [k,v] of Object.entries(d.values)) if(form.elements[k]) form.elements[k].value=v;
 document.querySelector('#status').textContent=JSON.stringify(d.status||{{}},null,2);
}}
form.addEventListener('submit',async e=>{{e.preventDefault();const values={{}};
 for(const el of form.elements) if(el.name) values[el.name]=Number(el.value);
 const r=await fetch('/api/config',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(values)}});
 const d=await r.json(); message.textContent=r.ok?'Applied and saved.':(d.error||'Update failed'); if(r.ok) refresh();
}});
refresh(); setInterval(async()=>{{try{{const r=await fetch('/api/config');const d=await r.json();document.querySelector('#status').textContent=JSON.stringify(d.status||{{}},null,2)}}catch(e){{}}}},1000);
</script></body></html>"""


class VisionTuningServer:
    def __init__(self, config_path, on_update, get_status, logger=None, port=8765):
        self.config_path = Path(config_path)
        self.on_update = on_update
        self.get_status = get_status
        self.logger = logger
        self._lock = threading.RLock()
        self._preview = None
        self.values = dict(DEFAULTS)
        self._load()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                if owner.logger:
                    owner.logger.debug("Vision tuning HTTP: " + fmt % args)

            def _send(self, code, body, content_type="application/json"):
                payload = body.encode("utf-8") if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    return self._send(200, _page(), "text/html; charset=utf-8")
                if path == "/api/config":
                    with owner._lock:
                        body = {"values": dict(owner.values), "status": owner.get_status()}
                    return self._send(200, json.dumps(body))
                if path == "/preview.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            with owner._lock:
                                frame = owner._preview
                            if frame is not None:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                            threading.Event().wait(0.15)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return self._send(404, json.dumps({"error": "not found"}))

            def do_POST(self):
                if urlparse(self.path).path != "/api/config":
                    return self._send(404, json.dumps({"error": "not found"}))
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 4096:
                        raise ValueError("invalid request size")
                    update = json.loads(self.rfile.read(size))
                    values = owner._validate(update)
                    with owner._lock:
                        merged = dict(owner.values)
                        merged.update(values)
                        if merged["seat_aspect_min"] >= merged["seat_aspect_max"]:
                            raise ValueError("Minimum aspect ratio must be below maximum")
                        owner.config_path.parent.mkdir(parents=True, exist_ok=True)
                        temp_path = owner.config_path.with_suffix(".tmp")
                        temp_path.write_text(json.dumps(merged, indent=2) + "\n")
                        temp_path.replace(owner.config_path)
                        owner.values = merged
                        owner.on_update(dict(merged))
                    return self._send(200, json.dumps({"values": merged}))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    return self._send(400, json.dumps({"error": str(exc)}))
                except OSError as exc:
                    return self._send(500, json.dumps({"error": f"Could not save settings: {exc}"}))

        try:
            self.httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        except OSError as exc:
            raise RuntimeError(f"Could not start local vision tuning server on port {port}: {exc}") from exc
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def _load(self):
        try:
            raw = json.loads(self.config_path.read_text())
            self.values.update(self._validate(raw))
            if self.values["seat_aspect_min"] >= self.values["seat_aspect_max"]:
                raise ValueError("invalid aspect ratio bounds")
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            if self.logger:
                self.logger.warning(f"Ignoring invalid saved vision tuning config: {exc}")

    @staticmethod
    def _validate(update):
        if not isinstance(update, dict):
            raise ValueError("request must be a JSON object")
        unknown = set(update) - set(DEFAULTS)
        if unknown:
            raise ValueError("unknown setting(s): " + ", ".join(sorted(unknown)))
        values = {}
        for key, value in update.items():
            low, high, _step = RANGES[key]
            number = float(value)
            if not low <= number <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            if key in ("seat_min_width", "seat_min_height", "seat_min_area", "reid_min_hits"):
                if not number.is_integer():
                    raise ValueError(f"{key} must be an integer")
                number = int(number)
            values[key] = number
        return values

    def update_preview(self, frame):
        import cv2
        height, width = frame.shape[:2]
        if width > 960:
            frame = cv2.resize(frame, (960, int(height * 960 / width)))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ok:
            with self._lock:
                self._preview = encoded.tobytes()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
