#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
import zlib
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import onnxruntime as ort
import serial  
from scipy.signal import butter, lfilter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ==============================
# ====== User-tunable vars =====
# ==============================

# --- Web server ---
HOST = "0.0.0.0"
PORT = 8000
HISTORY_MAX_POINTS = 400
UI_THEME = "dark"        # "dark" | "light"

# --- Sleep classes  ---
CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

# --- Sampling / segmentation ---
FS = 100
PACKET_SEC = 3.0
PACKET_SAMPLES = int(PACKET_SEC * FS)   # 300
SEGMENT_SEC = 30.0
SEGMENT_SAMPLES = int(SEGMENT_SEC * FS) # 3000
SEQUENCE_LEN = 4                        # history length in segments for the forecaster

# --- UART settings ---
SERIAL_PORT = "/dev/ttyAMA0"  
BAUD = 115_200
TIMEOUT_S = 0.5

# --- Wire protocol ---
MAG_DATA = b'EEG3'
MAG_ACK  = b'ACK3'
HDR_DATA_LEN = 4 + 2 + 2  # 'EEG3' + seq + len
CRC_LEN = 4
PAYLOAD_SIG_BYTES = 3 * PACKET_SAMPLES * 4  # 3 channels × 300 float32 = 3600
PAYLOAD_BYTES = PAYLOAD_SIG_BYTES + 1       # +1 label byte (uint8) = 3601

# --- DSP filter ---
LOWCUT = 0.5
HIGHCUT = 49.9
FILTER_ORDER = 5

# --- Model ---
ONNX_MODEL_PATH = "/home/raspberry_sleepModel/neurolingua_forecast.onnx"

# --- GPIO / Relay control ---
# Map each stage to its dedicated BCM pin 
STAGE_TO_GPIO = {
    "Wake": 17,  # BCM 17
    "N1":   27,  # BCM 27
    "N2":   22,  # BCM 22
    "N3":   23,  # BCM 23
    "REM":  24,  # BCM 24
}
RELAY_ACTIVE_HIGH = False  
USE_TRUE_FOR_RELAY = False 


# ============================
# ====== Helper routines =====
# ============================
def crc32(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF

def read_exact(ser, n: int, deadline_s: float) -> Optional[bytes]:
    buf = bytearray()
    start = time.monotonic()
    while len(buf) < n:
        if time.monotonic() - start > deadline_s:
            return None
        chunk = ser.read(n - len(buf))
        if not chunk:
            time.sleep(0.001)
            continue
        buf += chunk
    return bytes(buf)

def read_one_data_frame(ser) -> Optional[Tuple[int, bytes]]:
    
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            return None
        window += b
        if len(window) > len(MAG_DATA):
            window.pop(0)
        if bytes(window) == MAG_DATA:
            # read seq + len
            rest = read_exact(ser, 2 + 2, 1.0)
            if rest is None:
                return None
            seq, payload_len = struct.unpack('<HH', rest)
            if payload_len != PAYLOAD_BYTES:
                _ = read_exact(ser, payload_len + CRC_LEN, 1.0)  # resync
                return None
            payload = read_exact(ser, payload_len, 1.0)
            crc_rx_b = read_exact(ser, CRC_LEN, 1.0)
            if payload is None or crc_rx_b is None:
                return None
            crc_rx, = struct.unpack('<I', crc_rx_b)
            if crc32(payload) != crc_rx:
                return None
            return seq, payload

def send_ack(ser, seq: int):
    body = struct.pack('<H', seq & 0xFFFF)
    frame = MAG_ACK + body + struct.pack('<I', crc32(body))
    ser.write(frame)
    ser.flush()

def preprocess(sig: np.ndarray, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, sig)

def segment_to_windows_like_dataset(segment_1d_np: np.ndarray):
    sig_f = preprocess(segment_1d_np)
    starts = range(0, SEGMENT_SAMPLES - PACKET_SAMPLES + 1, int(2.25*FS))  
    return np.stack([sig_f[s:s+PACKET_SAMPLES] for s in starts], axis=0).astype(np.float32)

def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ==================================
# ====== Forecaster (ONNX RT)  =====
# ==================================
class AlignedSlidingForecasterONNX:

    def __init__(self, onnx_path: str):
        self.S = SEQUENCE_LEN
        self.total_len = self.S * SEGMENT_SAMPLES
        self.hist_fpz = np.empty((0,), dtype=np.float32)
        self.hist_pz  = np.empty((0,), dtype=np.float32)
        self.hist_eog = np.empty((0,), dtype=np.float32)
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def ingest_and_predict(self, pkt_bytes: bytes) -> Optional[Dict[str, Any]]:
        pkt = np.frombuffer(pkt_bytes, dtype=np.float32).reshape(3, PACKET_SAMPLES)
        fpz, pz, eog = pkt

        self.hist_fpz = np.concatenate([self.hist_fpz, fpz])[-self.total_len:]
        self.hist_pz  = np.concatenate([self.hist_pz,  pz])[-self.total_len:]
        self.hist_eog = np.concatenate([self.hist_eog, eog])[-self.total_len:]

        if self.hist_fpz.size < self.total_len:
            return None  # warm-up

        def split(hist):
            return np.stack([hist[i*SEGMENT_SAMPLES:(i+1)*SEGMENT_SAMPLES] for i in range(self.S)], axis=0)

        fpz_stack, pz_stack, eog_stack = map(split, (self.hist_fpz, self.hist_pz, self.hist_eog))
        fpz_seq = np.stack([segment_to_windows_like_dataset(s) for s in fpz_stack], axis=0)
        pz_seq  = np.stack([segment_to_windows_like_dataset(s) for s in pz_stack],  axis=0)
        eog_seq = np.stack([segment_to_windows_like_dataset(s) for s in eog_stack], axis=0)

        x_fpz = fpz_seq[np.newaxis, ...].astype(np.float32)
        x_pz  = pz_seq [np.newaxis, ...].astype(np.float32)
        x_eog = eog_seq[np.newaxis, ...].astype(np.float32)

        logits_curr, logits_next = self.sess.run(
            ["logits_current", "logits_next"],
            {"eeg_fpz_cz": x_fpz, "eeg_pz_oz": x_pz, "eog": x_eog}
        )
        probs = softmax_np(logits_next[0])
        pred_idx = int(np.argmax(probs))
        return {"pred_next_idx": pred_idx, "pred_next_lbl": CLASS_NAMES[pred_idx], "probs_next": probs.tolist()}


# ==================================
# ====== Web app & broadcasting ====
# ==================================
app = FastAPI(title="Live Hypnogram")

# Shared state
CLIENTS: "set[WebSocket]" = set()
HISTORY: "deque[Dict[str, Any]]" = deque(maxlen=HISTORY_MAX_POINTS)
METRICS = {"total": 0, "correct": 0, "running_acc": 0.0}
LOOP: Optional[asyncio.AbstractEventLoop] = None  # filled in lifespan

# ===== GPIO setup (with safe fallback) =====
class _GPIOMock:
    BCM = 'BCM'
    OUT = 'OUT'
    _state: Dict[int, int] = {}
    @classmethod
    def setmode(cls, *_): pass
    @classmethod
    def setwarnings(cls, *_): pass
    @classmethod
    def setup(cls, pin, *_): cls._state[pin] = 0
    @classmethod
    def output(cls, pin, val):
        cls._state[pin] = int(val)
        print(f"[MockGPIO] pin {pin} <- {val}")
    @classmethod
    def cleanup(cls):
        for p in list(cls._state.keys()):
            cls._state[p] = 0
        print("[MockGPIO] cleanup")

try:
    import RPi.GPIO as GPIO  # type: ignore
    USING_GPIO = True
except Exception as e:
    print(f"[WARN] RPi.GPIO not available ({e}). Using mock GPIO.")
    GPIO = _GPIOMock  # type: ignore
    USING_GPIO = False

_ON  = 1 if RELAY_ACTIVE_HIGH else 0
_OFF = 0 if RELAY_ACTIVE_HIGH else 1
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for _stg, _pin in STAGE_TO_GPIO.items():
    GPIO.setup(_pin, GPIO.OUT)
    GPIO.output(_pin, _OFF)

def _relay_on(pin: int):  GPIO.output(pin, _ON)
def _relay_off(pin: int): GPIO.output(pin, _OFF)
def _all_relays_off():
    for _stg, _pin in STAGE_TO_GPIO.items():
        _relay_off(_pin)

# Relay selection state
selected_stage: Optional[str] = None      # one of CLASS_NAMES or None ("No action")
relay_engaged: bool = False               # whether selected relay is currently ON
last_seen_stage: Optional[str] = None     # tracks last stage label to detect transitions

# ===== broadcast helpers =====
async def _send_to_all(msg: Dict[str, Any]):
    dead: List[WebSocket] = []
    for ws in list(CLIENTS):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)

def broadcast_from_thread(msg: Dict[str, Any]):
    """Thread-safe: schedule an async send to all clients on the FastAPI loop."""
    if LOOP is None:
        return
    fut = asyncio.run_coroutine_threadsafe(_send_to_all(msg), LOOP)
    try:
        fut.result(timeout=2.0)
    except Exception:
        pass

def _stage_idx_to_name(idx: int) -> str:
    return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)

def maybe_toggle_relay_threadsafe(*, pred_idx: int, true_idx: int):
    """
    Called from the serial worker THREAD. Uses thread-safe broadcast helper.
    Follows prediction by default; set USE_TRUE_FOR_RELAY=True to follow ground-truth.
    """
    global relay_engaged, last_seen_stage, selected_stage

    # No action: make sure OFF and reset state
    if selected_stage is None:
        if relay_engaged:
            _all_relays_off()
            relay_engaged = False
            broadcast_from_thread({"type": "relay", "on": False})
        last_seen_stage = None
        return

    current_name = _stage_idx_to_name(true_idx if USE_TRUE_FOR_RELAY else pred_idx)

    just_entered = (current_name == selected_stage) and (last_seen_stage != selected_stage)
    just_left    = (current_name != selected_stage) and (last_seen_stage == selected_stage)

    if just_entered and not relay_engaged:
        _all_relays_off()
        _relay_on(STAGE_TO_GPIO[selected_stage])
        relay_engaged = True
        broadcast_from_thread({"type": "relay", "on": True})

    elif just_left and relay_engaged:
        _relay_off(STAGE_TO_GPIO[selected_stage])
        relay_engaged = False
        broadcast_from_thread({"type": "relay", "on": False})

    last_seen_stage = current_name

async def handle_selection_async(stage: Optional[str]):
    """
    Called from WS handler (async context). stage is one of CLASS_NAMES or None ("No action").
    """
    global selected_stage, relay_engaged, last_seen_stage
    if stage is not None and stage not in CLASS_NAMES:
        stage = None

    selected_stage = stage
    relay_was_on = relay_engaged
    relay_engaged = False
    last_seen_stage = None
    _all_relays_off()

    # Notify clients
    await _send_to_all({"type": "selection", "selected_stage": selected_stage})
    if relay_was_on:
        await _send_to_all({"type": "relay", "on": False})


@app.get("/health")
async def health():
    return JSONResponse({
        "ok": True,
        "clients": len(CLIENTS),
        "points": len(HISTORY),
        "using_gpio": USING_GPIO,
        "selected_stage": selected_stage,
        "relay_on": relay_engaged
    })

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML_DARK if UI_THEME.lower() == "dark" else INDEX_HTML_LIGHT)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    # send snapshot on connect
    await ws.send_json({
        "type": "snapshot",
        "class_names": CLASS_NAMES,
        "history": list(HISTORY),
        "metrics": METRICS,
        "server_time": datetime.utcnow().isoformat() + "Z",
        "selected_stage": selected_stage,
        "relay_on": relay_engaged
    })
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue
            if data.get("type") == "select_stage":
                await handle_selection_async(data.get("stage"))
    except WebSocketDisconnect:
        CLIENTS.discard(ws)
    except Exception:
        CLIENTS.discard(ws)


# ==================================
# ====== Serial worker thread  =====
# ==================================
def serial_worker():
    """Reads UART frames forever, runs model, broadcasts points, and toggles relays."""
    print(f"[RX] Opening {SERIAL_PORT} @ {BAUD} ...")
    forecaster = AlignedSlidingForecasterONNX(ONNX_MODEL_PATH)
    pending_pred: Optional[Dict[str, Any]] = None
    correct = 0
    total = 0

    with serial.Serial(SERIAL_PORT, BAUD, timeout=TIMEOUT_S) as ser:
        print(f"[RX] Listening on {SERIAL_PORT} @ {BAUD} ...")
        while True:
            frame = read_one_data_frame(ser)
            if frame is None:
                continue
            seq, payload = frame
            sig_bytes = payload[:PAYLOAD_SIG_BYTES]
            lbl_byte  = payload[PAYLOAD_SIG_BYTES:]
            y_curr = int(struct.unpack('<B', lbl_byte)[0])

            # ACK ASAP
            send_ack(ser, seq)

            out = forecaster.ingest_and_predict(sig_bytes)

            # If we have a pending prediction for NEXT, evaluate it now with THIS packet's label.
            if pending_pred is not None:
                y_true_next = y_curr
                y_pred_next = int(pending_pred["pred_next_idx"])
                total += 1
                correct += int(y_true_next == y_pred_next)
                acc = 100.0 * correct / max(1, total)

                point = {
                    "type": "point",
                    "t": time.time(),
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "seq": seq,
                    "pred_idx": y_pred_next,
                    "true_idx": y_true_next,
                    "pred_lbl": CLASS_NAMES[y_pred_next],
                    "true_lbl": CLASS_NAMES[y_true_next],
                    "running_acc": acc,
                }
                HISTORY.append(point)
                METRICS["total"] = total
                METRICS["correct"] = correct
                METRICS["running_acc"] = acc

                ok = (y_true_next == y_pred_next)
                print(f"[RX] seq={seq:5d}  pred={CLASS_NAMES[y_pred_next]:>3s}  "
                      f"true(next)={CLASS_NAMES[y_true_next]:>3s}  "
                      f"step_acc={'✓' if ok else '✗'}  running_acc={acc:.2f}%")

                # Broadcast to browsers (thread-safe)
                broadcast_from_thread(point)

                # Relay control (thread-safe)
                maybe_toggle_relay_threadsafe(pred_idx=y_pred_next, true_idx=y_true_next)
            else:
                # During warm-up
                broadcast_from_thread({
                    "type": "status",
                    "phase": "warming_up",
                    "seq": seq,
                    "curr_lbl": CLASS_NAMES[y_curr]
                })

            # Update pending for NEXT step (if model produced one)
            pending_pred = out  # can be None during warm-up


# ==============================
# ====== Lifespan startup  =====
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global LOOP
    LOOP = asyncio.get_running_loop()
    threading.Thread(target=serial_worker, daemon=True).start()
    yield

app.router.lifespan_context = lifespan


# =====================
# ====== HTML UI ======
# =====================
INDEX_HTML_DARK = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Live Hypnogram + Relay Control</title>
  <style>
    :root { --bg:#0b1020; --fg:#e7ecff; --muted:#9fb0ff33; --card:#121936; }
    *{box-sizing:border-box}
    body{margin:0;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg)}
    header{padding:10px 12px;border-bottom:1px solid var(--muted);position:sticky;top:0;background:rgba(11,16,32,.9);backdrop-filter:blur(6px)}
    .row{display:flex;gap:16px;flex-wrap:wrap;padding:16px}
    .card{flex:1 1 360px;background:var(--card);border:1px solid var(--muted);border-radius:16px;padding:12px}
    h1{font-size:18px;margin:0}
    .legend{display:flex;gap:12px;align-items:center;font-size:13px;margin-bottom:6px}
    .dot{width:10px;height:10px;border-radius:50%}
    .chart-wrap{position:relative;width:100%;height:260px;overflow:hidden}
    .chart-wrap>canvas{position:absolute;inset:0;width:100% !important;height:100% !important}
    .row-ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px}
    select,button{background:#0f1a3a;color:#e7ecff;border:1px solid #3a4a7a;border-radius:10px;padding:8px 10px;font-size:14px}
    .pill{display:inline-block;padding:3px 8px;border-radius:999px;border:1px solid #3a4a7a;background:#0f1a3a}
    .ok{border-color:#6ee7b7aa;background:#093428}
    .bad{border-color:#ef4444aa;background:#3a0f0f}
    @media (max-width:520px){
      .row{padding:8px;gap:10px}
      .card{padding:10px;border-radius:14px}
      .chart-wrap{height:150px}
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <header>
    <h1>Live Hypnogram <span id="status" class="pill">connecting…</span></h1>
  </header>

  <div class="row">
    <div class="card">
      <div class="legend">
        <div class="dot" style="background:#4fc3f7"></div> Predicted
      </div>
      <div class="chart-wrap"><canvas id="chart"></canvas></div>
      <div style="display:flex;gap:12px;align-items:baseline;margin-top:6px">
        <div style="opacity:.8">Points: <span id="npts">0</span></div>
        <div style="margin-left:auto">Relay: <span id="relay" class="pill">OFF</span></div>
      </div>

      <div class="row-ctl">
        <label for="stageSel">Action on stage:</label>
        <select id="stageSel">
          <option value="none">No action</option>
          <option value="Wake">Wake</option>
          <option value="N1">N1</option>
          <option value="N2">N2</option>
          <option value="N3">N3</option>
          <option value="REM">REM</option>
        </select>
        <button id="applyBtn">Apply</button>
        <span id="selEcho" class="pill">Selected: none</span>
      </div>
    </div>

    <div class="card" id="card-stats">
      <h3 style="margin:0 0 8px 0;font-size:14px;">Last prediction</h3>
      <div id="lastpred" style="font-family:ui-monospace,monospace">–</div>
    </div>
  </div>

  <script>
  (() => {
    const chartEl = document.getElementById('chart');
    const accEl = document.getElementById('acc');
    const nptsEl = document.getElementById('npts');
    const statusEl = document.getElementById('status');
    const lastpredEl = document.getElementById('lastpred');
    const relayEl = document.getElementById('relay');
    const stageSel = document.getElementById('stageSel');
    const applyBtn = document.getElementById('applyBtn');
    const selEcho = document.getElementById('selEcho');

    if (window.innerWidth <= 520) {
      const wrap = chartEl.parentElement;
      if (wrap && getComputedStyle(wrap).height === '0px') wrap.style.height = '150px';
    }

    const xs = [], ysPred = [], ysTrue = [];
    let classNames = ["Wake","N1","N2","N3","REM"];
    let pointCount = 0;

    function yTickLabel(v){ const i=Math.round(v); return classNames[i] ?? i; }
    function yToLabel(i){ return classNames[i] ?? i; }

    function setStatus(txt, ok=true){
      statusEl.textContent = txt;
      statusEl.classList.remove('ok','bad');
      statusEl.classList.add(ok ? 'ok' : 'bad');
    }
    function setRelay(on){
      relayEl.textContent = on ? 'ON' : 'OFF';
      relayEl.classList.remove('ok','bad');
      relayEl.classList.add(on ? 'ok' : 'bad');
    }

    const ctx = chartEl.getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: xs,
        datasets: [
          { label: 'Pred', data: ysPred, stepped: true, borderWidth: 2, pointRadius: 0, tension: 0, borderColor: '#4fc3f7' },
        ]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 2, bottom: 2 } },
        scales: {
          x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 6 }, grid: { display: false } },
          y: {
            min: 0, max: 4, reverse: true, offset: false,
            ticks: { callback: yTickLabel, stepSize: 1, maxTicksLimit: 5 },
            grid: { display: false }, border: { display: false }, title: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => `${c.dataset.label}: ${yToLabel(c.parsed.y)}` } }
        },
        elements: { line: { fill: false } }
      }
    });

    function pushPoint(p){
      xs.push(pointCount++);
      ysPred.push(p.pred_idx);
      ysTrue.push(p.true_idx);
      chart.update('none');
      nptsEl.textContent = xs.length;
      if (lastpredEl) lastpredEl.textContent = ` pred=${yToLabel(p.pred_idx)}`;
    }

    function loadSnapshot(snap){
      classNames = snap.class_names || classNames;
      const hist = snap.history || [];
      xs.length = 0; ysPred.length = 0; ysTrue.length = 0; pointCount = 0;
      for (const p of hist) pushPoint(p);
      setStatus('live', true);
      if (typeof snap.relay_on === 'boolean') setRelay(!!snap.relay_on);
      if (snap.selected_stage === null) { selEcho.textContent = 'Selected: none'; stageSel.value = 'none'; }
      else { selEcho.textContent = 'Selected: ' + snap.selected_stage; stageSel.value = snap.selected_stage; }
    }

    const wsProto = (location.protocol === 'https:') ? 'wss' : 'ws';
    const ws = new WebSocket(`${wsProto}://${location.host}/ws`);
    ws.onopen = () => setStatus('connected', true);
    ws.onclose = () => setStatus('disconnected', false);
    ws.onerror = () => setStatus('error', false);
    ws.onmessage = ev => {
      const msg = JSON.parse(ev.data || "{}");
      if (!msg || !msg.type) return;
      if (msg.type === 'snapshot') return loadSnapshot(msg);
      if (msg.type === 'point') {
          pushPoint(msg);
              if (msg.phase === 'warming_up') {
                setStatus('warming up…', true);
              } else {
                // fallback: whatever phase the server sends
                setStatus('live', true);
              }
          return;
        }

      if (msg.type === 'status') {
          if (msg.phase === 'warming_up') {
            setStatus('warming up…', true);
          } else {
            // fallback: whatever phase the server sends
            setStatus('live', true);
          }
        }

      if (msg.type === 'relay') return setRelay(!!msg.on);
      if (msg.type === 'selection') {
        const sel = msg.selected_stage;
        selEcho.textContent = 'Selected: ' + (sel === null ? 'none' : sel);
        stageSel.value = sel === null ? 'none' : sel;
      }
    };


    function sendSelection(){
      const v = stageSel.value;
      const stage = (v === 'none') ? null : v;
      ws.send(JSON.stringify({type: 'select_stage', stage}));
    }
    applyBtn.addEventListener('click', sendSelection);
  })();
  </script>
</body>
</html>
"""

# Light theme = same JS, different palette
INDEX_HTML_LIGHT = INDEX_HTML_DARK.replace(
    ":root { --bg:#0b1020; --fg:#e7ecff; --muted:#9fb0ff33; --card:#121936; }",
    ":root { --bg:#f6f8ff; --fg:#0b1020; --muted:#0b102022; --card:#ffffff; }"
).replace(
    "background:rgba(11,16,32,.9)",
    "background:rgba(255,255,255,.9)"
)

# =====================
# ====== Runner  ======
# =====================
def start_server():
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="info")
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(server.serve())
        print(f"Uvicorn scheduled on ws://{HOST}:{PORT} (running loop detected)")
        return server
    else:
        loop.run_until_complete(server.serve())

if __name__ == "__main__":
    start_server()
