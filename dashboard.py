import argparse
import json
import os
import random
import threading
import time
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


HTML_PAGE = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Lab Food/Drink Monitor</title>
    <style>
      :root {
        --bg: #f4f6fb;
        --card: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
        --new-bg: #fee2e2;
        --new-fg: #991b1b;
        --ack-bg: #dcfce7;
        --ack-fg: #166534;
        --btn: #1f2937;
        --btn-hover: #111827;
      }
      body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
      header {
        padding: 16px 20px;
        background: #111827;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }
      .header-actions {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .header-actions button {
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.14);
        color: #fff;
        font-size: 12px;
        padding: 6px 10px;
        cursor: pointer;
      }
      .header-actions button:hover {
        background: rgba(255, 255, 255, 0.24);
      }
      main { display: grid; grid-template-columns: 1.8fr 1fr; gap: 16px; padding: 16px; align-items: start; }
      .card { background: var(--card); border-radius: 12px; box-shadow: 0 10px 28px rgba(17,24,39,0.12); padding: 12px; }
      .settings-card { grid-column: 1 / -1; }
      .hidden { display: none !important; }
      .feed { width: 100%; border-radius: 8px; background: #000; }
      h2 { margin: 8px 0 12px; font-size: 16px; }
      .alerts-list { display: grid; gap: 10px; }
      .alert { border: 1px solid var(--border); border-radius: 10px; padding: 10px; background: #fff; }
      .alert-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
      .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
      .status-new { background: var(--new-bg); color: var(--new-fg); }
      .status-acknowledged { background: var(--ack-bg); color: var(--ack-fg); }
      .meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
      .det-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; margin-bottom: 10px; }
      .det-card { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #fafafa; }
      .det-card img { width: 100%; height: 82px; object-fit: cover; display: block; background: #d1d5db; }
      .det-card img.expandable { cursor: zoom-in; }
      .det-empty { width: 100%; height: 82px; background: #e5e7eb; display: flex; align-items: center; justify-content: center; color: #4b5563; font-size: 11px; }
      .det-label { font-size: 12px; padding: 6px; color: #1f2937; }
      .actions { display: flex; gap: 8px; }
      .actions button {
        border: 0;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 12px;
        color: #fff;
        background: var(--btn);
        cursor: pointer;
      }
      .actions button:hover { background: var(--btn-hover); }
      .actions button.delete { background: #991b1b; }
      .actions button.delete:hover { background: #7f1d1d; }
      .fold-section {
        margin-bottom: 10px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #f9fafb;
        padding: 8px 10px;
      }
      .fold-section summary {
        cursor: pointer;
        color: #374151;
        font-size: 13px;
        font-weight: 600;
      }
      .fold-section[open] summary { margin-bottom: 10px; }
      .empty-note {
        font-size: 12px;
        color: #6b7280;
        padding: 10px;
        border: 1px dashed var(--border);
        border-radius: 8px;
        background: #fafafa;
      }
      .settings-grid {
        display: grid;
        gap: 10px;
      }
      .settings-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
      }
      .settings-field {
        display: grid;
        gap: 4px;
        font-size: 12px;
        color: #374151;
      }
      .settings-field input {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 7px 8px;
        font-size: 13px;
        color: #111827;
      }
      .settings-toggle {
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 13px;
        color: #1f2937;
      }
      .settings-actions {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }
      .settings-actions button {
        border: 0;
        border-radius: 8px;
        padding: 7px 10px;
        font-size: 12px;
        color: #fff;
        background: #1f2937;
        cursor: pointer;
      }
      .settings-actions button.alt {
        background: #4b5563;
      }
      .settings-status {
        font-size: 12px;
        color: #374151;
      }
      .error { margin-top: 8px; color: #b91c1c; font-size: 12px; min-height: 16px; }
      .lightbox {
        position: fixed;
        inset: 0;
        background: rgba(15, 23, 42, 0.82);
        display: none;
        align-items: center;
        justify-content: center;
        z-index: 999;
        padding: 24px;
      }
      .lightbox.open { display: flex; }
      .lightbox-inner { position: relative; max-width: min(1000px, 96vw); max-height: 92vh; }
      .lightbox img { max-width: 100%; max-height: 92vh; border-radius: 10px; box-shadow: 0 20px 60px rgba(0,0,0,0.45); background: #111827; }
      .lightbox-close {
        position: absolute;
        top: -12px;
        right: -12px;
        border: 0;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #111827;
        color: #fff;
        font-size: 20px;
        line-height: 1;
        cursor: pointer;
      }
      @media (max-width: 600px) {
        .settings-row { grid-template-columns: 1fr; }
      }
      @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <header>
      <strong>Lab Food/Drink Monitor</strong>
      <div class=\"header-actions\">
        <button id=\"openSettingsBtn\" type=\"button\" aria-expanded=\"false\">Open Settings</button>
      </div>
    </header>
    <main>
      <section id=\"settingsCard\" class=\"card settings-card hidden\">
        <h2>Runtime Settings</h2>
        <form id=\"settingsForm\" class=\"settings-grid\">
          <label class=\"settings-toggle\">
            <input type=\"checkbox\" id=\"setCameraEnabled\" />
            Camera enabled
          </label>
          <label class=\"settings-toggle\">
            <input type=\"checkbox\" id=\"setDetectionEnabled\" />
            Detection enabled
          </label>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Confidence
              <input id=\"setConf\" type=\"number\" min=\"0.01\" max=\"1\" step=\"0.01\" />
            </label>
            <label class=\"settings-field\">IoU
              <input id=\"setIou\" type=\"number\" min=\"0.01\" max=\"1\" step=\"0.01\" />
            </label>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Persist frames
              <input id=\"setPersistFrames\" type=\"number\" min=\"1\" step=\"1\" />
            </label>
            <label class=\"settings-field\">Clear frames
              <input id=\"setClearFrames\" type=\"number\" min=\"1\" step=\"1\" />
            </label>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Cooldown (s)
              <input id=\"setCooldown\" type=\"number\" min=\"0\" step=\"0.1\" />
            </label>
            <label class=\"settings-field\">Stream FPS
              <input id=\"setStreamFps\" type=\"number\" min=\"1\" step=\"1\" />
            </label>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Display width (0=auto)
              <input id=\"setWidth\" type=\"number\" min=\"0\" step=\"1\" />
            </label>
            <label class=\"settings-field\">Display height (0=auto)
              <input id=\"setHeight\" type=\"number\" min=\"0\" step=\"1\" />
            </label>
          </div>
          <div class=\"settings-actions\">
            <button type=\"submit\">Apply settings</button>
            <button id=\"resetSettings\" type=\"button\" class=\"alt\">Reset defaults</button>
            <button id=\"reloadSettings\" type=\"button\" class=\"alt\">Reload</button>
            <span id=\"settingsStatus\" class=\"settings-status\"></span>
          </div>
        </form>
      </section>
      <section class=\"card\">
        <h2>Live Camera Feed</h2>
        <img class=\"feed\" src=\"/stream\" alt=\"Live camera feed\" />
      </section>
      <section class=\"card\">
        <h2>Recent Alerts</h2>
        <div id=\"alertsPanel\">
          <details id=\"ackSection\" class=\"fold-section\" style=\"display:none;\">
            <summary>Acknowledged (<span id=\"ackCount\">0</span>)</summary>
            <div id=\"ackAlerts\" class=\"alerts-list\"></div>
          </details>
          <details id=\"newSection\" class=\"fold-section\" open>
            <summary>New / Active (<span id=\"newCount\">0</span>)</summary>
            <div id=\"activeAlerts\" class=\"alerts-list\"></div>
          </details>
        </div>
        <div id=\"error\" class=\"error\"></div>
      </section>
    </main>
    <div id=\"lightbox\" class=\"lightbox\" aria-hidden=\"true\">
      <div class=\"lightbox-inner\">
        <button id=\"lightboxClose\" class=\"lightbox-close\" aria-label=\"Close expanded image\">×</button>
        <img id=\"lightboxImg\" alt=\"Expanded detection\" />
      </div>
    </div>
    <script>
      function escapeHtml(value) {
        return String(value)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('\"', '&quot;')
          .replaceAll(\"'\", '&#39;');
      }

      function renderDetection(det) {
        const confidence = Number(det.confidence || 0).toFixed(2);
        const className = escapeHtml(det.class_name || 'item');
        const snippet = det.snippet_file
          ? `<img class=\"expandable\" loading=\"lazy\" src=\"/snippets/${encodeURIComponent(det.snippet_file)}\" alt=\"${className} snippet\" />`
          : `<div class=\"det-empty\">No crop</div>`;
        return `<div class=\"det-card\">${snippet}<div class=\"det-label\">${className} (${confidence})</div></div>`;
      }

      function setupLightbox() {
        const lightbox = document.getElementById('lightbox');
        const lightboxImg = document.getElementById('lightboxImg');
        const closeBtn = document.getElementById('lightboxClose');
        const alertsPanel = document.getElementById('alertsPanel');

        function closeLightbox() {
          lightbox.classList.remove('open');
          lightbox.setAttribute('aria-hidden', 'true');
          lightboxImg.removeAttribute('src');
        }

        function openLightbox(src, altText) {
          lightboxImg.src = src;
          lightboxImg.alt = altText || 'Expanded detection';
          lightbox.classList.add('open');
          lightbox.setAttribute('aria-hidden', 'false');
        }

        alertsPanel.addEventListener('click', (event) => {
          const img = event.target.closest('img.expandable');
          if (!img) {
            return;
          }
          openLightbox(img.src, img.alt);
        });
        closeBtn.addEventListener('click', closeLightbox);
        lightbox.addEventListener('click', (event) => {
          if (event.target === lightbox) {
            closeLightbox();
          }
        });
        document.addEventListener('keydown', (event) => {
          if (event.key === 'Escape' && lightbox.classList.contains('open')) {
            closeLightbox();
          }
        });
      }

      async function manageAlert(alertId, action) {
        const res = await fetch('/alerts/manage', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ alert_id: alertId, action }),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
      }

      function setSettingsStatus(message, isError = false) {
        const statusEl = document.getElementById('settingsStatus');
        statusEl.textContent = message || '';
        statusEl.style.color = isError ? '#b91c1c' : '#374151';
      }

      function setupSettingsPanel() {
        const opener = document.getElementById('openSettingsBtn');
        const card = document.getElementById('settingsCard');
        function sync() {
          const isOpen = !card.classList.contains('hidden');
          opener.textContent = isOpen ? 'Hide Settings' : 'Open Settings';
          opener.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
        }
        opener.addEventListener('click', () => {
          card.classList.toggle('hidden');
          sync();
        });
        sync();
      }

      function applySettingsToForm(settings) {
        document.getElementById('setCameraEnabled').checked = Boolean(settings.camera_enabled);
        document.getElementById('setDetectionEnabled').checked = Boolean(settings.detection_enabled);
        document.getElementById('setConf').value = Number(settings.conf || 0.25).toFixed(2);
        document.getElementById('setIou').value = Number(settings.iou || 0.45).toFixed(2);
        document.getElementById('setPersistFrames').value = Number(settings.persist_frames || 3);
        document.getElementById('setClearFrames').value = Number(settings.clear_frames || 10);
        document.getElementById('setCooldown').value = Number(settings.cooldown || 10).toFixed(1);
        document.getElementById('setStreamFps').value = Number(settings.stream_fps || 10).toFixed(0);
        document.getElementById('setWidth').value = Number(settings.width || 0).toFixed(0);
        document.getElementById('setHeight').value = Number(settings.height || 0).toFixed(0);
      }

      function readNumberField(id, label) {
        const value = Number(document.getElementById(id).value);
        if (!Number.isFinite(value)) {
          throw new Error(`Invalid ${label}`);
        }
        return value;
      }

      function collectSettingsPayload() {
        return {
          camera_enabled: document.getElementById('setCameraEnabled').checked,
          detection_enabled: document.getElementById('setDetectionEnabled').checked,
          conf: readNumberField('setConf', 'confidence'),
          iou: readNumberField('setIou', 'IoU'),
          persist_frames: Math.round(readNumberField('setPersistFrames', 'persist frames')),
          clear_frames: Math.round(readNumberField('setClearFrames', 'clear frames')),
          cooldown: readNumberField('setCooldown', 'cooldown'),
          stream_fps: readNumberField('setStreamFps', 'stream FPS'),
          width: Math.round(readNumberField('setWidth', 'display width')),
          height: Math.round(readNumberField('setHeight', 'display height')),
        };
      }

      async function loadSettings(showStatus = false) {
        const res = await fetch('/settings');
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const settings = await res.json();
        applySettingsToForm(settings);
        if (showStatus) {
          setSettingsStatus(`Loaded (${settings.updated_at || 'now'})`);
        }
      }

      async function saveSettings(event) {
        event.preventDefault();
        try {
          const payload = collectSettingsPayload();
          const res = await fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          if (!res.ok) {
            const text = await res.text();
            throw new Error(text || `HTTP ${res.status}`);
          }
          const data = await res.json();
          if (data.settings) {
            applySettingsToForm(data.settings);
          }
          setSettingsStatus('Settings applied');
        } catch (err) {
          setSettingsStatus(err.message || 'Could not save settings', true);
        }
      }

      async function resetSettingsToDefaults() {
        try {
          const res = await fetch('/settings/reset', {
            method: 'POST',
          });
          if (!res.ok) {
            const text = await res.text();
            throw new Error(text || `HTTP ${res.status}`);
          }
          const data = await res.json();
          if (data.settings) {
            applySettingsToForm(data.settings);
          }
          setSettingsStatus('Settings reset to defaults');
        } catch (err) {
          setSettingsStatus(err.message || 'Could not reset settings', true);
        }
      }

      function setupSettingsForm() {
        const form = document.getElementById('settingsForm');
        const reloadBtn = document.getElementById('reloadSettings');
        const resetBtn = document.getElementById('resetSettings');
        form.addEventListener('submit', saveSettings);
        reloadBtn.addEventListener('click', async () => {
          try {
            await loadSettings(true);
          } catch (err) {
            setSettingsStatus('Could not reload settings', true);
          }
        });
        resetBtn.addEventListener('click', resetSettingsToDefaults);
        loadSettings().catch(() => {
          setSettingsStatus('Could not load settings', true);
        });
      }

      function fingerprintAlerts(alerts) {
        return alerts.map((alert) => {
          const detections = Array.isArray(alert.detections) ? alert.detections : [];
          const detectionKey = detections
            .map((det) => `${det.class_name || ''}:${det.confidence || ''}:${det.snippet_file || ''}`)
            .join('|');
          return `${alert.id || ''}:${alert.status || ''}:${alert.timestamp || ''}:${detectionKey}`;
        }).join('||');
      }

      let lastAlertsFingerprint = '';
      let refreshInFlight = false;

      function renderAlertCard(alert, errorEl) {
        const detections = Array.isArray(alert.detections) ? alert.detections : [];
        const status = alert.status || 'new';
        const badgeClass = status === 'acknowledged' ? 'status-acknowledged' : 'status-new';
        const actionBtn = status === 'acknowledged'
          ? `<button data-action=\"reopen\">Reopen</button>`
          : `<button data-action=\"acknowledge\">Acknowledge</button>`;
        const controls = alert.id
          ? `<div class=\"actions\">${actionBtn}<button class=\"delete\" data-action=\"delete\">Delete</button></div>`
          : '';
        const item = document.createElement('article');
        item.className = 'alert';
        item.innerHTML =
          `<div class=\"alert-head\"><span class=\"badge ${badgeClass}\">${escapeHtml(status)}</span>` +
          `<span>${escapeHtml(alert.timestamp || 'unknown time')}</span></div>` +
          `<div class=\"meta\">${detections.length} detection(s)</div>` +
          `<div class=\"det-grid\">${detections.map(renderDetection).join('')}</div>` +
          controls;
        item.querySelectorAll('button[data-action]').forEach(btn => {
          btn.addEventListener('click', async () => {
            try {
              btn.disabled = true;
              await manageAlert(alert.id, btn.dataset.action);
              await refreshAlerts();
            } catch (err) {
              errorEl.textContent = 'Could not update alert. Try again.';
            } finally {
              btn.disabled = false;
            }
          });
        });
        return item;
      }

      async function refreshAlerts() {
        if (refreshInFlight) {
          return;
        }
        refreshInFlight = true;
        const errorEl = document.getElementById('error');
        errorEl.textContent = '';
        try {
          const res = await fetch('/alerts?limit=20');
          const data = await res.json();
          const ordered = data.slice().reverse();
          const nextFingerprint = fingerprintAlerts(ordered);
          if (nextFingerprint === lastAlertsFingerprint) {
            return;
          }
          lastAlertsFingerprint = nextFingerprint;
          const activeList = document.getElementById('activeAlerts');
          const ackList = document.getElementById('ackAlerts');
          const ackCount = document.getElementById('ackCount');
          const newCount = document.getElementById('newCount');
          const ackSection = document.getElementById('ackSection');
          activeList.innerHTML = '';
          ackList.innerHTML = '';
          const activeAlerts = ordered.filter((alert) => (alert.status || 'new') !== 'acknowledged');
          const acknowledgedAlerts = ordered.filter((alert) => (alert.status || 'new') === 'acknowledged');
          if (activeAlerts.length === 0) {
            activeList.innerHTML = '<div class=\"empty-note\">No active alerts right now.</div>';
          } else {
            activeAlerts.forEach((alert) => {
              activeList.appendChild(renderAlertCard(alert, errorEl));
            });
          }
          newCount.textContent = String(activeAlerts.length);
          acknowledgedAlerts.forEach((alert) => {
            ackList.appendChild(renderAlertCard(alert, errorEl));
          });
          ackCount.textContent = String(acknowledgedAlerts.length);
          if (acknowledgedAlerts.length === 0) {
            ackSection.open = false;
            ackSection.style.display = 'none';
          } else {
            ackSection.style.display = '';
          }
        } catch (err) {
          errorEl.textContent = 'Could not load alerts right now.';
        } finally {
          refreshInFlight = false;
        }
      }
      setupLightbox();
      setupSettingsPanel();
      setupSettingsForm();
      refreshAlerts();
      setInterval(refreshAlerts, 2000);
    </script>
  </body>
</html>
"""


FOOD_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "bottle",
    "cup",
    "bowl",
}


def get_allowed_class_ids(model, allowed_names: set[str]) -> list[int]:
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, list):
        items = enumerate(names)
    else:
        return []
    allowed: list[int] = []
    for cls_id, name in items:
        if name and name.strip().lower() in allowed_names:
            allowed.append(int(cls_id))
    return sorted(allowed)


def read_alerts(log_path: str | None) -> list[dict]:
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        return []
    return []


def write_alerts(log_path: str | None, alerts: list[dict]) -> None:
    if not log_path:
        return
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(alerts, handle, indent=2)


def ensure_alert_metadata(alerts: list[dict]) -> bool:
    changed = False
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if not alert.get("id"):
            alert["id"] = uuid4().hex[:12]
            changed = True
        if not alert.get("status"):
            alert["status"] = "new"
            changed = True
    return changed


def append_alert(log_path: str | None, alert: dict) -> None:
    alerts = read_alerts(log_path)
    ensure_alert_metadata(alerts)
    alerts.append(alert)
    write_alerts(log_path, alerts)


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    token = token.strip("_")
    return token or "item"


def add_detection_snippets(frame, detections: list[dict], snippet_dir: str | None, alert_id: str):
    if not snippet_dir:
        return detections
    os.makedirs(snippet_dir, exist_ok=True)
    height, width = frame.shape[:2]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        left = max(0, min(width - 1, int(x1)))
        top = max(0, min(height - 1, int(y1)))
        right = max(left + 1, min(width, int(x2)))
        bottom = max(top + 1, min(height, int(y2)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        class_token = _safe_token(det.get("class_name", "item"))
        snippet_file = f"{alert_id}_{idx}_{class_token}.jpg"
        snippet_path = os.path.join(snippet_dir, snippet_file)
        if cv2.imwrite(snippet_path, crop):
            det["snippet_file"] = snippet_file
    return detections


def create_alert(frame, detections: list[dict], snippet_dir: str | None) -> dict:
    alert_id = uuid4().hex[:12]
    return {
        "id": alert_id,
        "status": "new",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "detections": add_detection_snippets(frame, detections, snippet_dir, alert_id),
    }


def detections_from_result(result, allowed_names: set[str] | None = None) -> list[dict]:
    detections: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    names = result.names
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy[idx])
        conf = float(result.boxes.conf[idx])
        cls_id = int(result.boxes.cls[idx])
        class_name = names.get(cls_id, str(cls_id))
        normalized = class_name.strip().lower()
        if allowed_names and normalized not in allowed_names:
            continue
        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            }
        )
    return detections


def draw_detections(frame, detections):
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 255),
            2,
        )
    return annotated


def make_placeholder_svg(width: int, height: int, label: str) -> str:
    safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" "
        f"viewBox=\"0 0 {width} {height}\">\n"
        "  <rect width=\"100%\" height=\"100%\" fill=\"#121824\" />\n"
        "  <text x=\"50%\" y=\"50%\" text-anchor=\"middle\" dominant-baseline=\"middle\" "
        "font-family=\"Arial, sans-serif\" font-size=\"32\" fill=\"#f3f4f6\">"
        f"{safe_label}</text>\n"
        "</svg>\n"
    )


def make_random_alerts(limit: int, frame_width: int, frame_height: int) -> list[dict]:
    alerts: list[dict] = []
    class_names = sorted(FOOD_CLASS_NAMES)
    for idx in range(max(1, limit)):
        det_count = random.randint(1, 3)
        detections = []
        for _ in range(det_count):
            class_name = random.choice(class_names)
            x1 = random.randint(0, max(0, frame_width - 60))
            y1 = random.randint(0, max(0, frame_height - 60))
            x2 = random.randint(x1 + 20, min(frame_width, x1 + 200))
            y2 = random.randint(y1 + 20, min(frame_height, y1 + 200))
            detections.append(
                {
                    "class_id": -1,
                    "class_name": class_name,
                    "confidence": round(random.uniform(0.5, 0.99), 2),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
                }
            )
        alert_time = datetime.now() - timedelta(seconds=idx * 3)
        alerts.append(
            {
                "id": uuid4().hex[:12],
                "status": random.choice(["new", "acknowledged"]),
                "timestamp": alert_time.isoformat(timespec="seconds"),
                "frame_size": {"width": frame_width, "height": frame_height},
                "detections": detections,
            }
        )
    return alerts


def make_status_frame(width: int, height: int, label: str):
    if cv2 is None or np is None:
        return None
    safe_width = max(320, int(width or 640))
    safe_height = max(180, int(height or 360))
    frame = np.zeros((safe_height, safe_width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        label,
        (24, safe_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
    )
    return frame


def parse_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Invalid boolean for '{field_name}'")


def clamp_float(value, field_name: str, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid number for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def clamp_int(value, field_name: str, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def settings_snapshot(config) -> dict:
    with config.settings_lock:
        return {
            "camera_enabled": bool(config.camera_enabled),
            "detection_enabled": bool(config.detection_enabled),
            "conf": float(config.conf),
            "iou": float(config.iou),
            "persist_frames": int(config.persist_frames),
            "cooldown": float(config.cooldown),
            "clear_frames": int(config.clear_frames),
            "stream_fps": float(config.stream_fps),
            "width": int(config.width),
            "height": int(config.height),
            "updated_at": config.settings_updated_at,
            "test_mode": bool(config.test_mode),
        }


def default_settings_snapshot(config) -> dict:
    with config.settings_lock:
        defaults = dict(config.default_settings)
    defaults["test_mode"] = bool(config.test_mode)
    return defaults


def update_runtime_settings(config, payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Settings payload must be a JSON object")
    with config.settings_lock:
        if "camera_enabled" in payload:
            config.camera_enabled = parse_bool(payload["camera_enabled"], "camera_enabled")
        if "detection_enabled" in payload:
            config.detection_enabled = parse_bool(payload["detection_enabled"], "detection_enabled")
        if "conf" in payload:
            config.conf = clamp_float(payload["conf"], "conf", 0.01, 1.0)
        if "iou" in payload:
            config.iou = clamp_float(payload["iou"], "iou", 0.01, 1.0)
        if "persist_frames" in payload:
            config.persist_frames = clamp_int(payload["persist_frames"], "persist_frames", 1, 120)
        if "cooldown" in payload:
            config.cooldown = clamp_float(payload["cooldown"], "cooldown", 0.0, 3600.0)
        if "clear_frames" in payload:
            config.clear_frames = clamp_int(payload["clear_frames"], "clear_frames", 1, 600)
        if "stream_fps" in payload:
            config.stream_fps = clamp_float(payload["stream_fps"], "stream_fps", 1.0, 60.0)
        if "width" in payload:
            config.width = clamp_int(payload["width"], "width", 0, 3840)
        if "height" in payload:
            config.height = clamp_int(payload["height"], "height", 0, 2160)
        config.settings_updated_at = datetime.now().isoformat(timespec="seconds")
    return settings_snapshot(config)


def reset_runtime_settings(config) -> dict:
    defaults = default_settings_snapshot(config)
    defaults.pop("test_mode", None)
    return update_runtime_settings(config, defaults)


class DashboardConfig:
    def __init__(
        self,
        model,
        alert_log,
        width,
        height,
        stream_fps,
        conf,
        iou,
        persist_frames,
        cooldown,
        clear_frames,
        snippet_dir,
        test_mode,
    ):
        self.model = model
        self.alert_log = alert_log
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.conf = conf
        self.iou = iou
        self.persist_frames = persist_frames
        self.cooldown = cooldown
        self.clear_frames = clear_frames
        self.camera_enabled = True
        self.detection_enabled = True
        self.settings_updated_at = datetime.now().isoformat(timespec="seconds")
        self.snippet_dir = snippet_dir
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.default_settings = {
            "camera_enabled": True,
            "detection_enabled": True,
            "conf": float(conf),
            "iou": float(iou),
            "persist_frames": int(persist_frames),
            "cooldown": float(cooldown),
            "clear_frames": int(clear_frames),
            "stream_fps": float(stream_fps),
            "width": int(width),
            "height": int(height),
        }
        self.stop = False
        self.consecutive = 0
        self.clear_count = 0
        self.armed = True
        self.last_alert_ts = 0.0
        self.test_mode = test_mode


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FoodDrinkDashboard/0.2"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/alerts":
            params = parse_qs(parsed.query)
            try:
                limit = int(params.get("limit", [50])[0])
            except ValueError:
                limit = 50
            limit = max(0, min(500, limit))
            self._send_alerts(limit)
            return
        if parsed.path == "/settings":
            self._send_settings()
            return
        if parsed.path.startswith("/snippets/"):
            self._send_snippet(parsed.path.removeprefix("/snippets/"))
            return
        if parsed.path == "/stream":
            self._stream_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/alerts/manage":
            self._manage_alert()
            return
        if parsed.path == "/settings/reset":
            self._reset_settings()
            return
        if parsed.path == "/settings":
            self._update_settings()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, payload: dict | list, status=HTTPStatus.OK):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0:
            raise ValueError("Missing request body")
        raw = self.rfile.read(min(content_length, 1_000_000))
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("Request body must be valid JSON") from None

    def _send_alerts(self, limit):
        config: DashboardConfig = self.server.config
        if config.test_mode:
            frame_width = config.width or 640
            frame_height = config.height or 360
            alerts = make_random_alerts(limit, frame_width, frame_height)
            self._send_json(alerts, HTTPStatus.OK)
            return
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            if ensure_alert_metadata(alerts):
                write_alerts(config.alert_log, alerts)
        if limit > 0:
            alerts = alerts[-limit:]
        self._send_json(alerts, HTTPStatus.OK)

    def _manage_alert(self):
        config: DashboardConfig = self.server.config
        if config.test_mode:
            self.send_error(HTTPStatus.BAD_REQUEST, "Alert management disabled in --test mode")
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        alert_id = str(payload.get("alert_id", "")).strip()
        action = str(payload.get("action", "")).strip().lower()
        if not alert_id or action not in {"acknowledge", "reopen", "delete"}:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid alert_id or action")
            return

        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            ensure_alert_metadata(alerts)
            target_index = -1
            for idx, alert in enumerate(alerts):
                if isinstance(alert, dict) and alert.get("id") == alert_id:
                    target_index = idx
                    break
            if target_index < 0:
                self.send_error(HTTPStatus.NOT_FOUND, "Alert not found")
                return
            if action == "delete":
                alerts.pop(target_index)
            else:
                status = "acknowledged" if action == "acknowledge" else "new"
                alerts[target_index]["status"] = status
                alerts[target_index]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            write_alerts(config.alert_log, alerts)
        self._send_json({"ok": True, "action": action, "alert_id": alert_id}, HTTPStatus.OK)

    def _send_settings(self):
        config: DashboardConfig = self.server.config
        self._send_json(settings_snapshot(config), HTTPStatus.OK)

    def _update_settings(self):
        config: DashboardConfig = self.server.config
        try:
            payload = self._read_json_body()
            updated = update_runtime_settings(config, payload)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _reset_settings(self):
        config: DashboardConfig = self.server.config
        updated = reset_runtime_settings(config)
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _send_snippet(self, encoded_name: str):
        config: DashboardConfig = self.server.config
        if not config.snippet_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Snippet storage is disabled")
            return
        requested_name = unquote(encoded_name)
        if requested_name != os.path.basename(requested_name):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid snippet path")
            return
        snippet_root = os.path.abspath(config.snippet_dir)
        snippet_path = os.path.abspath(os.path.join(snippet_root, requested_name))
        if not snippet_path.startswith(f"{snippet_root}{os.sep}"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid snippet path")
            return
        if not os.path.exists(snippet_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Snippet not found")
            return
        with open(snippet_path, "rb") as handle:
            body = handle.read()
        content_type = "image/jpeg"
        lower = snippet_path.lower()
        if lower.endswith(".png"):
            content_type = "image/png"
        elif lower.endswith(".webp"):
            content_type = "image/webp"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self):
        config: DashboardConfig = self.server.config
        if config.test_mode:
            width = config.width or 640
            height = config.height or 360
            svg = make_placeholder_svg(width, height, "CAMERA FEED")
            body = svg.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if cv2 is None:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "OpenCV not available")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with config.frame_lock:
                    frame = None if config.latest_frame is None else config.latest_frame.copy()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                with config.settings_lock:
                    stream_fps = float(config.stream_fps)
                delay = 1.0 / max(1.0, stream_fps)
                payload = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


def camera_worker(config: DashboardConfig, cam_index: int):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for live camera mode.")
    cap = None

    allowed_ids = get_allowed_class_ids(config.model, FOOD_CLASS_NAMES)

    try:
        while not config.stop:
            with config.settings_lock:
                camera_enabled = bool(config.camera_enabled)
                detection_enabled = bool(config.detection_enabled)
                conf = float(config.conf)
                iou = float(config.iou)
                persist_frames = int(config.persist_frames)
                cooldown = float(config.cooldown)
                clear_frames = int(config.clear_frames)
                out_width = int(config.width)
                out_height = int(config.height)

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True
                paused = make_status_frame(out_width or 640, out_height or 360, "Camera is OFF")
                if paused is not None:
                    with config.frame_lock:
                        config.latest_frame = paused
                time.sleep(0.15)
                continue

            if cap is None:
                cap = cv2.VideoCapture(cam_index)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    unavailable = make_status_frame(
                        out_width or 640,
                        out_height or 360,
                        "Camera unavailable",
                    )
                    if unavailable is not None:
                        with config.frame_lock:
                            config.latest_frame = unavailable
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                time.sleep(0.1)
                continue

            detections = []
            if detection_enabled:
                try:
                    results = config.model.predict(
                        frame,
                        verbose=False,
                        conf=conf,
                        iou=iou,
                        classes=allowed_ids if allowed_ids else None,
                    )
                except TypeError:
                    results = config.model.predict(
                        frame,
                        verbose=False,
                        conf=conf,
                        iou=iou,
                    )
                result = results[0]
                detections = detections_from_result(result, allowed_names=FOOD_CLASS_NAMES)

            if detection_enabled and detections:
                config.consecutive += 1
                config.clear_count = 0
            elif detection_enabled:
                config.consecutive = 0
                config.clear_count += 1
                if config.clear_count >= max(1, clear_frames):
                    config.armed = True
            else:
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True

            now = time.time()
            if (
                detections
                and config.consecutive >= max(1, persist_frames)
                and config.armed
                and (now - config.last_alert_ts) >= max(0.0, cooldown)
            ):
                alert = create_alert(frame, detections, snippet_dir=config.snippet_dir)
                with config.alert_lock:
                    append_alert(config.alert_log, alert)
                config.last_alert_ts = now
                config.armed = False

            annotated = draw_detections(frame, detections) if detection_enabled else frame.copy()
            if not detection_enabled:
                cv2.putText(
                    annotated,
                    "Detection is OFF",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            if out_width or out_height:
                annotated = cv2.resize(
                    annotated,
                    (
                        out_width or annotated.shape[1],
                        out_height or annotated.shape[0],
                    ),
                )

            with config.frame_lock:
                config.latest_frame = annotated
    finally:
        if cap is not None:
            cap.release()


def main():
    parser = argparse.ArgumentParser(description="Camera dashboard with live alerts")
    parser.add_argument("--test", action="store_true", help="Run with synthetic feed/alerts")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model weights")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--alert-log", default="alerts.json", help="Alert JSON path")
    parser.add_argument(
        "--snippet-dir",
        default="snippets",
        help="Directory where per-detection crop images are stored (set empty to disable)",
    )
    parser.add_argument("--width", type=int, default=0, help="Resize width (0 = original)")
    parser.add_argument("--height", type=int, default=0, help="Resize height (0 = original)")
    parser.add_argument("--fps", type=int, default=10, help="Stream FPS")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=3,
        help="Require this many consecutive frames with detections to alert",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="Minimum seconds between alerts",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=10,
        help="Require this many consecutive clear frames before re-arming alerts",
    )
    args = parser.parse_args()

    if not args.test:
        if YOLO is None:
            raise RuntimeError("ultralytics is required unless --test is used.")
        if cv2 is None:
            raise RuntimeError("opencv-python is required unless --test is used.")

    model = None if args.test else YOLO(args.model)

    config = DashboardConfig(
        model=model,
        alert_log=args.alert_log,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        conf=args.conf,
        iou=args.iou,
        persist_frames=args.persist_frames,
        cooldown=args.cooldown,
        clear_frames=args.clear_frames,
        snippet_dir=args.snippet_dir or None,
        test_mode=args.test,
    )

    if not args.test:
        worker = threading.Thread(target=camera_worker, args=(config, args.cam), daemon=True)
        worker.start()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.config = config
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        config.stop = True


if __name__ == "__main__":
    main()
