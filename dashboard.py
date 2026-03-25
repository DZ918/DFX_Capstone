"""Browser dashboard for live detection, alert review, and accepted-sample training."""

import argparse
from collections import deque
import json
import math
import os
import random
import shutil
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


# The frontend is small enough to keep inline: the Python server exposes JSON and
# MJPEG endpoints, and this page consumes them directly without a separate build step.
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
        --accepted-bg: #dcfce7;
        --accepted-fg: #166534;
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
      .status-accepted { background: var(--accepted-bg); color: var(--accepted-fg); }
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
      .actions button.reject { background: #991b1b; }
      .actions button.reject:hover { background: #7f1d1d; }
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
      .settings-field input,
      .settings-field select {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 7px 8px;
        font-size: 13px;
        color: #111827;
        background: #fff;
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
          <div class=\"settings-row\">
            <label class=\"settings-field\">Camera zone
              <select id=\"setCameraZone\">
                <option value=\"Zone A\">Zone A</option>
                <option value=\"Zone B\">Zone B</option>
                <option value=\"Zone C\">Zone C</option>
                <option value=\"Zone D\">Zone D</option>
                <option value=\"Zone E\">Zone E</option>
                <option value=\"Zone F\">Zone F</option>
                <option value=\"Zone G\">Zone G</option>
                <option value=\"Zone H\">Zone H</option>
                <option value=\"Zone I\">Zone I</option>
              </select>
            </label>
          </div>
          <div class=\"settings-actions\">
            <button type=\"submit\">Apply settings</button>
            <button id=\"resetSettings\" type=\"button\" class=\"alt\">Reset defaults</button>
            <button id=\"reloadSettings\" type=\"button\" class=\"alt\">Reload</button>
            <button id=\"trainAccepted\" type=\"button\">Train on accepted</button>
            <span id=\"settingsStatus\" class=\"settings-status\"></span>
            <span id=\"trainStatus\" class=\"settings-status\"></span>
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
          <details id=\"acceptedSection\" class=\"fold-section\" style=\"display:none;\">
            <summary>Accepted (<span id=\"acceptedCount\">0</span>)</summary>
            <div id=\"acceptedAlerts\" class=\"alerts-list\"></div>
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
      // Escape user-controlled strings before placing them into HTML templates.
      function escapeHtml(value) {
        return String(value)
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('\"', '&quot;')
          .replaceAll(\"'\", '&#39;');
      }

      // Each detection renders as a small card so alerts can show multiple crops at once.
      function renderDetection(det) {
        const confidence = Number(det.confidence || 0).toFixed(2);
        const className = escapeHtml(det.class_name || 'item');
        const snippet = det.snippet_file
          ? `<img class=\"expandable\" loading=\"lazy\" src=\"/snippets/${encodeURIComponent(det.snippet_file)}\" alt=\"${className} snippet\" />`
          : `<div class=\"det-empty\">No crop</div>`;
        return `<div class=\"det-card\">${snippet}<div class=\"det-label\">${className} (${confidence})</div></div>`;
      }

      // The lightbox keeps the alert list compact while still allowing full-size inspection.
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

      function setTrainStatus(message, isError = false) {
        const statusEl = document.getElementById('trainStatus');
        statusEl.textContent = message || '';
        statusEl.style.color = isError ? '#b91c1c' : '#374151';
      }

      async function refreshTrainStatus() {
        try {
          const res = await fetch('/train/status');
          if (!res.ok) {
            return;
          }
          const data = await res.json();
          if (data.running) {
            setTrainStatus('Training running...');
            return;
          }
          if (data.last_error) {
            setTrainStatus(`Train failed: ${data.last_error}`, true);
            return;
          }
          if (data.last_completed_at) {
            setTrainStatus(`Last train: ${data.last_completed_at}`);
            return;
          }
          setTrainStatus('');
        } catch (err) {
          setTrainStatus('Could not read train status', true);
        }
      }

      async function triggerTraining() {
        try {
          const res = await fetch('/train/accepted', {
            method: 'POST',
          });
          if (!res.ok) {
            const text = await res.text();
            throw new Error(text || `HTTP ${res.status}`);
          }
          setTrainStatus('Training started...');
          refreshTrainStatus();
        } catch (err) {
          setTrainStatus(err.message || 'Could not start training', true);
        }
      }

      // The settings panel is hidden by default because operators mostly watch the live feed.
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

      // Mirror server-side runtime settings into the form so edits always start from real state.
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
        document.getElementById('setCameraZone').value = settings.camera_zone || 'Zone A';
      }

      function readNumberField(id, label) {
        const value = Number(document.getElementById(id).value);
        if (!Number.isFinite(value)) {
          throw new Error(`Invalid ${label}`);
        }
        return value;
      }

      // Build the API payload from the current form values, with client-side validation first.
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
          camera_zone: document.getElementById('setCameraZone').value,
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

      // Wire the settings form once, then let periodic polling keep the training status fresh.
      function setupSettingsForm() {
        const form = document.getElementById('settingsForm');
        const reloadBtn = document.getElementById('reloadSettings');
        const resetBtn = document.getElementById('resetSettings');
        const trainBtn = document.getElementById('trainAccepted');
        form.addEventListener('submit', saveSettings);
        reloadBtn.addEventListener('click', async () => {
          try {
            await loadSettings(true);
          } catch (err) {
            setSettingsStatus('Could not reload settings', true);
          }
        });
        resetBtn.addEventListener('click', resetSettingsToDefaults);
        trainBtn.addEventListener('click', triggerTraining);
        loadSettings().catch(() => {
          setSettingsStatus('Could not load settings', true);
        });
        refreshTrainStatus();
      }

      // A cheap fingerprint prevents unnecessary DOM rebuilds when the alert list is unchanged.
      function fingerprintAlerts(alerts) {
        return alerts.map((alert) => {
          const detections = Array.isArray(alert.detections) ? alert.detections : [];
          const detectionKey = detections
            .map((det) => `${det.class_name || ''}:${det.confidence || ''}:${det.snippet_file || ''}:${det.zone || ''}`)
            .join('|');
          return `${alert.id || ''}:${alert.status || ''}:${alert.timestamp || ''}:${alert.zone || ''}:${detectionKey}`;
        }).join('||');
      }

      let lastAlertsFingerprint = '';
      let refreshInFlight = false;

      // Alert cards are rebuilt from API data on every meaningful refresh.
      function renderAlertCard(alert, errorEl) {
        const detections = Array.isArray(alert.detections) ? alert.detections : [];
        const zone = alert.zone || (detections[0] && detections[0].zone) || '';
        const status = alert.status || 'new';
        const isAccepted = status === 'accepted';
        const badgeClass = isAccepted ? 'status-accepted' : 'status-new';
        const actionBtn = isAccepted
          ? `<button class=\"reject\" data-action=\"delete\">Delete</button>`
          : `<button data-action=\"accept\">Accept</button>`;
        const controls = alert.id
          ? `<div class=\"actions\">${actionBtn}${isAccepted ? '' : '<button class=\"reject\" data-action=\"reject\">Reject</button>'}</div>`
          : '';
        const motionTag = alert.consumption_motion_detected
          ? ` | consumption motion (${Number(alert.consumption_motion_score || 0).toFixed(2)})`
          : '';
        const zoneTag = zone ? ` | ${zone}` : '';
        const item = document.createElement('article');
        item.className = 'alert';
        item.innerHTML =
          `<div class=\"alert-head\"><span class=\"badge ${badgeClass}\">${escapeHtml(status)}</span>` +
          `<span>${escapeHtml(alert.timestamp || 'unknown time')}</span></div>` +
          `<div class=\"meta\">${detections.length} detection(s)${escapeHtml(zoneTag + motionTag)}</div>` +
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
          const acceptedList = document.getElementById('acceptedAlerts');
          const acceptedCount = document.getElementById('acceptedCount');
          const newCount = document.getElementById('newCount');
          const acceptedSection = document.getElementById('acceptedSection');
          activeList.innerHTML = '';
          acceptedList.innerHTML = '';
          const activeAlerts = ordered.filter((alert) => (alert.status || 'new') !== 'accepted');
          const acceptedAlerts = ordered.filter((alert) => (alert.status || 'new') === 'accepted');
          if (activeAlerts.length === 0) {
            activeList.innerHTML = '<div class=\"empty-note\">No active alerts right now.</div>';
          } else {
            activeAlerts.forEach((alert) => {
              activeList.appendChild(renderAlertCard(alert, errorEl));
            });
          }
          newCount.textContent = String(activeAlerts.length);
          acceptedAlerts.forEach((alert) => {
            acceptedList.appendChild(renderAlertCard(alert, errorEl));
          });
          acceptedCount.textContent = String(acceptedAlerts.length);
          if (acceptedAlerts.length === 0) {
            acceptedSection.open = false;
            acceptedSection.style.display = 'none';
          } else {
            acceptedSection.style.display = '';
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
      refreshTrainStatus();
      setInterval(refreshAlerts, 2000);
      setInterval(refreshTrainStatus, 5000);
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

CONSUMPTION_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
    "bottle",
    "cup",
}

CAMERA_ZONES = tuple(f"Zone {chr(ord('A') + idx)}" for idx in range(9))


def get_allowed_class_ids(model, allowed_names: set[str]) -> list[int]:
    """Map readable class names to the numeric IDs expected by the YOLO API."""
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


def detect_consumption_motion(config, detections: list[dict], frame_width: int, frame_height: int) -> tuple[bool, float]:
    """Heuristically score whether a detected item is moving like it is being consumed."""
    if not config.motion_enabled:
        return False, 0.0
    now = time.time()
    class_centers: dict[str, list[tuple[float, float]]] = {}
    for det in detections:
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name not in CONSUMPTION_CLASS_NAMES:
            continue
        center = det.get("center_xy")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        try:
            x = float(center[0])
            y = float(center[1])
        except (TypeError, ValueError):
            continue
        class_centers.setdefault(class_name, []).append((x, y))

    # Drop old center points so motion is based only on a recent rolling window.
    stale_after = max(2.0, config.motion_window / max(1.0, config.stream_fps))
    for class_name, history in list(config.motion_history.items()):
        fresh = [entry for entry in history if (now - entry[2]) <= stale_after]
        if fresh:
            config.motion_history[class_name] = deque(fresh, maxlen=config.motion_window)
        else:
            config.motion_history.pop(class_name, None)

    for class_name, centers in class_centers.items():
        # Multiple detections of the same class in one frame are collapsed to one average center.
        avg_x = sum(c[0] for c in centers) / len(centers)
        avg_y = sum(c[1] for c in centers) / len(centers)
        history = config.motion_history.get(class_name)
        if history is None:
            history = deque(maxlen=config.motion_window)
            config.motion_history[class_name] = history
        history.append((avg_x, avg_y, now))

    frame_diag = max(1.0, math.hypot(frame_width, frame_height))
    class_scores: dict[str, float] = {}
    for class_name, history in config.motion_history.items():
        if class_name not in class_centers or len(history) < 4:
            continue
        first_x, first_y, _ = history[0]
        last_x, last_y, _ = history[-1]
        displacement_norm = math.hypot(last_x - first_x, last_y - first_y) / frame_diag
        upward_norm = max(0.0, (first_y - last_y) / max(1.0, float(frame_height)))
        displacement_score = displacement_norm / max(1e-6, config.motion_displacement_threshold)
        upward_score = upward_norm / max(1e-6, config.motion_upward_threshold)
        # The score mixes overall movement with upward travel, which tends to match the
        # "object being lifted toward a person" pattern better than displacement alone.
        class_scores[class_name] = 0.6 * displacement_score + 0.4 * upward_score

    max_score = 0.0
    for det in detections:
        class_name = str(det.get("class_name", "")).strip().lower()
        score = class_scores.get(class_name, 0.0)
        det["motion_score"] = round(score, 3)
        det["consumption_motion"] = bool(score >= 1.0)
        if score > max_score:
            max_score = score

    return max_score >= 1.0, round(max_score, 3)


def read_alerts(log_path: str | None) -> list[dict]:
    """Read the persisted alert list; invalid or missing files degrade to an empty list."""
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
    """Persist the full alert list back to disk."""
    if not log_path:
        return
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(alerts, handle, indent=2)


def ensure_alert_metadata(alerts: list[dict]) -> bool:
    """Backfill IDs/status fields so old alert files still work with the current UI."""
    changed = False
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        if not alert.get("id"):
            alert["id"] = uuid4().hex[:12]
            changed = True
        status = str(alert.get("status", "")).strip().lower()
        if status == "acknowledged":
            alert["status"] = "accepted"
            changed = True
        elif status not in {"new", "accepted"}:
            alert["status"] = "new"
            changed = True
    return changed


def read_class_map(path: str) -> dict[str, int]:
    """Load the class-name-to-index map used for accepted-sample training."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, dict):
                return {}
            parsed: dict[str, int] = {}
            for class_name, class_idx in data.items():
                if isinstance(class_name, str):
                    try:
                        parsed[class_name] = int(class_idx)
                    except (TypeError, ValueError):
                        continue
            return parsed
    except (OSError, json.JSONDecodeError):
        return {}


def write_class_map(path: str, class_map: dict[str, int]) -> None:
    """Persist the training class map in a deterministic format."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(class_map, handle, indent=2, sort_keys=True)


def update_dataset_yaml(config, class_map: dict[str, int]) -> None:
    """Rewrite the YOLO dataset config so training reflects the current accepted classes."""
    os.makedirs(config.training_data_dir, exist_ok=True)
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    if not names:
        names = ["item"]
    yaml_lines = [
        f"path: {os.path.abspath(config.training_data_dir)}",
        "train: images",
        "val: images",
        "names:",
    ]
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name}")
    with open(config.training_yaml_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(yaml_lines) + "\n")


def export_accepted_alert_samples(alert: dict, config) -> int:
    """Copy accepted snippets into a YOLO-style dataset and write matching label files."""
    if not isinstance(alert, dict):
        return 0
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return 0
    if not config.snippet_dir:
        return 0
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    class_map = read_class_map(config.class_map_path)
    accepted = 0
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        if det.get("training_exported"):
            continue
        snippet_file = str(det.get("snippet_file", "")).strip()
        class_name = str(det.get("class_name", "")).strip().lower()
        if not snippet_file or not class_name:
            continue
        source_path = os.path.join(config.snippet_dir, snippet_file)
        if not os.path.exists(source_path):
            continue
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
        class_id = class_map[class_name]
        ext = os.path.splitext(snippet_file)[1] or ".jpg"
        sample_stem = f"{alert.get('id', 'alert')}_{idx}_{_safe_token(class_name)}"
        dest_image = os.path.join(config.training_images_dir, f"{sample_stem}{ext}")
        dest_label = os.path.join(config.training_labels_dir, f"{sample_stem}.txt")
        shutil.copy2(source_path, dest_image)
        with open(dest_label, "w", encoding="utf-8") as label_handle:
            # Snippets are tight crops of one object, so full-image box is correct.
            label_handle.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        det["training_exported"] = True
        det["training_sample"] = os.path.basename(dest_image)
        accepted += 1
    write_class_map(config.class_map_path, class_map)
    update_dataset_yaml(config, class_map)
    return accepted


def training_status_snapshot(config) -> dict:
    """Expose the last-known training state for the dashboard polling endpoint."""
    with config.training_lock:
        return {
            "running": bool(config.training_running),
            "last_started_at": config.training_last_started_at,
            "last_completed_at": config.training_last_completed_at,
            "last_error": config.training_last_error,
            "last_message": config.training_last_message,
            "last_weights": config.training_last_weights,
        }


def _train_on_accepted_samples(config) -> None:
    """Background worker that exports accepted data, trains, and hot-swaps the model."""
    with config.training_lock:
        config.training_last_started_at = datetime.now().isoformat(timespec="seconds")
        config.training_last_error = ""
        config.training_last_message = "Preparing dataset"
    try:
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            changed = ensure_alert_metadata(alerts)
            exported_total = 0
            for alert in alerts:
                if str(alert.get("status", "")).strip().lower() == "accepted":
                    exported_total += export_accepted_alert_samples(alert, config)
            if changed or exported_total > 0:
                write_alerts(config.alert_log, alerts)
        # Training only makes sense once at least one accepted snippet has been exported.
        image_files = [
            name
            for name in os.listdir(config.training_images_dir)
            if os.path.isfile(os.path.join(config.training_images_dir, name))
        ] if os.path.isdir(config.training_images_dir) else []
        if not image_files:
            raise RuntimeError("No accepted snippets available for training yet.")
        if YOLO is None:
            raise RuntimeError("ultralytics is required to train.")
        with config.training_lock:
            config.training_last_message = f"Training on {len(image_files)} accepted snippets"
        train_model = YOLO(config.model_path)
        train_result = train_model.train(
            data=config.training_yaml_path,
            epochs=config.train_epochs,
            imgsz=config.train_imgsz,
            project=config.training_runs_dir,
            name="accepted",
            exist_ok=True,
            verbose=False,
        )
        best_path = ""
        if hasattr(train_result, "save_dir"):
            candidate = os.path.join(str(train_result.save_dir), "weights", "best.pt")
            if os.path.exists(candidate):
                best_path = candidate
        with config.training_lock:
            config.training_last_message = "Loading trained weights"
        if best_path and YOLO is not None:
            new_model = YOLO(best_path)
            with config.model_lock:
                # Replace the in-memory model so new detections use the freshly trained weights.
                config.model = new_model
                config.model_path = best_path
        with config.training_lock:
            config.training_last_completed_at = datetime.now().isoformat(timespec="seconds")
            config.training_last_weights = best_path
            config.training_last_message = "Training completed"
    except Exception as exc:
        with config.training_lock:
            config.training_last_error = str(exc)
            config.training_last_message = "Training failed"
    finally:
        with config.training_lock:
            config.training_running = False


def start_training_job(config) -> bool:
    """Start the background training worker unless one is already running."""
    with config.training_lock:
        if config.training_running:
            return False
        config.training_running = True
        worker = threading.Thread(target=_train_on_accepted_samples, args=(config,), daemon=True)
        config.training_thread = worker
    worker.start()
    return True


def append_alert(log_path: str | None, alert: dict) -> None:
    """Append one alert while preserving compatibility metadata."""
    alerts = read_alerts(log_path)
    ensure_alert_metadata(alerts)
    alerts.append(alert)
    write_alerts(log_path, alerts)


def _safe_token(value: str) -> str:
    """Convert labels into safe filename fragments."""
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    token = token.strip("_")
    return token or "item"


def add_detection_snippets(frame, detections: list[dict], snippet_dir: str | None, alert_id: str):
    """Save one crop per detection and attach the generated filenames to the alert payload."""
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


def create_alert(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    camera_zone: str,
    motion_detected: bool = False,
    motion_score: float = 0.0,
) -> dict:
    """Build the alert record stored in JSON and rendered by the dashboard."""
    alert_id = uuid4().hex[:12]
    zone = normalize_camera_zone(camera_zone)
    for det in detections:
        det["zone"] = zone
    return {
        "id": alert_id,
        "status": "new",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "zone": zone,
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "consumption_motion_detected": bool(motion_detected),
        "consumption_motion_score": round(float(motion_score), 3),
        "detections": add_detection_snippets(frame, detections, snippet_dir, alert_id),
    }


def detections_from_result(result, allowed_names: set[str] | None = None) -> list[dict]:
    """Normalize Ultralytics results into the alert/dashboard detection schema."""
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
    """Overlay bounding boxes and labels onto a frame for streaming to the browser."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        if det.get("consumption_motion"):
            label = f"{label} motion"
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
    """Build a lightweight placeholder image for test mode or missing camera states."""
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
    """Generate synthetic alerts so the UI can be exercised without a live camera."""
    alerts: list[dict] = []
    class_names = sorted(FOOD_CLASS_NAMES)
    for idx in range(max(1, limit)):
        det_count = random.randint(1, 3)
        detections = []
        zone = random.choice(CAMERA_ZONES)
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
                    "zone": zone,
                }
            )
        alert_time = datetime.now() - timedelta(seconds=idx * 3)
        alerts.append(
            {
                "id": uuid4().hex[:12],
                "status": random.choice(["new", "accepted"]),
                "timestamp": alert_time.isoformat(timespec="seconds"),
                "zone": zone,
                "frame_size": {"width": frame_width, "height": frame_height},
                "detections": detections,
            }
        )
    return alerts


def make_status_frame(width: int, height: int, label: str):
    """Create a simple text frame shown when the real camera feed is unavailable/off."""
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
    """Accept a few JSON-friendly boolean representations from the settings API."""
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


def normalize_camera_zone(value, field_name: str = "camera_zone") -> str:
    """Validate and normalize the configured camera zone label."""
    zone = str(value).strip().upper()
    if zone.startswith("ZONE "):
        zone = zone[5:].strip()
    if len(zone) == 1 and zone in "ABCDEFGHI":
        return f"Zone {zone}"
    raise ValueError(f"Invalid value for '{field_name}'. Expected Zone A through Zone I.")


def clamp_float(value, field_name: str, minimum: float, maximum: float) -> float:
    """Parse and bound a float setting so runtime updates stay within safe limits."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid number for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def clamp_int(value, field_name: str, minimum: int, maximum: int) -> int:
    """Parse and bound an integer setting so runtime updates stay within safe limits."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid integer for '{field_name}'") from None
    return max(minimum, min(maximum, numeric))


def settings_snapshot(config) -> dict:
    """Return the live runtime settings exposed to the dashboard."""
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
            "camera_zone": str(config.camera_zone),
            "updated_at": config.settings_updated_at,
            "test_mode": bool(config.test_mode),
        }


def default_settings_snapshot(config) -> dict:
    """Return the original startup settings so the UI can restore defaults."""
    with config.settings_lock:
        defaults = dict(config.default_settings)
    defaults["test_mode"] = bool(config.test_mode)
    return defaults


def update_runtime_settings(config, payload: dict) -> dict:
    """Apply validated settings updates atomically while the camera thread is running."""
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
        if "camera_zone" in payload:
            config.camera_zone = normalize_camera_zone(payload["camera_zone"])
        config.settings_updated_at = datetime.now().isoformat(timespec="seconds")
    return settings_snapshot(config)


def reset_runtime_settings(config) -> dict:
    """Reset mutable runtime settings back to their startup defaults."""
    defaults = default_settings_snapshot(config)
    defaults.pop("test_mode", None)
    return update_runtime_settings(config, defaults)


class DashboardConfig:
    """Shared mutable state for the HTTP handlers, camera loop, and training worker."""
    def __init__(
        self,
        model,
        model_path,
        alert_log,
        width,
        height,
        stream_fps,
        conf,
        iou,
        persist_frames,
        cooldown,
        clear_frames,
        camera_zone,
        snippet_dir,
        training_dir,
        train_epochs,
        train_imgsz,
        motion_enabled,
        motion_window,
        motion_displacement_threshold,
        motion_upward_threshold,
        test_mode,
    ):
        self.model = model
        self.model_path = model_path
        self.alert_log = alert_log
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.conf = conf
        self.iou = iou
        self.persist_frames = persist_frames
        self.cooldown = cooldown
        self.clear_frames = clear_frames
        self.camera_zone = normalize_camera_zone(camera_zone)
        self.camera_enabled = True
        self.detection_enabled = True
        self.settings_updated_at = datetime.now().isoformat(timespec="seconds")
        self.snippet_dir = snippet_dir
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        self.settings_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.training_lock = threading.Lock()
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
            "camera_zone": self.camera_zone,
        }
        self.stop = False
        self.consecutive = 0
        self.clear_count = 0
        self.armed = True
        self.last_alert_ts = 0.0
        self.test_mode = test_mode
        self.training_dir = os.path.abspath(training_dir)
        self.training_data_dir = os.path.join(self.training_dir, "dataset")
        self.training_images_dir = os.path.join(self.training_data_dir, "images")
        self.training_labels_dir = os.path.join(self.training_data_dir, "labels")
        self.training_yaml_path = os.path.join(self.training_data_dir, "data.yaml")
        self.class_map_path = os.path.join(self.training_dir, "class_map.json")
        self.training_runs_dir = os.path.join(self.training_dir, "runs")
        self.train_epochs = int(train_epochs)
        self.train_imgsz = int(train_imgsz)
        self.motion_enabled = bool(motion_enabled)
        self.motion_window = max(4, int(motion_window))
        self.motion_displacement_threshold = float(motion_displacement_threshold)
        self.motion_upward_threshold = float(motion_upward_threshold)
        self.motion_history: dict[str, deque] = {}
        self.training_thread = None
        self.training_running = False
        self.training_last_started_at = ""
        self.training_last_completed_at = ""
        self.training_last_error = ""
        self.training_last_message = ""
        self.training_last_weights = ""
        os.makedirs(self.training_dir, exist_ok=True)


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard page, JSON APIs, snippet images, and MJPEG stream."""
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
        if parsed.path == "/train/status":
            self._send_train_status()
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
        if parsed.path == "/train/accepted":
            self._trigger_train_accepted()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _send_json(self, payload: dict | list, status=HTTPStatus.OK):
        """Send a JSON response with no-cache headers for live dashboard polling."""
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        """Send the inline dashboard HTML page."""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        """Read and parse a bounded JSON request body."""
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
        """Return recent alerts, using generated data in test mode."""
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
        """Accept, reject, or delete an alert and persist the updated alert log."""
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
        if not alert_id or action not in {"accept", "reject", "delete"}:
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
            if action in {"reject", "delete"}:
                alerts.pop(target_index)
            else:
                # Accepting an alert also exports its snippets into the training dataset.
                exported_count = export_accepted_alert_samples(alerts[target_index], config)
                alerts[target_index]["status"] = "accepted"
                if exported_count > 0:
                    try:
                        current_samples = int(alerts[target_index].get("accepted_samples", 0))
                    except (TypeError, ValueError):
                        current_samples = 0
                    alerts[target_index]["accepted_samples"] = current_samples + exported_count
                alerts[target_index]["accepted_at"] = datetime.now().isoformat(timespec="seconds")
                alerts[target_index]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            write_alerts(config.alert_log, alerts)
        self._send_json({"ok": True, "action": action, "alert_id": alert_id}, HTTPStatus.OK)

    def _send_settings(self):
        """Return the current runtime settings to the browser."""
        config: DashboardConfig = self.server.config
        self._send_json(settings_snapshot(config), HTTPStatus.OK)

    def _update_settings(self):
        """Apply settings posted from the dashboard form."""
        config: DashboardConfig = self.server.config
        try:
            payload = self._read_json_body()
            updated = update_runtime_settings(config, payload)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _reset_settings(self):
        """Restore runtime settings to the startup defaults."""
        config: DashboardConfig = self.server.config
        updated = reset_runtime_settings(config)
        self._send_json({"ok": True, "settings": updated}, HTTPStatus.OK)

    def _send_train_status(self):
        """Return the latest background-training status snapshot."""
        config: DashboardConfig = self.server.config
        self._send_json(training_status_snapshot(config), HTTPStatus.OK)

    def _trigger_train_accepted(self):
        """Start training on accepted snippets if the environment supports it."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            self.send_error(HTTPStatus.BAD_REQUEST, "Training is disabled in --test mode")
            return
        if YOLO is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "ultralytics is required to train")
            return
        started = start_training_job(config)
        if not started:
            self.send_error(HTTPStatus.CONFLICT, "Training is already running")
            return
        self._send_json({"ok": True, "started": True}, HTTPStatus.ACCEPTED)

    def _send_snippet(self, encoded_name: str):
        """Serve one saved detection crop after validating the requested filename."""
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
        """Stream the latest annotated frame as multipart MJPEG for the browser <img> tag."""
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
    """Capture frames, run detection, update the stream frame, and create alerts."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for live camera mode.")
    cap = None

    try:
        while not config.stop:
            # Snapshot the tunable settings once per loop so the frame is processed consistently.
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
                camera_zone = str(config.camera_zone)

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                # Turning the camera off also resets the alert state machine.
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True
                config.motion_history.clear()
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
            motion_detected = False
            motion_score = 0.0
            if detection_enabled:
                with config.model_lock:
                    model = config.model
                    allowed_ids = get_allowed_class_ids(model, FOOD_CLASS_NAMES)
                    # Keep compatibility with Ultralytics releases that differ on `classes`.
                    try:
                        results = model.predict(
                            frame,
                            verbose=False,
                            conf=conf,
                            iou=iou,
                            classes=allowed_ids if allowed_ids else None,
                        )
                    except TypeError:
                        results = model.predict(
                            frame,
                            verbose=False,
                            conf=conf,
                            iou=iou,
                        )
                result = results[0]
                detections = detections_from_result(result, allowed_names=FOOD_CLASS_NAMES)
                motion_detected, motion_score = detect_consumption_motion(
                    config,
                    detections,
                    frame_width=int(frame.shape[1]),
                    frame_height=int(frame.shape[0]),
                )
            else:
                config.motion_history.clear()

            # This debounce logic makes "item stays in view" produce one alert rather than many.
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
                alert = create_alert(
                    frame,
                    detections,
                    snippet_dir=config.snippet_dir,
                    camera_zone=camera_zone,
                    motion_detected=motion_detected,
                    motion_score=motion_score,
                )
                with config.alert_lock:
                    append_alert(config.alert_log, alert)
                config.last_alert_ts = now
                config.armed = False

            annotated = draw_detections(frame, detections) if detection_enabled else frame.copy()
            if detection_enabled and motion_detected:
                cv2.putText(
                    annotated,
                    f"Eating/Drinking motion detected ({motion_score:.2f})",
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (18, 200, 18),
                    2,
                )
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
    """Start the camera worker and HTTP server that power the dashboard."""
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
    parser.add_argument(
        "--training-dir",
        default="training_data",
        help="Directory where accepted samples and train runs are stored",
    )
    parser.add_argument("--train-epochs", type=int, default=10, help="Epochs when training accepted samples")
    parser.add_argument("--train-imgsz", type=int, default=640, help="Image size for accepted-sample training")
    parser.add_argument("--width", type=int, default=0, help="Resize width (0 = original)")
    parser.add_argument("--height", type=int, default=0, help="Resize height (0 = original)")
    parser.add_argument("--camera-zone", default="Zone A", help="Zone label assigned to this camera")
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
    parser.add_argument(
        "--motion-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable heuristic eating/drinking motion detection",
    )
    parser.add_argument(
        "--motion-window",
        type=int,
        default=12,
        help="Number of recent frames used for motion scoring",
    )
    parser.add_argument(
        "--motion-displacement-threshold",
        type=float,
        default=0.07,
        help="Normalized displacement threshold for motion scoring",
    )
    parser.add_argument(
        "--motion-upward-threshold",
        type=float,
        default=0.02,
        help="Normalized upward movement threshold for motion scoring",
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
        model_path=args.model,
        alert_log=args.alert_log,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        conf=args.conf,
        iou=args.iou,
        persist_frames=args.persist_frames,
        cooldown=args.cooldown,
        clear_frames=args.clear_frames,
        camera_zone=args.camera_zone,
        snippet_dir=args.snippet_dir or None,
        training_dir=args.training_dir,
        train_epochs=args.train_epochs,
        train_imgsz=args.train_imgsz,
        motion_enabled=args.motion_enabled,
        motion_window=args.motion_window,
        motion_displacement_threshold=args.motion_displacement_threshold,
        motion_upward_threshold=args.motion_upward_threshold,
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
