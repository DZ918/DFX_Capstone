"""Browser dashboard for live detection, alert review, and accepted-sample training."""

import argparse
from collections import deque
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
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
      .alerts-card { display: flex; flex-direction: column; }
      .alerts-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 10px;
      }
      .alerts-toolbar label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #374151;
      }
            .backend-stats {
                border: 1px solid var(--border);
                border-radius: 10px;
                background: #f8fafc;
                margin-bottom: 10px;
                overflow: hidden;
            }
            .stats-total {
                padding: 10px;
                border-bottom: 1px solid var(--border);
                font-size: 13px;
                color: #1f2937;
            }
            .stats-total strong {
                font-size: 16px;
                color: #0f172a;
            }
            .stats-table-wrap {
                max-height: 190px;
                overflow-y: auto;
            }
            .stats-table {
                width: 100%;
                border-collapse: collapse;
            }
            .stats-table th,
            .stats-table td {
                border-bottom: 1px solid var(--border);
                padding: 7px 10px;
                font-size: 12px;
                text-align: left;
            }
            .stats-table th {
                background: #f1f5f9;
                color: #334155;
            }
            .stats-table td:last-child,
            .stats-table th:last-child {
                text-align: right;
            }
            .stats-updated {
                padding: 8px 10px;
                font-size: 11px;
                color: #64748b;
            }
      .alerts-scroll {
        height: min(68vh, 760px);
        overflow-y: auto;
        padding-right: 4px;
      }
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
            .alert-video-wrap {
                margin: 8px 0 10px;
                border: 1px solid var(--border);
                border-radius: 8px;
                overflow: hidden;
                background: #111827;
            }
            .alert-video {
                width: 100%;
                display: block;
                background: #000;
            }
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
      .zone-group {
        display: grid;
        gap: 8px;
        margin-bottom: 12px;
      }
      .zone-group-title {
        font-size: 12px;
        font-weight: 700;
        color: #374151;
        letter-spacing: 0.03em;
        text-transform: uppercase;
      }
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
            .lightbox img {
                position: relative;
                z-index: 1;
                max-width: 100%;
                max-height: 92vh;
                border-radius: 10px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.45);
                background: #111827;
                transform-origin: center center;
                transition: transform 0.08s ease-out;
            }
      .lightbox-close {
        position: absolute;
                z-index: 3;
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
            .lightbox-tools {
                position: absolute;
                z-index: 3;
                top: -12px;
                right: 30px;
                display: flex;
                gap: 6px;
            }
            .lightbox-tools button {
                border: 0;
                border-radius: 8px;
                background: #111827;
                color: #fff;
                width: 34px;
                height: 34px;
                font-size: 16px;
                cursor: pointer;
            }
            .map-modal {
                position: fixed;
                inset: 0;
                background: rgba(15, 23, 42, 0.86);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 1001;
                padding: 24px;
            }
            .map-modal.open { display: flex; }
            .map-modal-card {
                position: relative;
                width: min(1100px, 96vw);
                max-height: 92vh;
                background: #0f172a;
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.45);
                overflow: hidden;
            }
            .map-modal-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 10px 12px;
                color: #e2e8f0;
                background: rgba(15, 23, 42, 0.95);
                border-bottom: 1px solid rgba(148, 163, 184, 0.35);
                font-size: 13px;
            }
            .map-modal-close {
                border: 0;
                border-radius: 8px;
                background: #1e293b;
                color: #fff;
                width: 32px;
                height: 32px;
                cursor: pointer;
                font-size: 18px;
                line-height: 1;
            }
            .map-modal-body {
                padding: 10px;
                background: #020617;
            }
            .map-modal-body img {
                width: 100%;
                max-height: calc(92vh - 100px);
                object-fit: contain;
                display: block;
                border-radius: 8px;
                background: #0b1220;
            }
            .map-modal-note {
                margin-top: 8px;
                color: #94a3b8;
                font-size: 12px;
            }
      @media (max-width: 600px) {
        .settings-row { grid-template-columns: 1fr; }
        .alerts-scroll { height: 58vh; }
      }
      @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <header>
      <strong>Lab Food/Drink Monitor</strong>
      <div class=\"header-actions\">
                <button id=\"openMapBtn\" type=\"button\" aria-expanded=\"false\">Map</button>
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
            <label class=\"settings-field\">Inference image size
              <input id=\"setInferenceImgsz\" type=\"number\" min=\"160\" max=\"1280\" step=\"32\" />
            </label>
            <label class=\"settings-field\">Max inference FPS (0=unlimited)
              <input id=\"setMaxInferenceFps\" type=\"number\" min=\"0\" max=\"60\" step=\"0.5\" />
            </label>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">JPEG quality
              <input id=\"setJpegQuality\" type=\"number\" min=\"40\" max=\"95\" step=\"1\" />
            </label>
            <label class=\"settings-toggle\">
              <input id=\"setMotionEnabled\" type=\"checkbox\" />
              Motion detection
            </label>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Motion hold seconds
              <input id=\"setMotionHoldSeconds\" type=\"number\" min=\"0\" max=\"5\" step=\"0.1\" />
            </label>
            <div></div>
          </div>
          <div class=\"settings-row\">
            <label class=\"settings-field\">Webcam device
              <select id=\"setCameraIndex\"></select>
            </label>
            <div></div>
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
      <section class=\"card alerts-card\">
        <h2>Recent Alerts</h2>
        <div class=\"alerts-toolbar\">
          <label>
            <input id=\"groupAlertsByZone\" type=\"checkbox\" />
            Group by zone
          </label>
        </div>
                <section class="backend-stats" aria-label="Backend eating and drinking counts">
                    <div class="stats-total">Total people detected eating/drinking: <strong id="consumptionTotalCount">0</strong></div>
                    <div class="stats-table-wrap">
                        <table class="stats-table">
                            <thead>
                                <tr>
                                    <th>Zone</th>
                                    <th>Category</th>
                                    <th>Count</th>
                                </tr>
                            </thead>
                            <tbody id="consumptionStatsBody">
                                <tr>
                                    <td colspan="3">Loading...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div id="consumptionUpdatedAt" class="stats-updated"></div>
                </section>
        <div class=\"alerts-scroll\">
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
        </div>
        <div id=\"error\" class=\"error\"></div>
      </section>
    </main>
    <div id=\"lightbox\" class=\"lightbox\" aria-hidden=\"true\">
      <div class=\"lightbox-inner\">
                <div class=\"lightbox-tools\">
                    <button id=\"zoomOutBtn\" type=\"button\" aria-label=\"Zoom out\">-</button>
                    <button id=\"zoomResetBtn\" type=\"button\" aria-label=\"Reset zoom\">1:1</button>
                    <button id=\"zoomInBtn\" type=\"button\" aria-label=\"Zoom in\">+</button>
                </div>
        <button id=\"lightboxClose\" class=\"lightbox-close\" aria-label=\"Close expanded image\">×</button>
        <img id=\"lightboxImg\" alt=\"Expanded detection\" />
      </div>
    </div>
        <div id=\"mapModal\" class=\"map-modal\" aria-hidden=\"true\">
            <div class=\"map-modal-card\">
                <div class=\"map-modal-head\">
                    <strong>DFX Lab Layout</strong>
                    <button id=\"mapModalClose\" class=\"map-modal-close\" type=\"button\" aria-label=\"Close map\">×</button>
                </div>
                <div class=\"map-modal-body\">
                    <img id=\"mapImage\" alt=\"DFX lab layout map\" />
                    <div id=\"mapModalNote\" class=\"map-modal-note\"></div>
                </div>
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

            // The map modal displays the static DFX lab layout image served by the backend.
            function setupMapModal() {
                const openBtn = document.getElementById('openMapBtn');
                const modal = document.getElementById('mapModal');
                const closeBtn = document.getElementById('mapModalClose');
                const img = document.getElementById('mapImage');
                const note = document.getElementById('mapModalNote');

                function closeModal() {
                    modal.classList.remove('open');
                    modal.setAttribute('aria-hidden', 'true');
                    openBtn.setAttribute('aria-expanded', 'false');
                }

                function openModal() {
                    note.textContent = 'Loading map...';
                    img.src = `/map-image?ts=${Date.now()}`;
                    modal.classList.add('open');
                    modal.setAttribute('aria-hidden', 'false');
                    openBtn.setAttribute('aria-expanded', 'true');
                }

                img.addEventListener('load', () => {
                    note.textContent = '';
                });
                img.addEventListener('error', () => {
                    note.textContent = 'Map image not found. Place your file at assets/dfx_lab_map.png and refresh.';
                });
                openBtn.addEventListener('click', openModal);
                closeBtn.addEventListener('click', closeModal);
                modal.addEventListener('click', (event) => {
                    if (event.target === modal) {
                        closeModal();
                    }
                });
                document.addEventListener('keydown', (event) => {
                    if (event.key === 'Escape' && modal.classList.contains('open')) {
                        closeModal();
                    }
                });
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
                const zoomInBtn = document.getElementById('zoomInBtn');
                const zoomOutBtn = document.getElementById('zoomOutBtn');
                const zoomResetBtn = document.getElementById('zoomResetBtn');
        const alertsPanel = document.getElementById('alertsPanel');
                const minZoom = 0.5;
                const maxZoom = 5.0;
                let zoom = 1.0;

                function applyZoom() {
                    lightboxImg.style.transform = `scale(${zoom.toFixed(2)})`;
                }

                function setZoom(nextZoom) {
                    zoom = Math.min(maxZoom, Math.max(minZoom, Number(nextZoom) || 1.0));
                    applyZoom();
                }

                function resetZoom() {
                    setZoom(1.0);
                }

        function closeLightbox() {
          lightbox.classList.remove('open');
          lightbox.setAttribute('aria-hidden', 'true');
          lightboxImg.removeAttribute('src');
                    resetZoom();
        }

        function openLightbox(src, altText) {
          lightboxImg.src = src;
          lightboxImg.alt = altText || 'Expanded detection';
                    resetZoom();
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
                zoomInBtn.addEventListener('click', () => setZoom(zoom + 0.25));
                zoomOutBtn.addEventListener('click', () => setZoom(zoom - 0.25));
                zoomResetBtn.addEventListener('click', resetZoom);
                lightboxImg.addEventListener('wheel', (event) => {
                    if (!lightbox.classList.contains('open')) {
                        return;
                    }
                    event.preventDefault();
                    setZoom(zoom + (event.deltaY < 0 ? 0.15 : -0.15));
                }, { passive: false });
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
        document.getElementById('setConf').value = Number(settings.conf || 0.55).toFixed(2);
        document.getElementById('setIou').value = Number(settings.iou || 0.40).toFixed(2);
        document.getElementById('setPersistFrames').value = Number(settings.persist_frames || 5);
        document.getElementById('setClearFrames').value = Number(settings.clear_frames || 15);
        document.getElementById('setCooldown').value = Number(settings.cooldown || 15).toFixed(1);
        document.getElementById('setStreamFps').value = Number(settings.stream_fps || 10).toFixed(0);
        document.getElementById('setWidth').value = Number(settings.width || 0).toFixed(0);
        document.getElementById('setHeight').value = Number(settings.height || 0).toFixed(0);
        document.getElementById('setInferenceImgsz').value = Number(settings.inference_imgsz || 640).toFixed(0);
        document.getElementById('setMaxInferenceFps').value = Number(settings.max_inference_fps || 0).toFixed(1);
        document.getElementById('setJpegQuality').value = Number(settings.jpeg_quality || 75).toFixed(0);
        document.getElementById('setMotionEnabled').checked = Boolean(settings.motion_enabled);
        document.getElementById('setMotionHoldSeconds').value = Number(settings.motion_hold_seconds || 0.1).toFixed(1);
        document.getElementById('setCameraIndex').value = String(Number(settings.camera_index || 0));
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
          inference_imgsz: Math.round(readNumberField('setInferenceImgsz', 'inference image size')),
          max_inference_fps: readNumberField('setMaxInferenceFps', 'max inference FPS'),
          jpeg_quality: Math.round(readNumberField('setJpegQuality', 'JPEG quality')),
          motion_enabled: document.getElementById('setMotionEnabled').checked,
          motion_hold_seconds: readNumberField('setMotionHoldSeconds', 'motion hold seconds'),
          camera_index: Math.round(readNumberField('setCameraIndex', 'webcam device')),
          camera_zone: document.getElementById('setCameraZone').value,
        };
      }

      async function loadCameraDevices(selectedIndex = 0) {
        const select = document.getElementById('setCameraIndex');
        try {
          const res = await fetch('/cameras');
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
          }
          const devices = await res.json();
          const targetValue = String(Number(selectedIndex || 0));
          select.innerHTML = '';
          devices.forEach((device) => {
            const option = document.createElement('option');
            option.value = String(Number(device.index || 0));
            option.textContent = device.label || `Camera ${option.value}`;
            select.appendChild(option);
          });
          const hasSelected = Array.from(select.options).some((option) => option.value === targetValue);
          if (!hasSelected) {
            const fallback = document.createElement('option');
            fallback.value = targetValue;
            fallback.textContent = `Camera ${targetValue}`;
            select.appendChild(fallback);
          }
          select.value = targetValue;
        } catch (err) {
          select.innerHTML = '';
          const fallback = document.createElement('option');
          fallback.value = String(Number(selectedIndex || 0));
          fallback.textContent = `Camera ${fallback.value}`;
          select.appendChild(fallback);
        }
      }

      async function loadSettings(showStatus = false) {
        const res = await fetch('/settings');
        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }
        const settings = await res.json();
        await loadCameraDevices(settings.camera_index || 0);
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
                    return `${alert.id || ''}:${alert.status || ''}:${alert.timestamp || ''}:${alert.zone || ''}:${alert.video_file || ''}:${detectionKey}`;
        }).join('||');
      }

      let lastAlertsFingerprint = '';
      let refreshInFlight = false;
      let cachedOrderedAlerts = [];

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
                    ? ` | hand-to-mouth (${Number(alert.consumption_motion_score || 0).toFixed(2)})`
                    : '';
                const motionSource = alert.hand_to_mouth_source
                    ? ` | source: ${String(alert.hand_to_mouth_source).replaceAll('_', ' ')}`
                    : '';
                const motionEvents = alert.hand_to_mouth_event_count
                    ? ` | events/30s: ${Number(alert.hand_to_mouth_event_count || 0)}`
                    : '';
        const zoneTag = zone ? ` | ${zone}` : '';
                                const video = alert.video_file
                                        ? `<div class=\"alert-video-wrap\">` +
                                            `<video class=\"alert-video\" controls preload=\"metadata\" playsinline>` +
                                            `<source src=\"/videos/${encodeURIComponent(alert.video_file)}\" />` +
                                            `Your browser cannot play this recording.` +
                                            `</video>` +
                                            `<div class=\"det-label\"><a href=\"/videos/${encodeURIComponent(alert.video_file)}\" target=\"_blank\" rel=\"noopener\">Open/download recording</a></div>` +
                                            `</div>`
                                        : '';
        const item = document.createElement('article');
        item.className = 'alert';
        item.innerHTML =
          `<div class=\"alert-head\"><span class=\"badge ${badgeClass}\">${escapeHtml(status)}</span>` +
          `<span>${escapeHtml(alert.timestamp || 'unknown time')}</span></div>` +
                    `<div class=\"meta\">${detections.length} detection(s)${escapeHtml(zoneTag + motionTag + motionSource + motionEvents)}</div>` +
                    video +
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

      function groupAlertsByZone(alerts) {
        const groups = new Map();
        alerts.forEach((alert) => {
          const detections = Array.isArray(alert.detections) ? alert.detections : [];
          const zone = alert.zone || (detections[0] && detections[0].zone) || 'Unassigned';
          if (!groups.has(zone)) {
            groups.set(zone, []);
          }
          groups.get(zone).push(alert);
        });
        return groups;
      }

      function renderAlertsInto(container, alerts, errorEl, emptyMessage) {
        container.innerHTML = '';
        if (alerts.length === 0) {
          container.innerHTML = `<div class=\"empty-note\">${escapeHtml(emptyMessage)}</div>`;
          return;
        }
        const groupByZone = document.getElementById('groupAlertsByZone').checked;
        if (!groupByZone) {
          alerts.forEach((alert) => {
            container.appendChild(renderAlertCard(alert, errorEl));
          });
          return;
        }
        groupAlertsByZone(alerts).forEach((groupAlerts, zone) => {
          const group = document.createElement('section');
          group.className = 'zone-group';
          const title = document.createElement('div');
          title.className = 'zone-group-title';
          title.textContent = `${zone} (${groupAlerts.length})`;
          const list = document.createElement('div');
          list.className = 'alerts-list';
          groupAlerts.forEach((alert) => {
            list.appendChild(renderAlertCard(alert, errorEl));
          });
          group.appendChild(title);
          group.appendChild(list);
          container.appendChild(group);
        });
      }

      function renderAlerts(orderedAlerts, errorEl) {
        const activeList = document.getElementById('activeAlerts');
        const acceptedList = document.getElementById('acceptedAlerts');
        const acceptedCount = document.getElementById('acceptedCount');
        const newCount = document.getElementById('newCount');
        const acceptedSection = document.getElementById('acceptedSection');
        const activeAlerts = orderedAlerts.filter((alert) => (alert.status || 'new') !== 'accepted');
        const acceptedAlerts = orderedAlerts.filter((alert) => (alert.status || 'new') === 'accepted');

        renderAlertsInto(activeList, activeAlerts, errorEl, 'No active alerts right now.');
        renderAlertsInto(acceptedList, acceptedAlerts, errorEl, 'No accepted alerts right now.');
        newCount.textContent = String(activeAlerts.length);
        acceptedCount.textContent = String(acceptedAlerts.length);
        if (acceptedAlerts.length === 0) {
          acceptedSection.open = false;
          acceptedSection.style.display = 'none';
        } else {
          acceptedSection.style.display = '';
        }
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
          cachedOrderedAlerts = ordered;
          const nextFingerprint = fingerprintAlerts(ordered);
          if (nextFingerprint === lastAlertsFingerprint) {
            return;
          }
          lastAlertsFingerprint = nextFingerprint;
          renderAlerts(ordered, errorEl);
        } catch (err) {
          errorEl.textContent = 'Could not load alerts right now.';
        } finally {
          refreshInFlight = false;
        }
      }

            function renderConsumptionStats(data) {
                const totalEl = document.getElementById('consumptionTotalCount');
                const bodyEl = document.getElementById('consumptionStatsBody');
                const updatedEl = document.getElementById('consumptionUpdatedAt');
                totalEl.textContent = String(Number(data.total_people_detected || 0));
                const rows = Array.isArray(data.breakdown) ? data.breakdown : [];
                if (rows.length === 0) {
                    bodyEl.innerHTML = '<tr><td colspan="3">No eating/drinking detections yet.</td></tr>';
                } else {
                    bodyEl.innerHTML = rows
                        .map((row) => {
                            const zone = escapeHtml(row.zone || 'Unassigned');
                            const category = escapeHtml(row.category || 'unknown');
                            const count = Number(row.count || 0);
                            return `<tr><td>${zone}</td><td>${category}</td><td>${count}</td></tr>`;
                        })
                        .join('');
                }
                const active = Number(data.active_alerts || 0);
                const accepted = Number(data.accepted_alerts || 0);
                const generatedAt = data.generated_at ? String(data.generated_at) : 'just now';
                updatedEl.textContent = `Updated: ${generatedAt} | active: ${active} | accepted: ${accepted}`;
            }

            async function refreshConsumptionStats() {
                try {
                    const res = await fetch('/stats/consumption');
                    if (!res.ok) {
                        throw new Error(`HTTP ${res.status}`);
                    }
                    const data = await res.json();
                    renderConsumptionStats(data || {});
                } catch (err) {
                    const updatedEl = document.getElementById('consumptionUpdatedAt');
                    updatedEl.textContent = 'Could not load backend counts right now.';
                }
            }

      document.getElementById('groupAlertsByZone').addEventListener('change', () => {
        renderAlerts(cachedOrderedAlerts, document.getElementById('error'));
      });
            setupMapModal();
      setupLightbox();
      setupSettingsPanel();
      setupSettingsForm();
      refreshAlerts();
    refreshConsumptionStats();
      refreshTrainStatus();
      setInterval(refreshAlerts, 2000);
    setInterval(refreshConsumptionStats, 5000);
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

INFERENCE_CLASS_NAMES = set(FOOD_CLASS_NAMES) | {"person"}

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

DRINK_CONTAINER_CLASS_NAMES = {"bottle", "cup"}
HANDHELD_FOOD_CLASS_NAMES = {
    "apple",
    "banana",
    "orange",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
}
MOTION_TRIGGER_SCORE = 0.85
FOOD_MOTION_MIN_SCORE = 1.0
FOOD_MOTION_CONFIRM_FRAMES = 3
FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE = 1.12
PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE = 1.2
STATIONARY_FOLLOWUP_SECONDS = 30 * 60
HAND_TO_MOUTH_WINDOW_SECONDS = 30.0
HAND_TO_MOUTH_REQUIRED_EVENTS = 3
FOOD_OCCLUSION_LOOKBACK_SECONDS = 2.0
OCCLUDED_MOTION_HOLD_SECONDS = 1.2
OCCLUDED_MOTION_PROXY_SCORE = 0.86
PERSON_PROXY_MIN_AREA_RATIO = 0.02
PERSON_PROXY_DIFF_THRESHOLD = 24
PERSON_PROXY_MOUTH_MOTION_RATIO = 0.05
PERSON_PROXY_MIN_MOUTH_RATIO = 0.04
PERSON_PROXY_HOLD_SECONDS = 0.8
PERSON_PROXY_SCORE_FLOOR = 0.86
PERSON_PROXY_APPROACH_MOTION_RATIO = 0.035
PERSON_PROXY_MIN_APPROACH_RATIO = 0.028
PERSON_PROXY_TRIGGER_SCORE = 1.1
PERSON_PROXY_CONFIRM_FRAMES = 3
PERSON_PROXY_MIN_CONFIDENCE = 0.25
ALERT_DETECTION_CONFIDENCE_FLOOR = 0.62
ALERT_SNIPPET_CONFIDENCE_FLOOR = 0.64
DEFAULT_MAP_IMAGE_PATH = os.path.join("assets", "dfx_lab_map.png")
TRAIN_VIDEO_SAMPLE_MAX_FRAMES = 12
SAME_PERSON_ALERT_WINDOW_SECONDS = 120.0
SAME_PERSON_MAX_ALERTS_IN_WINDOW = 3
SAME_PERSON_SUPPRESSION_DISTANCE_RATIO = 0.18
NEW_OBJECT_LOOKBACK_SECONDS = 45.0
NEW_OBJECT_MATCH_DISTANCE_RATIO = 0.1
NEW_OBJECT_MIN_ALERT_GAP_SECONDS = 1.0
NEW_OBJECT_MIN_CONFIDENCE = 0.68

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


def _extract_detection_geometry(det: dict) -> tuple[float, float, float] | None:
    """Return center and diagonal size for a detection, or None when incomplete."""
    center = det.get("center_xy")
    bbox = det.get("bbox_xyxy")
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x = float(center[0])
        y = float(center[1])
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    box_diag = math.hypot(max(1.0, x2 - x1), max(1.0, y2 - y1))
    return x, y, box_diag


def _consumption_track_key(class_name: str) -> str:
    """Group classes that often flicker between similar labels across nearby frames."""
    normalized = class_name.strip().lower()
    if normalized in DRINK_CONTAINER_CLASS_NAMES:
        return "drink_container"
    if normalized in HANDHELD_FOOD_CLASS_NAMES:
        return "handheld_food"
    return normalized


def _smooth_motion_history(history: list[tuple[float, float, float, float]]):
    """Reduce detector jitter so motion scoring reacts to the actual trajectory."""
    if not history:
        return []
    smoothed: list[tuple[float, float, float, float]] = [history[0]]
    alpha = 0.45
    prev_x, prev_y, prev_diag, _ = history[0]
    for x, y, diag, ts in history[1:]:
        prev_x = (prev_x * (1.0 - alpha)) + (x * alpha)
        prev_y = (prev_y * (1.0 - alpha)) + (y * alpha)
        prev_diag = (prev_diag * (1.0 - alpha)) + (diag * alpha)
        smoothed.append((prev_x, prev_y, prev_diag, ts))
    return smoothed


def _extract_person_anchor(det: dict) -> tuple[float, float, float, float, float, float] | None:
    """Estimate a rough mouth-area target from one person bounding box."""
    bbox = det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + (width * 0.5)
    mouth_y = y1 + (height * 0.28)
    radius = max(30.0, min(width, height) * 0.22)
    return center_x, mouth_y, radius, x1, y1, x2, y2


def _score_person_proximity(
    person_detections: list[dict],
    latest_point: tuple[float, float],
    peak_point: tuple[float, float],
) -> float:
    """Reward motions that end near a detected person's upper face/head area."""
    best_score = 0.0
    latest_x, latest_y = latest_point
    peak_x, peak_y = peak_point
    for det in person_detections:
        anchor = _extract_person_anchor(det)
        if anchor is None:
            continue
        mouth_x, mouth_y, radius, x1, y1, x2, y2 = anchor
        latest_distance = math.hypot(latest_x - mouth_x, latest_y - mouth_y)
        peak_distance = math.hypot(peak_x - mouth_x, peak_y - mouth_y)
        latest_score = max(0.0, 1.0 - (latest_distance / max(1.0, radius * 2.0)))
        peak_score = max(0.0, 1.0 - (peak_distance / max(1.0, radius * 2.0)))
        inside_upper_person = (
            x1 <= latest_x <= x2 and y1 <= latest_y <= (y1 + ((y2 - y1) * 0.45))
        )
        best_score = max(
            best_score,
            latest_score,
            peak_score,
            1.0 if inside_upper_person else 0.0,
        )
    return best_score


def _score_motion_track(
    track: dict,
    frame_diag: float,
    frame_height: int,
    config,
    person_detections: list[dict] | None = None,
) -> float:
    """Score a matched object track based on recent path, lift, and proximity to a person."""
    raw_history = list(track.get("history", ()))
    if len(raw_history) < 4:
        return 0.0
    history = _smooth_motion_history(raw_history)
    path_length = 0.0
    upward_total = 0.0
    downward_total = 0.0
    min_y = history[0][1]
    min_entry = history[0]
    rising_steps = 0
    for prev, cur in zip(history, history[1:]):
        path_length += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        delta_y = prev[1] - cur[1]
        upward_total += max(0.0, delta_y)
        downward_total += max(0.0, -delta_y)
        if delta_y > max(2.0, frame_height * 0.006):
            rising_steps += 1
        if cur[1] < min_y:
            min_y = cur[1]
            min_entry = cur
    first_x, first_y, first_diag, _ = history[0]
    last_x, last_y, _, _ = history[-1]
    net_displacement = math.hypot(last_x - first_x, last_y - first_y)
    net_upward = max(0.0, first_y - last_y)
    horizontal_travel = abs(last_x - first_x)
    path_norm = path_length / max(1.0, frame_diag)
    displacement_norm = net_displacement / max(1.0, frame_diag)
    upward_norm = upward_total / max(1.0, float(frame_height))
    downward_norm = downward_total / max(1.0, float(frame_height))
    net_upward_norm = net_upward / max(1.0, float(frame_height))
    vertical_gain_norm = max(0.0, first_y - min_y) / max(1.0, float(frame_height))
    linearity = net_displacement / max(1.0, path_length)
    vertical_dominance = net_upward / max(1.0, horizontal_travel + net_upward)
    directional_consistency = rising_steps / max(1.0, float(len(history) - 1))
    size_growth = max(0.0, (max(point[2] for point in history) - first_diag) / max(1.0, first_diag))
    upper_zone_score = max(0.0, 0.72 - (min_y / max(1.0, float(frame_height)))) / 0.72
    person_proximity = _score_person_proximity(
        person_detections or [],
        latest_point=(last_x, last_y),
        peak_point=(min_entry[0], min_entry[1]),
    )

    # Hard gate: there should be meaningful object movement, or a clear approach toward a
    # detected person's head/face region, otherwise we are likely just seeing bbox jitter.
    if displacement_norm < (config.motion_displacement_threshold * 0.35) and person_proximity < 0.65:
        track.get("score_history", deque()).clear()
        return 0.0
    if vertical_gain_norm < (config.motion_upward_threshold * 0.45) and person_proximity < 0.55:
        track.get("score_history", deque()).clear()
        return 0.0
    if net_upward_norm < (config.motion_upward_threshold * 0.35) and person_proximity < 0.75:
        track.get("score_history", deque()).clear()
        return 0.0
    if linearity < 0.22 and directional_consistency < 0.35 and person_proximity < 0.75:
        track.get("score_history", deque()).clear()
        return 0.0
    if vertical_dominance < 0.18 and person_proximity < 0.7:
        track.get("score_history", deque()).clear()
        return 0.0

    displacement_score = displacement_norm / max(1e-6, config.motion_displacement_threshold)
    path_score = path_norm / max(1e-6, config.motion_displacement_threshold * 2.0)
    upward_score = upward_norm / max(1e-6, config.motion_upward_threshold)
    net_upward_score = net_upward_norm / max(1e-6, config.motion_upward_threshold)
    lift_score = vertical_gain_norm / max(1e-6, config.motion_upward_threshold * 1.25)
    downward_penalty = downward_norm / max(1e-6, config.motion_upward_threshold)
    size_growth_score = size_growth / 0.25

    raw_score = (
        0.16 * displacement_score
        + 0.08 * path_score
        + 0.16 * upward_score
        + 0.16 * net_upward_score
        + 0.12 * lift_score
        + 0.08 * linearity
        + 0.07 * directional_consistency
        + 0.05 * upper_zone_score
        + 0.05 * size_growth_score
        + 0.17 * person_proximity
        - 0.10 * downward_penalty
    )
    score_history = track.setdefault("score_history", deque(maxlen=max(3, config.motion_window // 3)))
    score_history.append(max(0.0, raw_score))
    return sum(score_history) / len(score_history)


def detect_consumption_motion(
    config,
    detections: list[dict],
    frame_width: int,
    frame_height: int,
    person_detections: list[dict] | None = None,
) -> tuple[bool, float]:
    """Heuristically score whether a detected item is moving like it is being consumed."""
    if not config.motion_enabled:
        return False, 0.0

    now = time.time()
    frame_diag = max(1.0, math.hypot(frame_width, frame_height))
    sample_fps = max(
        1.0,
        config.max_inference_fps if getattr(config, "max_inference_fps", 0.0) > 0.0 else config.stream_fps,
    )
    stale_after = max(2.0, config.motion_window / sample_fps)

    for track_id, track in list(config.motion_tracks.items()):
        if (now - float(track.get("last_seen", 0.0))) > stale_after:
            config.motion_tracks.pop(track_id, None)

    candidates: list[tuple[int, dict, str, float, float, float]] = []
    for det_index, det in enumerate(detections):
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name not in CONSUMPTION_CLASS_NAMES:
            continue
        geometry = _extract_detection_geometry(det)
        if geometry is None:
            continue
        x, y, box_diag = geometry
        candidates.append((det_index, det, _consumption_track_key(class_name), x, y, box_diag))

    used_track_ids: set[int] = set()
    matched_track_ids: dict[int, int] = {}
    for det_index, det, track_key, x, y, box_diag in sorted(
        candidates,
        key=lambda item: float(item[1].get("confidence", 0.0)),
        reverse=True,
    ):
        best_track_id = None
        best_distance = float("inf")
        max_match_distance = max(frame_diag * 0.05, box_diag * 1.6, 60.0)
        for track_id, track in config.motion_tracks.items():
            if track_id in used_track_ids:
                continue
            if track.get("track_key") != track_key:
                continue
            last_x, last_y, _, _ = track["history"][-1]
            distance = math.hypot(x - last_x, y - last_y)
            if distance <= max_match_distance and distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        if best_track_id is None:
            best_track_id = config.next_motion_track_id
            config.next_motion_track_id += 1
            config.motion_tracks[best_track_id] = {
                "track_key": track_key,
                "history": deque(maxlen=config.motion_window),
                "last_seen": now,
                "score_history": deque(maxlen=max(3, config.motion_window // 3)),
                "active_until": 0.0,
            }
        track = config.motion_tracks[best_track_id]
        track["last_seen"] = now
        track["history"].append((x, y, box_diag, now))
        used_track_ids.add(best_track_id)
        matched_track_ids[det_index] = best_track_id

    max_score = 0.0
    for det_index, det in enumerate(detections):
        track_id = matched_track_ids.get(det_index)
        if track_id is None:
            det["motion_score"] = 0.0
            det["consumption_motion"] = False
            continue
        track = config.motion_tracks.get(track_id)
        score = (
            0.0
            if track is None
            else _score_motion_track(
                track,
                frame_diag,
                frame_height,
                config,
                person_detections=person_detections,
            )
        )
        if track is not None and score >= MOTION_TRIGGER_SCORE:
            track["active_until"] = now + max(0.0, float(getattr(config, "motion_hold_seconds", 0.1)))
        effective_score = score
        if track is not None and now <= float(track.get("active_until", 0.0)):
            effective_score = max(effective_score, MOTION_TRIGGER_SCORE)
        det["motion_track_id"] = track_id
        det["motion_score"] = round(effective_score, 3)
        det["consumption_motion"] = bool(effective_score >= MOTION_TRIGGER_SCORE)
        if effective_score > max_score:
            max_score = effective_score

    return max_score >= MOTION_TRIGGER_SCORE, round(max_score, 3)


def detect_person_hand_to_mouth_proxy(
    config,
    frame,
    person_detections: list[dict],
    now_ts: float,
) -> tuple[bool, float]:
    """Fallback hand-to-mouth motion signal from person-only upper-face ROI movement."""
    if cv2 is None or np is None:
        return False, 0.0
    if not person_detections:
        history = getattr(config, "person_proxy_score_history", None)
        if history is not None:
            history.append(0.0)
        config.person_proxy_trigger_streak = 0
        return (now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))), 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    frame_area = max(1.0, float(frame_h * frame_w))
    downsample = 0.5
    small = cv2.resize(
        gray,
        (max(1, int(frame_w * downsample)), max(1, int(frame_h * downsample))),
        interpolation=cv2.INTER_AREA,
    )
    prev_small = getattr(config, "person_proxy_prev_gray", None)
    config.person_proxy_prev_gray = small
    if prev_small is None or getattr(prev_small, "shape", None) != small.shape:
        config.person_proxy_trigger_streak = 0
        return (now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))), 0.0

    diff = cv2.absdiff(small, prev_small)
    _, motion_mask = cv2.threshold(diff, PERSON_PROXY_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    best_mouth_ratio = 0.0
    best_approach_ratio = 0.0
    best_raw_score = 0.0
    scale_x = float(small.shape[1]) / max(1.0, float(frame_w))
    scale_y = float(small.shape[0]) / max(1.0, float(frame_h))
    for det in person_detections:
        try:
            person_confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            person_confidence = 0.0
        if person_confidence < PERSON_PROXY_MIN_CONFIDENCE:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        person_w = max(1.0, x2 - x1)
        person_h = max(1.0, y2 - y1)
        if (person_w * person_h) / frame_area < PERSON_PROXY_MIN_AREA_RATIO:
            continue

        # Focus on mouth/hand interaction area in upper-middle of the person box.
        roi_x1 = x1 + (person_w * 0.24)
        roi_x2 = x1 + (person_w * 0.76)
        roi_y1 = y1 + (person_h * 0.08)
        roi_y2 = y1 + (person_h * 0.48)

        sx1 = max(0, min(motion_mask.shape[1] - 1, int(roi_x1 * scale_x)))
        sy1 = max(0, min(motion_mask.shape[0] - 1, int(roi_y1 * scale_y)))
        sx2 = max(sx1 + 1, min(motion_mask.shape[1], int(roi_x2 * scale_x)))
        sy2 = max(sy1 + 1, min(motion_mask.shape[0], int(roi_y2 * scale_y)))
        roi = motion_mask[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            continue
        mouth_ratio = float(cv2.countNonZero(roi)) / float(roi.size)

        approach_x1 = x1 + (person_w * 0.18)
        approach_x2 = x1 + (person_w * 0.82)
        approach_y1 = y1 + (person_h * 0.22)
        approach_y2 = y1 + (person_h * 0.78)
        ax1 = max(0, min(motion_mask.shape[1] - 1, int(approach_x1 * scale_x)))
        ay1 = max(0, min(motion_mask.shape[0] - 1, int(approach_y1 * scale_y)))
        ax2 = max(ax1 + 1, min(motion_mask.shape[1], int(approach_x2 * scale_x)))
        ay2 = max(ay1 + 1, min(motion_mask.shape[0], int(approach_y2 * scale_y)))
        approach_roi = motion_mask[ay1:ay2, ax1:ax2]
        approach_ratio = 0.0
        if approach_roi.size > 0:
            approach_ratio = float(cv2.countNonZero(approach_roi)) / float(approach_roi.size)

        mouth_score = mouth_ratio / max(1e-6, PERSON_PROXY_MOUTH_MOTION_RATIO)
        approach_score = approach_ratio / max(1e-6, PERSON_PROXY_APPROACH_MOTION_RATIO)
        raw_score = (0.72 * mouth_score) + (0.28 * approach_score)
        if raw_score > best_raw_score:
            best_raw_score = raw_score
            best_mouth_ratio = mouth_ratio
            best_approach_ratio = approach_ratio

    score_history = getattr(config, "person_proxy_score_history", None)
    if score_history is not None:
        score_history.append(best_raw_score)
        smoothed_score = sum(score_history) / max(1, len(score_history))
    else:
        smoothed_score = best_raw_score

    candidate_trigger = (
        smoothed_score >= PERSON_PROXY_TRIGGER_SCORE
        and best_mouth_ratio >= PERSON_PROXY_MIN_MOUTH_RATIO
        and best_approach_ratio >= PERSON_PROXY_MIN_APPROACH_RATIO
    )
    if candidate_trigger:
        config.person_proxy_trigger_streak = int(getattr(config, "person_proxy_trigger_streak", 0)) + 1
    else:
        config.person_proxy_trigger_streak = max(0, int(getattr(config, "person_proxy_trigger_streak", 0)) - 1)
    triggered = int(getattr(config, "person_proxy_trigger_streak", 0)) >= PERSON_PROXY_CONFIRM_FRAMES
    if triggered:
        config.person_proxy_active_until = now_ts + PERSON_PROXY_HOLD_SECONDS
    active = now_ts <= float(getattr(config, "person_proxy_active_until", 0.0))
    score = min(1.5, max(0.0, smoothed_score))
    if active:
        score = max(score, PERSON_PROXY_SCORE_FLOOR)
    return active, round(score, 3)


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


DETECTION_SUMMARY_HEADERS = (
    "alert_id",
    "timestamp",
    "date",
    "weekday",
    "time",
    "zone",
    "category",
    "confidence",
    "status",
    "consumption_motion_detected",
    "consumption_motion_score",
    "snippet_file",
)


def _split_alert_timestamp(value: str) -> tuple[str, str, str, str]:
    """Normalize one alert timestamp into CSV-friendly date and time columns."""
    raw = str(value or "").strip()
    if not raw:
        return "", "", "", ""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw, "", "", ""
    return (
        parsed.isoformat(timespec="seconds"),
        parsed.date().isoformat(),
        parsed.strftime("%A"),
        parsed.time().isoformat(timespec="seconds"),
    )


def append_detection_summary_csv(summary_path: str | None, alert: dict) -> None:
    """Append one CSV row per detection when a new alert is created."""
    if not summary_path or not isinstance(alert, dict):
        return
    detections = alert.get("detections")
    if not isinstance(detections, list) or not detections:
        return
    summary_path = os.path.abspath(summary_path)
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    timestamp, date_value, weekday, time_value = _split_alert_timestamp(alert.get("timestamp", ""))
    rows: list[dict[str, str | float | bool]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        rows.append(
            {
                "alert_id": str(alert.get("id", "")).strip(),
                "timestamp": timestamp,
                "date": date_value,
                "weekday": weekday,
                "time": time_value,
                "zone": str(alert.get("zone", "")).strip(),
                "category": str(det.get("class_name", "")).strip().lower(),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "status": str(alert.get("status", "")).strip().lower() or "new",
                "consumption_motion_detected": bool(alert.get("consumption_motion_detected", False)),
                "consumption_motion_score": round(float(alert.get("consumption_motion_score", 0.0)), 3),
                "snippet_file": str(det.get("snippet_file", "")).strip(),
            }
        )
    if not rows:
        return
    with open(summary_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_SUMMARY_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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


def alert_has_consumption_event(alert: dict) -> bool:
    """Treat one alert as one eating/drinking person event for backend counts."""
    if not isinstance(alert, dict):
        return False
    if bool(alert.get("consumption_motion_detected", False)):
        return True
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return False
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name in CONSUMPTION_CLASS_NAMES:
            return True
    return False


def _primary_consumption_category(alert: dict) -> str:
    """Pick one category per alert so table totals stay person-event based."""
    detections = alert.get("detections") if isinstance(alert, dict) else None
    if not isinstance(detections, list):
        return "unknown"
    best_name = "unknown"
    best_conf = -1.0
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if class_name not in CONSUMPTION_CLASS_NAMES:
            continue
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence >= best_conf:
            best_conf = confidence
            best_name = class_name
    return best_name


def build_consumption_stats(alerts: list[dict]) -> dict:
    """Aggregate eating/drinking counts for the dashboard stats table."""
    total_people_detected = 0
    active_alerts = 0
    accepted_alerts = 0
    breakdown_counts: dict[tuple[str, str], int] = {}
    for alert in alerts:
        if not alert_has_consumption_event(alert):
            continue
        total_people_detected += 1
        status = str(alert.get("status", "")).strip().lower()
        if status == "accepted":
            accepted_alerts += 1
        else:
            active_alerts += 1
        zone = str(alert.get("zone", "")).strip() or "Unassigned"
        category = _primary_consumption_category(alert)
        key = (zone, category)
        breakdown_counts[key] = breakdown_counts.get(key, 0) + 1
    breakdown = [
        {"zone": zone, "category": category, "count": count}
        for (zone, category), count in sorted(
            breakdown_counts.items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    return {
        "total_people_detected": total_people_detected,
        "active_alerts": active_alerts,
        "accepted_alerts": accepted_alerts,
        "breakdown": breakdown,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


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


def _normalized_xywh_from_xyxy(
    bbox_xyxy: list[float],
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> tuple[float, float, float, float] | None:
    """Project an XYXY box between frame sizes and return YOLO-normalized XYWH."""
    if source_width <= 1 or source_height <= 1 or target_width <= 1 or target_height <= 1:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1 *= scale_x
    x2 *= scale_x
    y1 *= scale_y
    y2 *= scale_y
    x1 = max(0.0, min(target_width - 1.0, x1))
    x2 = max(0.0, min(target_width, x2))
    y1 = max(0.0, min(target_height - 1.0, y1))
    y2 = max(0.0, min(target_height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) * 0.5) / target_width
    cy = ((y1 + y2) * 0.5) / target_height
    bw = (x2 - x1) / target_width
    bh = (y2 - y1) / target_height
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, bw)),
        max(0.0, min(1.0, bh)),
    )


def export_video_frames_for_training(alert: dict, config, class_map: dict[str, int]) -> int:
    """Extract labeled frames from one alert video so training can learn from motion clips."""
    if cv2 is None:
        return 0
    if not isinstance(alert, dict):
        return 0
    if alert.get("training_video_exported"):
        return 0
    video_file = str(alert.get("video_file", "")).strip()
    detections = alert.get("detections")
    frame_size = alert.get("frame_size")
    if not video_file or not isinstance(detections, list):
        return 0
    if not isinstance(frame_size, dict):
        return 0
    source_width = float(frame_size.get("width", 0) or 0)
    source_height = float(frame_size.get("height", 0) or 0)
    if source_width <= 1 or source_height <= 1:
        return 0
    if not config.video_dir:
        return 0
    video_path = os.path.join(config.video_dir, video_file)
    if not os.path.exists(video_path):
        return 0

    usable_detections: list[tuple[int, list[float], str]] = []
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        bbox_xyxy = det.get("bbox_xyxy")
        if (
            not class_name
            or not isinstance(bbox_xyxy, (list, tuple))
            or len(bbox_xyxy) != 4
        ):
            continue
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
        usable_detections.append((class_map[class_name], [float(v) for v in bbox_xyxy], class_name))
    if not usable_detections:
        return 0

    capture = cv2.VideoCapture(video_path)
    if not capture or not capture.isOpened():
        return 0
    exported = 0
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = TRAIN_VIDEO_SAMPLE_MAX_FRAMES
        sample_count = max(1, min(TRAIN_VIDEO_SAMPLE_MAX_FRAMES, total_frames))
        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = sorted({
                int(round(i * (max(0, total_frames - 1) / float(sample_count - 1))))
                for i in range(sample_count)
            })

        alert_id = str(alert.get("id", "alert")).strip() or "alert"
        for sample_idx, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or not hasattr(frame, "shape"):
                continue
            frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
            if frame_w <= 1 or frame_h <= 1:
                continue

            stem = f"{alert_id}_video_{sample_idx:02d}"
            image_name = f"{stem}.jpg"
            image_path = os.path.join(config.training_images_dir, image_name)
            label_path = os.path.join(config.training_labels_dir, f"{stem}.txt")
            if not cv2.imwrite(image_path, frame):
                continue

            label_lines: list[str] = []
            for class_id, bbox_xyxy, _class_name in usable_detections:
                normalized = _normalized_xywh_from_xyxy(
                    bbox_xyxy,
                    source_width=source_width,
                    source_height=source_height,
                    target_width=float(frame_w),
                    target_height=float(frame_h),
                )
                if normalized is None:
                    continue
                cx, cy, bw, bh = normalized
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not label_lines:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
                continue
            with open(label_path, "w", encoding="utf-8") as label_handle:
                label_handle.write("\n".join(label_lines) + "\n")
            exported += 1
    finally:
        capture.release()

    if exported > 0:
        alert["training_video_exported"] = True
        alert["training_video_samples"] = int(exported)
    return exported


def select_alert_person_center(person_detections: list[dict], detections: list[dict]) -> tuple[float, float] | None:
    """Choose one representative person center for duplicate-alert suppression."""
    if not person_detections:
        return None
    target_center = None
    if detections:
        sum_x = 0.0
        sum_y = 0.0
        count = 0
        for det in detections:
            center = det.get("center_xy") if isinstance(det, dict) else None
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                continue
            try:
                cx = float(center[0])
                cy = float(center[1])
            except (TypeError, ValueError):
                continue
            sum_x += cx
            sum_y += cy
            count += 1
        if count > 0:
            target_center = (sum_x / count, sum_y / count)

    best = None
    best_score = float("-inf")
    for det in person_detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        if target_center is None:
            score = confidence
        else:
            score = (confidence * 2.0) - math.hypot(center_x - target_center[0], center_y - target_center[1])
        if score > best_score:
            best_score = score
            best = (center_x, center_y)
    return best


def _iter_alert_object_candidates(detections: list[dict]):
    """Yield class/geometry tuples from detections for novelty checks."""
    for det in detections:
        if not isinstance(det, dict):
            continue
        class_name = str(det.get("class_name", "")).strip().lower()
        if not class_name:
            continue
        geometry = _extract_detection_geometry(det)
        if geometry is None:
            continue
        try:
            confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        x, y, box_diag = geometry
        yield class_name, confidence, x, y, box_diag


def has_novel_alert_object(config, detections: list[dict], frame_diag: float, now_ts: float) -> bool:
    """Return True when at least one detection is a new object versus recent same-class alerts."""
    while (
        config.alert_object_history
        and (now_ts - float(config.alert_object_history[0][4])) > NEW_OBJECT_LOOKBACK_SECONDS
    ):
        config.alert_object_history.popleft()

    for class_name, confidence, x, y, box_diag in _iter_alert_object_candidates(detections):
        if confidence < NEW_OBJECT_MIN_CONFIDENCE:
            continue
        matched = False
        for prev_class, prev_x, prev_y, prev_diag, _prev_ts in config.alert_object_history:
            if prev_class != class_name:
                continue
            max_dist = max(
                float(frame_diag) * NEW_OBJECT_MATCH_DISTANCE_RATIO,
                float(box_diag) * 0.6,
                float(prev_diag) * 0.6,
            )
            if math.hypot(x - float(prev_x), y - float(prev_y)) <= max_dist:
                matched = True
                break
        if not matched:
            return True
    return False


def remember_alert_objects(config, detections: list[dict], now_ts: float) -> None:
    """Store recently alerted objects so we can distinguish repeats from genuinely new objects."""
    for class_name, confidence, x, y, box_diag in _iter_alert_object_candidates(detections):
        if confidence < ALERT_DETECTION_CONFIDENCE_FLOOR:
            continue
        config.alert_object_history.append((class_name, x, y, box_diag, float(now_ts)))


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
        snippet_bbox = det.get("snippet_bbox_xywhn")
        if not (
            isinstance(snippet_bbox, (list, tuple))
            and len(snippet_bbox) == 4
            and all(isinstance(v, (int, float)) for v in snippet_bbox)
        ):
            snippet_bbox = [0.5, 0.5, 1.0, 1.0]
        cx, cy, bw, bh = (max(0.0, min(1.0, float(v))) for v in snippet_bbox)
        with open(dest_label, "w", encoding="utf-8") as label_handle:
            label_handle.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        det["training_exported"] = True
        det["training_sample"] = os.path.basename(dest_image)
        accepted += 1
    accepted += export_video_frames_for_training(alert, config, class_map)
    write_class_map(config.class_map_path, class_map)
    update_dataset_yaml(config, class_map)
    return accepted


def export_rejected_alert_samples(alert: dict, config) -> int:
    """Copy rejected snippets into the dataset as negative samples with empty labels."""
    if not isinstance(alert, dict):
        return 0
    detections = alert.get("detections")
    if not isinstance(detections, list):
        return 0
    if not config.snippet_dir:
        return 0
    os.makedirs(config.training_images_dir, exist_ok=True)
    os.makedirs(config.training_labels_dir, exist_ok=True)
    exported = 0
    for idx, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        snippet_file = str(det.get("snippet_file", "")).strip()
        class_name = str(det.get("class_name", "")).strip().lower() or "item"
        if not snippet_file:
            continue
        source_path = os.path.join(config.snippet_dir, snippet_file)
        if not os.path.exists(source_path):
            continue
        ext = os.path.splitext(snippet_file)[1] or ".jpg"
        sample_stem = f"{alert.get('id', 'alert')}_{idx}_{_safe_token(class_name)}_negative"
        dest_image = os.path.join(config.training_images_dir, f"{sample_stem}{ext}")
        dest_label = os.path.join(config.training_labels_dir, f"{sample_stem}.txt")
        shutil.copy2(source_path, dest_image)
        # Empty label files tell YOLO this image is a hard negative: it should contain none
        # of the tracked classes even though the detector previously thought it did.
        with open(dest_label, "w", encoding="utf-8") as label_handle:
            label_handle.write("")
        exported += 1
    update_dataset_yaml(config, read_class_map(config.class_map_path))
    return exported


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


def validate_training_environment() -> None:
    """Fail early with a readable message when binary dependencies are incompatible."""
    try:
        from matplotlib import font_manager  # noqa: F401
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        if "numpy.core.multiarray failed to import" in detail or "_ARRAY_API" in detail:
            raise RuntimeError(
                "Training dependencies are incompatible: reinstall or upgrade matplotlib so it matches the installed NumPy version."
            ) from exc
        raise RuntimeError(f"Training dependency import failed: {detail}") from exc


def _train_on_accepted_samples(config) -> None:
    """Background worker that exports accepted data, trains, and hot-swaps the model."""
    with config.training_lock:
        config.training_last_started_at = datetime.now().isoformat(timespec="seconds")
        config.training_last_error = ""
        config.training_last_message = "Preparing dataset"
    try:
        validate_training_environment()
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
            # Keep the live detector on the original general-purpose model. Accepted-sample
            # training is often class-specific and can quickly become too narrow for runtime
            # detection if its weights are hot-swapped immediately.
            config.training_last_message = "Training completed; weights saved for manual review"
        with config.training_lock:
            config.training_last_completed_at = datetime.now().isoformat(timespec="seconds")
            config.training_last_weights = best_path
            if not config.training_last_message:
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


def append_alert(log_path: str | None, alert: dict, summary_csv_path: str | None = None) -> None:
    """Append one alert while preserving compatibility metadata."""
    alerts = read_alerts(log_path)
    ensure_alert_metadata(alerts)
    alerts.append(alert)
    write_alerts(log_path, alerts)
    append_detection_summary_csv(summary_csv_path, alert)


def _safe_token(value: str) -> str:
    """Convert labels into safe filename fragments."""
    token = "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower())
    token = token.strip("_")
    return token or "item"


def _clamp_box(bounds: list[float], frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Clamp a float bbox into valid image coordinates."""
    x1, y1, x2, y2 = bounds
    left = max(0, min(frame_width - 1, int(round(float(x1)))))
    top = max(0, min(frame_height - 1, int(round(float(y1)))))
    right = max(left + 1, min(frame_width, int(round(float(x2)))))
    bottom = max(top + 1, min(frame_height, int(round(float(y2)))))
    return left, top, right, bottom


def _nearest_person_box(target_box: tuple[int, int, int, int], context_detections: list[dict]) -> tuple[int, int, int, int] | None:
    """Pick the person bbox nearest to the target item center."""
    tx1, ty1, tx2, ty2 = target_box
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    best_distance = float("inf")
    best_box = None
    for det in context_detections:
        if not isinstance(det, dict):
            continue
        if str(det.get("class_name", "")).strip().lower() != "person":
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        px1, py1, px2, py2 = (int(float(v)) for v in bbox)
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        distance = math.hypot(target_cx - pcx, target_cy - pcy)
        if distance < best_distance:
            best_distance = distance
            best_box = (px1, py1, px2, py2)
    return best_box


def add_detection_snippets(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    alert_id: str,
    context_detections: list[dict] | None = None,
):
    """Save contextual crops with item/person framing and attach filenames to detections."""
    if not snippet_dir:
        return detections
    os.makedirs(snippet_dir, exist_ok=True)
    height, width = frame.shape[:2]
    context_detections = context_detections or detections
    for idx, det in enumerate(detections):
        try:
            det_confidence = float(det.get("confidence", 0.0))
        except (TypeError, ValueError):
            det_confidence = 0.0
        if det_confidence < ALERT_SNIPPET_CONFIDENCE_FLOOR:
            continue
        bbox = det.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        item_left, item_top, item_right, item_bottom = _clamp_box(list(bbox), width, height)
        person_box = _nearest_person_box(
            (item_left, item_top, item_right, item_bottom),
            context_detections,
        )

        crop_left, crop_top, crop_right, crop_bottom = item_left, item_top, item_right, item_bottom
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            crop_left = min(crop_left, max(0, px1))
            crop_top = min(crop_top, max(0, py1))
            crop_right = max(crop_right, min(width, px2))
            crop_bottom = max(crop_bottom, min(height, py2))

        box_w = max(1, crop_right - crop_left)
        box_h = max(1, crop_bottom - crop_top)
        margin_x = max(16, int(box_w * 0.2))
        margin_y = max(16, int(box_h * 0.2))
        crop_left = max(0, crop_left - margin_x)
        crop_top = max(0, crop_top - margin_y)
        crop_right = min(width, crop_right + margin_x)
        crop_bottom = min(height, crop_bottom + margin_y)

        crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()
        if crop.size == 0:
            continue

        local_item_left = max(0, item_left - crop_left)
        local_item_top = max(0, item_top - crop_top)
        local_item_right = max(local_item_left + 1, item_right - crop_left)
        local_item_bottom = max(local_item_top + 1, item_bottom - crop_top)
        cv2.rectangle(crop, (local_item_left, local_item_top), (local_item_right, local_item_bottom), (0, 180, 255), 2)
        class_name = str(det.get("class_name", "item"))
        cv2.putText(
            crop,
            class_name,
            (local_item_left, max(16, local_item_top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )
        if person_box is not None:
            px1, py1, px2, py2 = person_box
            local_px1 = max(0, px1 - crop_left)
            local_py1 = max(0, py1 - crop_top)
            local_px2 = max(local_px1 + 1, px2 - crop_left)
            local_py2 = max(local_py1 + 1, py2 - crop_top)
            cv2.rectangle(crop, (local_px1, local_py1), (local_px2, local_py2), (48, 195, 110), 2)

        class_token = _safe_token(det.get("class_name", "item"))
        snippet_file = f"{alert_id}_{idx}_{class_token}.jpg"
        snippet_path = os.path.join(snippet_dir, snippet_file)
        if cv2.imwrite(snippet_path, crop):
            det["snippet_file"] = snippet_file
            crop_h, crop_w = crop.shape[:2]
            cx = ((local_item_left + local_item_right) / 2.0) / max(1.0, float(crop_w))
            cy = ((local_item_top + local_item_bottom) / 2.0) / max(1.0, float(crop_h))
            bw = (local_item_right - local_item_left) / max(1.0, float(crop_w))
            bh = (local_item_bottom - local_item_top) / max(1.0, float(crop_h))
            det["snippet_bbox_xywhn"] = [
                round(max(0.0, min(1.0, cx)), 6),
                round(max(0.0, min(1.0, cy)), 6),
                round(max(0.0, min(1.0, bw)), 6),
                round(max(0.0, min(1.0, bh)), 6),
            ]
    return detections


def add_alert_video(
    recent_frames: list,
    video_dir: str | None,
    alert_id: str,
    fps: float,
) -> tuple[str, str] | None:
    """Persist a short alert clip captured around the trigger moment."""
    if cv2 is None:
        return None
    if not video_dir or not recent_frames:
        return None
    os.makedirs(video_dir, exist_ok=True)

    first = recent_frames[0]
    if first is None or not hasattr(first, "shape") or len(first.shape) < 2:
        return None
    height, width = int(first.shape[0]), int(first.shape[1])
    if width <= 0 or height <= 0:
        return None
    safe_fps = max(3.0, float(fps or 8.0))

    def _is_usable_alert_video(path: str, min_written_frames: int) -> bool:
        """Accept only files that are non-trivial and decodable for browser playback."""
        if not os.path.exists(path):
            return False
        # Tiny files are often invalid headers with no real media payload.
        if os.path.getsize(path) < 4096:
            return False
        capture = cv2.VideoCapture(path)
        if not capture or not capture.isOpened():
            return False
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok, first_frame = capture.read()
        finally:
            capture.release()
        return bool(ok and first_frame is not None and frame_count >= max(2, min_written_frames // 2))

    # Ubuntu browsers often cannot play MPEG-4 Part 2 (mp4v) streams even in .mp4 files.
    # Prefer H264/WebM on Linux; if unavailable, skip attachment instead of creating an
    # unplayable 0:00 clip.
    if platform.system() == "Linux":
        codec_candidates = [
            ("avc1", "mp4", "video/mp4"),
            ("H264", "mp4", "video/mp4"),
            ("X264", "mp4", "video/mp4"),
            ("VP80", "webm", "video/webm"),
            ("VP90", "webm", "video/webm"),
        ]
    else:
        codec_candidates = [
            ("avc1", "mp4", "video/mp4"),
            ("H264", "mp4", "video/mp4"),
            ("X264", "mp4", "video/mp4"),
            ("VP90", "webm", "video/webm"),
            ("VP80", "webm", "video/webm"),
            ("mp4v", "mp4", "video/mp4"),
        ]
    for codec, extension, mime in codec_candidates:
        output_name = f"{alert_id}.{extension}"
        output_path = os.path.join(video_dir, output_name)
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*codec),
            safe_fps,
            (width, height),
        )
        if not writer or not writer.isOpened():
            continue
        written_frames = 0
        try:
            for frame in recent_frames:
                if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
                    continue
                frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
                if frame_h != height or frame_w != width:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                written_frames += 1
        finally:
            writer.release()
        if written_frames <= 0:
            try:
                os.remove(output_path)
            except OSError:
                pass
            continue
        if _is_usable_alert_video(output_path, min_written_frames=written_frames):
            return output_name, mime
        try:
            os.remove(output_path)
        except OSError:
            pass
    return None


def create_alert(
    frame,
    detections: list[dict],
    snippet_dir: str | None,
    video_dir: str | None,
    recent_frames: list | None,
    video_fps: float,
    camera_zone: str,
    context_detections: list[dict] | None = None,
    motion_detected: bool = False,
    motion_score: float = 0.0,
    hand_to_mouth_source: str = "none",
    hand_to_mouth_event_count: int = 0,
    attach_video: bool = False,
    alert_reason: str = "standard",
) -> dict:
    """Build the alert record stored in JSON and rendered by the dashboard."""
    alert_id = uuid4().hex[:12]
    zone = normalize_camera_zone(camera_zone)
    for det in detections:
        det["zone"] = zone
    video_file = None
    video_mime = ""
    if motion_detected and attach_video:
        video_result = add_alert_video(
            recent_frames=recent_frames or [],
            video_dir=video_dir,
            alert_id=alert_id,
            fps=video_fps,
        )
        if video_result is not None:
            video_file, video_mime = video_result
    return {
        "id": alert_id,
        "status": "new",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "alert_reason": str(alert_reason or "standard"),
        "zone": zone,
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "consumption_motion_detected": bool(motion_detected),
        "consumption_motion_score": round(float(motion_score), 3),
        "hand_to_mouth_source": str(hand_to_mouth_source or "none"),
        "hand_to_mouth_event_count": int(max(0, hand_to_mouth_event_count)),
        "video_file": video_file,
        "video_mime": video_mime,
        "detections": add_detection_snippets(
            frame,
            detections,
            snippet_dir,
            alert_id,
            context_detections=context_detections,
        ),
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


def _camera_backend_flag() -> int:
    """Prefer AVFoundation on macOS so camera probing stays on the native backend."""
    if cv2 is None:
        return 0
    if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        return int(cv2.CAP_AVFOUNDATION)
    return int(getattr(cv2, "CAP_ANY", 0))


def _open_camera_capture(index: int):
    """Open one camera index using the best backend for the current platform."""
    if cv2 is None:
        return None
    backend = _camera_backend_flag()
    capture = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
    if capture is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _suggest_camera_probe_count(default_count: int) -> int:
    """Use the host OS to avoid probing obviously invalid camera indices."""
    if platform.system() != "Darwin":
        return max(1, default_count)
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPCameraDataType", "-json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
        cameras = payload.get("SPCameraDataType", [])
        if isinstance(cameras, list) and cameras:
            return max(1, min(default_count, len(cameras)))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return max(1, default_count)


def list_camera_devices(max_devices: int = 8) -> list[dict]:
    """Probe a small range of camera indices for use in the dashboard dropdown."""
    devices: list[dict] = []
    if cv2 is None:
        return devices
    probe_count = _suggest_camera_probe_count(max_devices)
    for index in range(probe_count):
        cap = _open_camera_capture(index)
        if cap is None:
            continue
        available = bool(cap.isOpened())
        if available:
            cap.release()
        devices.append(
            {
                "index": index,
                "label": f"Camera {index}",
                "available": available,
            }
        )
    return [device for device in devices if device["available"]] or devices[:1]


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
            "motion_enabled": bool(config.motion_enabled),
            "conf": float(config.conf),
            "iou": float(config.iou),
            "persist_frames": int(config.persist_frames),
            "cooldown": float(config.cooldown),
            "clear_frames": int(config.clear_frames),
            "stream_fps": float(config.stream_fps),
            "width": int(config.width),
            "height": int(config.height),
            "inference_imgsz": int(config.inference_imgsz),
            "max_inference_fps": float(config.max_inference_fps),
            "jpeg_quality": int(config.jpeg_quality),
            "motion_hold_seconds": float(config.motion_hold_seconds),
            "camera_index": int(config.camera_index),
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
        if "motion_enabled" in payload:
            config.motion_enabled = parse_bool(payload["motion_enabled"], "motion_enabled")
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
        if "inference_imgsz" in payload:
            config.inference_imgsz = clamp_int(payload["inference_imgsz"], "inference_imgsz", 160, 1280)
        if "max_inference_fps" in payload:
            config.max_inference_fps = clamp_float(
                payload["max_inference_fps"], "max_inference_fps", 0.0, 60.0
            )
        if "jpeg_quality" in payload:
            config.jpeg_quality = clamp_int(payload["jpeg_quality"], "jpeg_quality", 40, 95)
        if "motion_hold_seconds" in payload:
            config.motion_hold_seconds = clamp_float(
                payload["motion_hold_seconds"], "motion_hold_seconds", 0.0, 5.0
            )
        if "camera_index" in payload:
            config.camera_index = clamp_int(payload["camera_index"], "camera_index", 0, 32)
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
        camera_index,
        width,
        height,
        stream_fps,
        conf,
        iou,
        persist_frames,
        cooldown,
        clear_frames,
        camera_zone,
        map_image_path,
        snippet_dir,
        detection_summary_csv,
        inference_imgsz,
        max_inference_fps,
        jpeg_quality,
        motion_hold_seconds,
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
        self.camera_index = int(camera_index)
        self.width = width
        self.height = height
        self.stream_fps = stream_fps
        self.conf = conf
        self.iou = iou
        self.persist_frames = persist_frames
        self.cooldown = cooldown
        self.clear_frames = clear_frames
        self.camera_zone = normalize_camera_zone(camera_zone)
        if os.path.isabs(map_image_path):
            self.map_image_path = os.path.abspath(map_image_path)
        else:
            project_root = os.path.dirname(os.path.abspath(__file__))
            self.map_image_path = os.path.abspath(os.path.join(project_root, map_image_path))
        self.camera_enabled = True
        self.detection_enabled = True
        self.settings_updated_at = datetime.now().isoformat(timespec="seconds")
        self.snippet_dir = snippet_dir
        self.video_dir = (
            os.path.join(os.path.abspath(snippet_dir), "videos") if snippet_dir else None
        )
        self.detection_summary_csv = (
            os.path.abspath(detection_summary_csv) if detection_summary_csv else None
        )
        self.latest_frame = None
        self.latest_jpeg = None
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
            "inference_imgsz": int(inference_imgsz),
            "max_inference_fps": float(max_inference_fps),
            "jpeg_quality": int(jpeg_quality),
            "motion_hold_seconds": float(motion_hold_seconds),
            "camera_index": int(camera_index),
            "camera_zone": self.camera_zone,
            "motion_enabled": bool(motion_enabled),
        }
        self.stop = False
        self.consecutive = 0
        self.clear_count = 0
        self.armed = True
        self.last_alert_ts = 0.0
        self.stationary_first_alert_ts = 0.0
        self.stationary_followup_sent = False
        self.motion_event_times: deque[float] = deque()
        self.last_motion_active = False
        self.last_food_seen_ts = 0.0
        self.occlusion_motion_until = 0.0
        self.person_proxy_prev_gray = None
        self.person_proxy_active_until = 0.0
        self.person_proxy_score_history: deque[float] = deque(maxlen=6)
        self.person_proxy_trigger_streak = 0
        self.food_motion_confirm_streak = 0
        self.person_alert_history: deque[tuple[float, float, float]] = deque(maxlen=300)
        self.alert_object_history: deque[tuple[str, float, float, float, float]] = deque(maxlen=500)
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
        self.inference_imgsz = int(inference_imgsz)
        self.max_inference_fps = float(max_inference_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.motion_hold_seconds = float(motion_hold_seconds)
        self.motion_enabled = bool(motion_enabled)
        self.motion_window = max(4, int(motion_window))
        self.motion_displacement_threshold = float(motion_displacement_threshold)
        self.motion_upward_threshold = float(motion_upward_threshold)
        self.motion_tracks: dict[int, dict] = {}
        self.next_motion_track_id = 1
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
        if parsed.path == "/cameras":
            self._send_cameras()
            return
        if parsed.path == "/train/status":
            self._send_train_status()
            return
        if parsed.path == "/stats/consumption":
            self._send_consumption_stats()
            return
        if parsed.path == "/map-image":
            self._send_map_image()
            return
        if parsed.path.startswith("/snippets/"):
            self._send_snippet(parsed.path.removeprefix("/snippets/"))
            return
        if parsed.path.startswith("/videos/"):
            self._send_video(parsed.path.removeprefix("/videos/"))
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
            if action == "reject":
                export_rejected_alert_samples(alerts[target_index], config)
                alerts.pop(target_index)
            elif action == "delete":
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

    def _send_cameras(self):
        """Return the currently probeable webcam devices for the dropdown."""
        self._send_json(list_camera_devices(), HTTPStatus.OK)

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

    def _send_consumption_stats(self):
        """Return aggregate eating/drinking counts for the dashboard table."""
        config: DashboardConfig = self.server.config
        if config.test_mode:
            frame_width = config.width or 640
            frame_height = config.height or 360
            alerts = make_random_alerts(40, frame_width, frame_height)
            self._send_json(build_consumption_stats(alerts), HTTPStatus.OK)
            return
        with config.alert_lock:
            alerts = read_alerts(config.alert_log)
            if ensure_alert_metadata(alerts):
                write_alerts(config.alert_log, alerts)
        self._send_json(build_consumption_stats(alerts), HTTPStatus.OK)

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

    def _send_map_image(self):
        """Serve the configured DFX lab layout image used by the map modal."""
        config: DashboardConfig = self.server.config
        map_path = os.path.abspath(str(config.map_image_path))
        if not os.path.exists(map_path):
            # Fallback for Linux/case-sensitive filesystems and alternate extensions.
            root_dir = os.path.dirname(map_path)
            candidates = [
                "dfx_lab_map.png",
                "dfx_lab_map.jpg",
                "dfx_lab_map.jpeg",
                "dfx_lab_map.webp",
                "dfx_lab_map.svg",
            ]
            for candidate in candidates:
                probe = os.path.join(root_dir, candidate)
                if os.path.exists(probe):
                    map_path = probe
                    break
        if not os.path.exists(map_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Map image not found")
            return
        with open(map_path, "rb") as handle:
            body = handle.read()
        content_type = "image/png"
        lower = map_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif lower.endswith(".webp"):
            content_type = "image/webp"
        elif lower.endswith(".svg"):
            content_type = "image/svg+xml"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self, encoded_name: str):
        """Serve one saved alert video clip after validating the filename."""
        config: DashboardConfig = self.server.config
        if not config.video_dir:
            self.send_error(HTTPStatus.NOT_FOUND, "Video storage is disabled")
            return
        requested_name = unquote(encoded_name)
        if requested_name != os.path.basename(requested_name):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video path")
            return
        video_root = os.path.abspath(config.video_dir)
        video_path = os.path.abspath(os.path.join(video_root, requested_name))
        if not video_path.startswith(f"{video_root}{os.sep}"):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid video path")
            return
        if not os.path.exists(video_path):
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        content_type = "video/mp4"
        lower = video_path.lower()
        if lower.endswith(".avi"):
            content_type = "video/x-msvideo"
        elif lower.endswith(".webm"):
            content_type = "video/webm"
        file_size = os.path.getsize(video_path)
        range_header = self.headers.get("Range", "").strip()
        if not range_header.startswith("bytes="):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(file_size))
            self.end_headers()
            with open(video_path, "rb") as handle:
                self.wfile.write(handle.read())
            return

        range_spec = range_header[6:].split(",", 1)[0].strip()
        start_text, _, end_text = range_spec.partition("-")
        if not _:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        try:
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else (file_size - 1)
            else:
                suffix_len = int(end_text)
                if suffix_len <= 0:
                    raise ValueError()
                start = max(0, file_size - suffix_len)
                end = file_size - 1
        except ValueError:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        if start < 0 or end < start or start >= file_size:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid range")
            return
        end = min(end, file_size - 1)
        length = (end - start) + 1

        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(video_path, "rb") as handle:
            handle.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

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
                    payload = None if config.latest_jpeg is None else bytes(config.latest_jpeg)
                if payload is None:
                    time.sleep(0.05)
                    continue
                with config.settings_lock:
                    stream_fps = float(config.stream_fps)
                delay = 1.0 / max(1.0, stream_fps)
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
    active_cam_index = int(cam_index)
    last_detections = []
    last_motion_detected = False
    last_motion_score = 0.0
    next_inference_at = 0.0
    allowed_ids = None
    recent_frames: deque = deque(maxlen=80)

    try:
        while not config.stop:
            loop_started_at = time.perf_counter()
            # Snapshot the tunable settings once per loop so the frame is processed consistently.
            with config.settings_lock:
                camera_enabled = bool(config.camera_enabled)
                detection_enabled = bool(config.detection_enabled)
                motion_enabled = bool(config.motion_enabled)
                conf = float(config.conf)
                iou = float(config.iou)
                persist_frames = int(config.persist_frames)
                cooldown = float(config.cooldown)
                clear_frames = int(config.clear_frames)
                stream_fps = float(config.stream_fps)
                out_width = int(config.width)
                out_height = int(config.height)
                inference_imgsz = int(config.inference_imgsz)
                max_inference_fps = float(config.max_inference_fps)
                jpeg_quality = int(config.jpeg_quality)
                camera_index = int(config.camera_index)
                camera_zone = str(config.camera_zone)

            if not camera_enabled:
                if cap is not None:
                    cap.release()
                    cap = None
                    active_cam_index = camera_index
                # Turning the camera off also resets the alert state machine.
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True
                config.stationary_first_alert_ts = 0.0
                config.stationary_followup_sent = False
                config.motion_event_times.clear()
                config.last_motion_active = False
                config.last_food_seen_ts = 0.0
                config.occlusion_motion_until = 0.0
                config.person_proxy_prev_gray = None
                config.person_proxy_active_until = 0.0
                config.person_proxy_trigger_streak = 0
                config.food_motion_confirm_streak = 0
                config.person_alert_history.clear()
                config.alert_object_history.clear()
                config.motion_tracks.clear()
                config.next_motion_track_id = 1
                last_detections = []
                last_motion_detected = False
                last_motion_score = 0.0
                recent_frames.clear()
                paused = make_status_frame(out_width or 640, out_height or 360, "Camera is OFF")
                if paused is not None:
                    ok, encoded = cv2.imencode(".jpg", paused, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    with config.frame_lock:
                        config.latest_frame = paused
                        config.latest_jpeg = encoded.tobytes() if ok else None
                time.sleep(0.15)
                continue

            if cap is not None and active_cam_index != camera_index:
                cap.release()
                cap = None
                active_cam_index = camera_index
                allowed_ids = None
                next_inference_at = 0.0

            if cap is None:
                cap = _open_camera_capture(camera_index)
                if cap is None:
                    raise RuntimeError("OpenCV camera backend is not available.")
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    unavailable = make_status_frame(
                        out_width or 640,
                        out_height or 360,
                        f"Camera {camera_index} unavailable",
                    )
                    if unavailable is not None:
                        ok, encoded = cv2.imencode(
                            ".jpg",
                            unavailable,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                        )
                        with config.frame_lock:
                            config.latest_frame = unavailable
                            config.latest_jpeg = encoded.tobytes() if ok else None
                    time.sleep(1.0)
                    continue
                active_cam_index = camera_index
                allowed_ids = None
                next_inference_at = 0.0

            wall_now = time.time()

            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                allowed_ids = None
                recent_frames.clear()
                time.sleep(0.1)
                continue

            recent_frames.append(frame.copy())

            detections = last_detections
            motion_detected = last_motion_detected
            motion_score = last_motion_score
            motion_source = "none"
            inference_ran = False
            all_detections = detections
            person_detections: list[dict] = []
            if detection_enabled:
                perf_now = time.perf_counter()
                inference_due = max_inference_fps <= 0.0 or perf_now >= next_inference_at
                if inference_due:
                    inference_ran = True
                    if max_inference_fps > 0.0:
                        next_inference_at = perf_now + (1.0 / max(0.1, max_inference_fps))
                    else:
                        next_inference_at = 0.0
                with config.model_lock:
                    model = config.model
                    if inference_ran:
                        if allowed_ids is None:
                            allowed_ids = get_allowed_class_ids(model, INFERENCE_CLASS_NAMES)
                        predict_kwargs = {
                            "verbose": False,
                            "conf": conf,
                            "iou": iou,
                            "imgsz": inference_imgsz,
                            "classes": allowed_ids if allowed_ids else None,
                        }
                        # Keep compatibility with Ultralytics releases that differ on `classes`.
                        try:
                            results = model.predict(frame, **predict_kwargs)
                        except TypeError:
                            predict_kwargs.pop("classes", None)
                            results = model.predict(frame, **predict_kwargs)
                if inference_ran:
                    result = results[0]
                    all_detections = detections_from_result(result, allowed_names=INFERENCE_CLASS_NAMES)
                    detections = [
                        det
                        for det in all_detections
                        if (
                            str(det.get("class_name", "")).strip().lower() in FOOD_CLASS_NAMES
                            and float(det.get("confidence", 0.0)) >= ALERT_DETECTION_CONFIDENCE_FLOOR
                        )
                    ]
                    person_detections = [
                        det
                        for det in all_detections
                        if str(det.get("class_name", "")).strip().lower() == "person"
                    ]
                    if motion_enabled:
                        raw_motion_detected, raw_motion_score = detect_consumption_motion(
                            config,
                            detections,
                            frame_width=int(frame.shape[1]),
                            frame_height=int(frame.shape[0]),
                            person_detections=person_detections,
                        )
                        if raw_motion_detected and raw_motion_score >= FOOD_MOTION_MIN_SCORE:
                            config.food_motion_confirm_streak = min(
                                FOOD_MOTION_CONFIRM_FRAMES + 2,
                                int(getattr(config, "food_motion_confirm_streak", 0)) + 1,
                            )
                        else:
                            config.food_motion_confirm_streak = 0
                        motion_detected = int(getattr(config, "food_motion_confirm_streak", 0)) >= FOOD_MOTION_CONFIRM_FRAMES
                        motion_score = float(raw_motion_score)
                        if motion_detected:
                            motion_source = "food_track"
                        person_proxy_detected, person_proxy_score = detect_person_hand_to_mouth_proxy(
                            config,
                            frame,
                            person_detections,
                            wall_now,
                        )
                        if not motion_detected and not detections and person_proxy_detected:
                            # Allow alerting on pure hand-to-mouth gesture only when no food object is visible.
                            motion_detected = True
                            motion_score = max(float(motion_score), float(person_proxy_score))
                            motion_source = "person_proxy"
                        if detections:
                            config.last_food_seen_ts = wall_now
                        if not motion_detected:
                            # Keep motion active briefly when food is momentarily occluded by a hand.
                            recently_saw_food = (
                                (wall_now - float(config.last_food_seen_ts)) <= FOOD_OCCLUSION_LOOKBACK_SECONDS
                            )
                            if (
                                not detections
                                and bool(person_detections)
                                and recently_saw_food
                                and wall_now <= float(config.occlusion_motion_until)
                            ):
                                motion_detected = True
                                motion_score = max(float(motion_score), OCCLUDED_MOTION_PROXY_SCORE)
                                motion_source = "food_occluded"
                        if motion_detected:
                            config.occlusion_motion_until = wall_now + OCCLUDED_MOTION_HOLD_SECONDS
                    else:
                        config.motion_tracks.clear()
                        config.next_motion_track_id = 1
                        config.food_motion_confirm_streak = 0
                        motion_detected = False
                        motion_score = 0.0

                    hand_to_mouth_event_active = (
                        motion_detected
                        and (
                            (
                                motion_source == "food_track"
                                and bool(detections)
                                and float(motion_score) >= FOOD_HAND_TO_MOUTH_EVENT_MIN_SCORE
                            )
                            or (
                                motion_source in {"person_proxy", "food_occluded"}
                                and not bool(detections)
                                and float(motion_score) >= PROXY_HAND_TO_MOUTH_EVENT_MIN_SCORE
                            )
                        )
                    )

                    if hand_to_mouth_event_active and not config.last_motion_active:
                        config.motion_event_times.append(wall_now)
                    while (
                        config.motion_event_times
                        and (wall_now - config.motion_event_times[0]) > HAND_TO_MOUTH_WINDOW_SECONDS
                    ):
                        config.motion_event_times.popleft()
                    config.last_motion_active = bool(hand_to_mouth_event_active)

                    last_detections = detections
                    last_motion_detected = motion_detected
                    last_motion_score = motion_score
            else:
                config.motion_tracks.clear()
                config.next_motion_track_id = 1
                config.motion_event_times.clear()
                config.last_motion_active = False
                config.last_food_seen_ts = 0.0
                config.occlusion_motion_until = 0.0
                config.person_proxy_prev_gray = None
                config.person_proxy_active_until = 0.0
                config.person_proxy_trigger_streak = 0
                config.food_motion_confirm_streak = 0
                config.person_alert_history.clear()
                config.alert_object_history.clear()
                last_detections = []
                last_motion_detected = False
                last_motion_score = 0.0
                next_inference_at = 0.0
                detections = []
                motion_detected = False
                motion_score = 0.0

            # This debounce logic makes "item stays in view" produce one alert rather than many.
            if detection_enabled and inference_ran and detections:
                config.consecutive += 1
                config.clear_count = 0
            elif detection_enabled and inference_ran:
                config.consecutive = 0
                config.clear_count += 1
                if config.clear_count >= max(1, clear_frames):
                    config.armed = True
                    config.stationary_first_alert_ts = 0.0
                    config.stationary_followup_sent = False
            elif not detection_enabled:
                config.consecutive = 0
                config.clear_count = 0
                config.armed = True
                config.stationary_first_alert_ts = 0.0
                config.stationary_followup_sent = False

            motion_burst_trigger = (
                inference_ran
                and motion_detected
                and (bool(detections) or bool(person_detections))
                and len(config.motion_event_times) >= HAND_TO_MOUTH_REQUIRED_EVENTS
            )
            stationary_followup_trigger = (
                inference_ran
                and bool(detections)
                and not motion_detected
                and config.stationary_first_alert_ts > 0.0
                and not config.stationary_followup_sent
                and (wall_now - config.stationary_first_alert_ts) >= STATIONARY_FOLLOWUP_SECONDS
            )
            initial_trigger = (
                inference_ran
                and detections
                and config.consecutive >= max(1, persist_frames)
                and config.armed
                and (wall_now - config.last_alert_ts) >= max(0.0, cooldown)
            )
            frame_diag = math.hypot(float(frame.shape[1]), float(frame.shape[0]))
            new_object_trigger = (
                inference_ran
                and bool(detections)
                and (wall_now - config.last_alert_ts) >= NEW_OBJECT_MIN_ALERT_GAP_SECONDS
                and has_novel_alert_object(
                    config,
                    detections,
                    frame_diag=frame_diag,
                    now_ts=wall_now,
                )
            )

            same_person_suppressed = False
            alert_person_center = select_alert_person_center(person_detections, detections)
            if alert_person_center is not None:
                # Keep only recent person-alert entries in the configured time window.
                while (
                    config.person_alert_history
                    and (wall_now - float(config.person_alert_history[0][2])) > SAME_PERSON_ALERT_WINDOW_SECONDS
                ):
                    config.person_alert_history.popleft()
                suppression_distance = SAME_PERSON_SUPPRESSION_DISTANCE_RATIO * max(1.0, frame_diag)
                matched_alerts = 0
                for px, py, pts in config.person_alert_history:
                    if (wall_now - float(pts)) > SAME_PERSON_ALERT_WINDOW_SECONDS:
                        continue
                    distance = math.hypot(alert_person_center[0] - px, alert_person_center[1] - py)
                    if distance <= suppression_distance:
                        matched_alerts += 1
                same_person_suppressed = matched_alerts >= SAME_PERSON_MAX_ALERTS_IN_WINDOW

            should_alert = (
                not same_person_suppressed
                and (motion_burst_trigger or stationary_followup_trigger or initial_trigger or new_object_trigger)
            )
            if should_alert:
                reason = "initial"
                if motion_burst_trigger:
                    reason = "motion_burst"
                elif new_object_trigger:
                    reason = "new_object"
                elif stationary_followup_trigger:
                    reason = "stationary_followup"
                alert = create_alert(
                    frame,
                    detections,
                    snippet_dir=config.snippet_dir,
                    video_dir=config.video_dir,
                    recent_frames=list(recent_frames),
                    video_fps=stream_fps,
                    camera_zone=camera_zone,
                    context_detections=all_detections,
                    motion_detected=motion_detected,
                    motion_score=motion_score,
                    hand_to_mouth_source=motion_source,
                    hand_to_mouth_event_count=len(config.motion_event_times),
                    attach_video=(
                        motion_burst_trigger
                        and (bool(detections) or bool(person_detections))
                    ),
                    alert_reason=reason,
                )
                with config.alert_lock:
                    append_alert(
                        config.alert_log,
                        alert,
                        summary_csv_path=config.detection_summary_csv,
                    )
                config.last_alert_ts = wall_now
                remember_alert_objects(config, detections, wall_now)
                if alert_person_center is not None:
                    config.person_alert_history.append(
                        (
                            float(alert_person_center[0]),
                            float(alert_person_center[1]),
                            float(wall_now),
                        )
                    )
                if motion_burst_trigger:
                    # Immediate burst alerts bypass cooldown/arming but should not spam every frame.
                    config.motion_event_times.clear()
                elif stationary_followup_trigger:
                    config.stationary_followup_sent = True
                elif new_object_trigger:
                    config.armed = True
                    config.stationary_first_alert_ts = 0.0
                    config.stationary_followup_sent = False
                else:
                    config.armed = False
                    if motion_detected:
                        config.stationary_first_alert_ts = 0.0
                        config.stationary_followup_sent = False
                    else:
                        config.stationary_first_alert_ts = wall_now
                        config.stationary_followup_sent = False
            elif same_person_suppressed and motion_burst_trigger:
                # Prevent burst-alert loops for one person while still allowing future events.
                config.motion_event_times.clear()

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

            ok, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            with config.frame_lock:
                config.latest_frame = annotated
                config.latest_jpeg = encoded.tobytes() if ok else None

            target_loop_delay = 1.0 / max(1.0, stream_fps)
            remaining = target_loop_delay - (time.perf_counter() - loop_started_at)
            if remaining > 0:
                time.sleep(remaining)
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
        "--detection-summary-csv",
        default="detections_summary.csv",
        help="CSV file where one row per new detection is appended",
    )
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
    parser.add_argument(
        "--inference-imgsz",
        type=int,
        default=640,
        help="Inference image size passed to YOLO; smaller values reduce CPU/GPU usage",
    )
    parser.add_argument(
        "--max-inference-fps",
        type=float,
        default=0.0,
        help="Upper bound for YOLO inference frequency (0 = unlimited)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality for the MJPEG browser stream",
    )
    parser.add_argument(
        "--motion-hold-seconds",
        type=float,
        default=0.1,
        help="How long a positive motion signal stays active after being detected",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--camera-zone", default="Zone A", help="Zone label assigned to this camera")
    parser.add_argument(
        "--map-image",
        default=DEFAULT_MAP_IMAGE_PATH,
        help="Path to DFX lab layout image shown by the Map button",
    )
    parser.add_argument("--fps", type=int, default=10, help="Stream FPS")
    parser.add_argument("--conf", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="IoU threshold")
    parser.add_argument(
        "--persist-frames",
        type=int,
        default=5,
        help="Require this many consecutive frames with detections to alert",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=15.0,
        help="Minimum seconds between alerts",
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=15,
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
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        conf=args.conf,
        iou=args.iou,
        persist_frames=args.persist_frames,
        cooldown=args.cooldown,
        clear_frames=args.clear_frames,
        camera_zone=args.camera_zone,
        map_image_path=args.map_image,
        snippet_dir=args.snippet_dir or None,
        detection_summary_csv=args.detection_summary_csv or None,
        inference_imgsz=args.inference_imgsz,
        max_inference_fps=args.max_inference_fps,
        jpeg_quality=args.jpeg_quality,
        motion_hold_seconds=args.motion_hold_seconds,
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
