// Platrix dashboard client.
"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => r.json());

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function addEventRow(ev, flash = true) {
  const body = $("events-body");
  const tr = document.createElement("tr");
  if (flash) tr.className = "flash";
  const snap = ev.snapshot_path
    ? `<img src="/${ev.snapshot_path.replace(/^.*snapshots\//, "snapshots/")}" alt="snap" />`
    : "—";
  tr.innerHTML = `
    <td>${fmtTime(ev.created_at)}</td>
    <td><span class="plate">${ev.plate_text || ev.plate_text_fa || "—"}</span></td>
    <td class="score">${Math.round((ev.score || 0) * 100)}%</td>
    <td class="meta">${(ev.source || "").slice(0, 22)}</td>
    <td>${snap}</td>`;
  body.prepend(tr);
  while (body.children.length > 200) body.removeChild(body.lastChild);
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const live = s.running;
    $("status-dot").className = "dot" + (live ? " live" : "");
    $("status-text").textContent = live ? `Live · ${s.source}` : "Idle";
    $("fps-meta").textContent = live ? `${s.fps} fps · ${s.detector}/${s.ocr}` : "— fps";
    $("video-idle").style.display = live ? "none" : "flex";
    if (s.stats) {
      $("stat-total").textContent = s.stats.total_events;
      $("stat-distinct").textContent = s.stats.distinct_plates;
    }
  } catch (_) {}
}

async function loadEvents() {
  const { events } = await api("/api/events?limit=100");
  $("events-body").innerHTML = "";
  events.forEach((ev) => addEventRow(ev, false));
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    addEventRow(ev, true);
    refreshStatus();
  };
  ws.onclose = () => setTimeout(connectWs, 3000);
}

$("btn-start").onclick = async () => {
  const source = $("source-input").value.trim() || "0";
  await api("/api/stream/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source }),
  });
  $("live").src = "/api/stream/mjpeg?t=" + Date.now();
  refreshStatus();
};

$("btn-stop").onclick = async () => {
  await api("/api/stream/stop", { method: "POST" });
  refreshStatus();
};

$("file-input").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("upload-result").textContent = "Analyzing…";
  const fd = new FormData();
  fd.append("file", file);
  const res = await api("/api/recognize", { method: "POST", body: fd });
  $("upload-result").textContent = `${res.count} plate(s) found`;
  $("upload-preview").innerHTML = `<img src="${res.annotated_image}" alt="result" />`;
  (res.plates || []).forEach((p) => addEventRow(p, true));
  refreshStatus();
};

async function version() {
  const h = await api("/api/health");
  $("version").textContent = h.version;
}

version();
loadEvents();
refreshStatus();
connectWs();
setInterval(refreshStatus, 4000);
