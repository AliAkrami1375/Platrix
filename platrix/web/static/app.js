// Platrix dashboard client.
"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => r.json());
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const state = {
  camDir: "unknown",
  watchList: "white",
  watchFilter: "all",
  watchSearch: "",
  histFilter: "all",
  lastTestUrl: null,
};

/* ---------- helpers ---------- */
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const now = new Date();
  const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return d.toDateString() === now.toDateString() ? t : `${d.toLocaleDateString()} ${t}`;
}
function snapUrl(path) {
  if (!path) return null;
  const i = path.indexOf("snapshots/");
  return i >= 0 ? "/" + path.slice(i) : null;
}
function toast(msg, kind = "black") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.className = "toast"), 3500);
}

/* ---------- navigation ---------- */
const titles = { image: "Image Detection", stream: "Video Stream", watch: "Watchlist", stats: "Statistics" };
const subtitles = { image: "Upload & recognize", stream: "Cameras & live view", watch: "Lists & history search", stats: "System overview" };
function switchView(view) {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("view-" + view).classList.add("active");
  $("appbar-title").textContent = titles[view];
  $("appbar-sub").textContent = subtitles[view];
  if (view === "stream") { loadCameras(); loadStreamLog(); }
  if (view === "watch") { loadWatchlist(); runHistorySearch(); }
  if (view === "stats") loadStats();
}
document.querySelectorAll(".nav-btn").forEach((b) => (b.onclick = () => switchView(b.dataset.view)));

/* ================= IMAGE DETECTION ================= */
const dz = $("dropzone");
["dragover", "dragenter"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, () => dz.classList.remove("drag")));
dz.addEventListener("drop", (ev) => { ev.preventDefault(); if (ev.dataTransfer.files[0]) recognize(ev.dataTransfer.files[0]); });
$("file-input").onchange = (e) => e.target.files[0] && recognize(e.target.files[0]);

async function recognize(file) {
  $("upload-result").textContent = "Analyzing…";
  $("detect-list").innerHTML = "";
  const fd = new FormData();
  fd.append("file", file);
  const res = await api("/api/recognize", { method: "POST", body: fd });
  $("image-preview").innerHTML = `<img src="${res.annotated_image}" alt="result" />`;
  $("upload-result").textContent = `${res.count} plate(s) detected`;
  $("detect-hint").style.display = res.count ? "none" : "block";
  (res.plates || []).forEach(addDetectCard);
  (res.plates || []).forEach(handleAlert);
}

function addDetectCard(ev) {
  const card = document.createElement("div");
  card.className = "detect-card";
  const read = ev.plate_text || "";
  const known = read
    ? `<div class="row-plate">${esc(read)}</div>`
    : `<div class="row-plate muted">No OCR model — label it manually</div>`;
  card.innerHTML = `
    <div class="detect-head">
      ${known}
      <span class="conf">${Math.round((ev.detection_confidence || 0) * 100)}%</span>
    </div>
    <div class="label-row">
      <input class="lbl-plate" type="text" placeholder="Plate number" value="${esc(read)}" />
      <input class="lbl-name" type="text" placeholder="Owner / label" />
    </div>
    <div class="label-actions">
      <button class="btn sm add-white">＋ Whitelist</button>
      <button class="btn sm add-black">＋ Blacklist</button>
    </div>`;
  const plateIn = card.querySelector(".lbl-plate");
  const nameIn = card.querySelector(".lbl-name");
  const add = async (list) => {
    const plate = plateIn.value.trim();
    if (!plate) { toast("Enter a plate number first"); return; }
    const r = await fetch("/api/watchlist", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plate, name: nameIn.value.trim(), list_type: list }),
    });
    if (r.ok) toast(`Saved to ${list}list`, list === "white" ? "white" : "black");
    else toast("Could not save");
  };
  card.querySelector(".add-white").onclick = () => add("white");
  card.querySelector(".add-black").onclick = () => add("black");
  $("detect-list").appendChild(card);
}

/* ================= VIDEO STREAM ================= */
$("cam-dir-seg").querySelectorAll(".seg-btn").forEach((b) => {
  b.onclick = () => {
    $("cam-dir-seg").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.camDir = b.dataset.dir;
  };
});

$("btn-test-cam").onclick = async () => {
  const url = $("cam-url").value.trim();
  if (!url) { toast("Enter a stream URL"); return; }
  $("cam-test-result").textContent = "Testing connection…";
  $("cam-test-preview").innerHTML = "";
  $("btn-save-cam").disabled = true;
  try {
    const res = await api("/api/cameras/test", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    $("cam-test-result").innerHTML = res.ok
      ? `<span class="ok">✔ ${esc(res.message)}</span>`
      : `<span class="fail">✘ ${esc(res.message)}</span>`;
    if (res.ok) {
      state.lastTestUrl = url;
      $("btn-save-cam").disabled = false;
      if (res.preview) $("cam-test-preview").innerHTML = `<img src="${res.preview}" alt="preview" />`;
    }
  } catch (_) {
    $("cam-test-result").innerHTML = `<span class="fail">✘ Test failed</span>`;
  }
};

$("btn-save-cam").onclick = async () => {
  const url = $("cam-url").value.trim();
  const r = await fetch("/api/cameras", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: $("cam-name").value.trim(), url, direction: state.camDir }),
  });
  if (r.ok) {
    toast("Camera saved", "white");
    $("cam-name").value = ""; $("cam-url").value = "";
    $("cam-test-result").textContent = ""; $("cam-test-preview").innerHTML = "";
    $("btn-save-cam").disabled = true;
    loadCameras();
  } else toast("Could not save camera");
};

async function loadCameras() {
  const { cameras } = await api("/api/cameras");
  const list = $("camera-list");
  list.innerHTML = "";
  cameras.forEach((c) => list.appendChild(cameraRow(c)));
  $("camera-empty").style.display = cameras.length ? "none" : "block";
}

function cameraRow(c) {
  const row = document.createElement("div");
  row.className = "cam-row";
  const dirBadge = c.direction && c.direction !== "unknown"
    ? `<span class="badge ${c.direction}">${c.direction}</span>` : "";
  row.innerHTML = `
    <button class="cam-play" title="View">
      <svg viewBox="0 0 24 24" class="ic-s"><polygon points="6 4 20 12 6 20 6 4"/></svg>
    </button>
    <div class="row-main">
      <div class="cam-name">${esc(c.name)} ${dirBadge}</div>
      <div class="row-sub">${esc(c.url)}</div>
    </div>
    <button class="row-del" title="Remove">
      <svg viewBox="0 0 24 24" class="ic-s"><polyline points="3 6 5 6 21 6"/><path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a1 1 0 01-1 1H7a1 1 0 01-1-1V6"/></svg>
    </button>`;
  row.querySelector(".cam-play").onclick = () => startCamera(c);
  row.querySelector(".row-del").onclick = async () => {
    await fetch("/api/cameras/" + c.id, { method: "DELETE" });
    loadCameras();
  };
  return row;
}

async function startCamera(c) {
  await api("/api/stream/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: c.url, direction: c.direction }),
  });
  $("live").src = "/api/stream/mjpeg?t=" + Date.now();
  $("stream-current").textContent = `Viewing: ${c.name}`;
  refreshStatus();
}

$("btn-stop").onclick = async () => {
  await api("/api/stream/stop", { method: "POST" });
  $("live").removeAttribute("src"); // close the MJPEG connection
  $("stream-current").textContent = "No active stream";
  refreshStatus();
};

async function loadStreamLog() {
  const { events } = await api("/api/events?limit=25");
  const list = $("stream-log");
  list.innerHTML = "";
  events.forEach((ev) => list.appendChild(eventRow(ev)));
  $("stream-log-empty").style.display = events.length ? "none" : "block";
}

/* ================= WATCHLIST ================= */
$("watch-seg").querySelectorAll(".seg-btn").forEach((b) => {
  b.onclick = () => {
    $("watch-seg").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.watchList = b.dataset.list;
  };
});
$("watch-filters").querySelectorAll(".chip").forEach((c) => {
  c.onclick = () => {
    $("watch-filters").querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
    c.classList.add("active");
    state.watchFilter = c.dataset.wf;
    loadWatchlist();
  };
});
$("watch-search").addEventListener("input", (e) => { state.watchSearch = e.target.value.trim().toLowerCase(); loadWatchlist(); });

$("btn-add-watch").onclick = async () => {
  const plate = $("watch-plate").value.trim();
  if (!plate) { toast("Enter a plate number"); return; }
  const r = await fetch("/api/watchlist", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plate, name: $("watch-name").value.trim(), list_type: state.watchList }),
  });
  if (r.ok) {
    $("watch-plate").value = ""; $("watch-name").value = "";
    $("watch-result").textContent = "Added";
    toast("Plate added to " + state.watchList + "list", state.watchList);
    loadWatchlist();
  } else {
    const err = await r.json().catch(() => ({}));
    toast(err.detail || "Could not add plate");
  }
};

async function loadWatchlist() {
  const q = state.watchFilter === "all" ? "" : "?list_type=" + state.watchFilter;
  const { entries } = await api("/api/watchlist" + q);
  const s = state.watchSearch;
  const filtered = s
    ? entries.filter((e) => (e.plate_text + " " + (e.name || "")).toLowerCase().includes(s))
    : entries;
  const list = $("watch-list");
  list.innerHTML = "";
  filtered.forEach((e) => list.appendChild(watchRow(e)));
  $("watch-empty").style.display = filtered.length ? "none" : "block";
}

function watchRow(e) {
  const row = document.createElement("div");
  row.className = "row list-" + e.list_type;
  row.innerHTML = `
    <div class="row-main">
      <div class="row-plate">${esc(e.plate_text)}</div>
      <div class="row-sub"><span class="tag-name">${esc(e.name || "—")}</span></div>
    </div>
    <div class="row-side"><span class="badge ${e.list_type}">${e.list_type}</span></div>
    <button class="row-del" title="Remove">
      <svg viewBox="0 0 24 24" class="ic-s"><polyline points="3 6 5 6 21 6"/><path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a1 1 0 01-1 1H7a1 1 0 01-1-1V6"/></svg>
    </button>`;
  row.querySelector(".row-del").onclick = async () => {
    await fetch("/api/watchlist/" + e.id, { method: "DELETE" });
    loadWatchlist();
  };
  return row;
}

/* --- detection history advanced search --- */
$("hist-filters").querySelectorAll(".chip").forEach((c) => {
  c.onclick = () => {
    $("hist-filters").querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
    c.classList.add("active");
    state.histFilter = c.dataset.filter;
  };
});
$("btn-hist-search").onclick = runHistorySearch;
$("hist-search").addEventListener("keydown", (e) => { if (e.key === "Enter") runHistorySearch(); });
$("btn-hist-clear").onclick = () => {
  $("hist-search").value = ""; $("hist-from").value = ""; $("hist-to").value = "";
  $("hist-filters").querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
  $("hist-filters").querySelector('[data-filter="all"]').classList.add("active");
  state.histFilter = "all";
  runHistorySearch();
};

async function runHistorySearch() {
  const p = new URLSearchParams({ limit: "200" });
  const s = $("hist-search").value.trim();
  if (s) p.set("plate", s);
  const f = state.histFilter;
  if (f === "entry" || f === "exit") p.set("direction", f);
  if (f === "white" || f === "black") p.set("list_type", f);
  const from = $("hist-from").value, to = $("hist-to").value;
  if (from) p.set("date_from", from + "T00:00:00");
  if (to) p.set("date_to", to + "T23:59:59");
  const { events } = await api("/api/events?" + p.toString());
  const list = $("hist-list");
  list.innerHTML = "";
  events.forEach((ev) => list.appendChild(eventRow(ev)));
  $("hist-empty").style.display = events.length ? "none" : "block";
}

/* ---------- shared event row ---------- */
function eventRow(ev, flash = false) {
  const row = document.createElement("div");
  row.className = "row" + (ev.matched_list ? " list-" + ev.matched_list : "") + (flash ? " flash" : "");
  const snap = snapUrl(ev.snapshot_path);
  const dirBadge = ev.direction && ev.direction !== "unknown" ? `<span class="badge ${ev.direction}">${ev.direction}</span>` : "";
  const listBadge = ev.matched_list ? `<span class="badge ${ev.matched_list}">${esc(ev.matched_name || ev.matched_list)}</span>` : "";
  row.innerHTML = `
    ${snap ? `<img class="row-snap" src="${snap}" alt="" />` : `<div class="row-snap"></div>`}
    <div class="row-main">
      <div class="row-plate">${esc(ev.plate_text || ev.plate_text_fa || "—")}</div>
      <div class="row-sub">${Math.round((ev.score || 0) * 100)}% · ${esc((ev.source || "").slice(0, 26))}</div>
    </div>
    <div class="row-side">
      <div class="row-time">${fmtTime(ev.created_at)}</div>
      ${listBadge || dirBadge}
    </div>`;
  return row;
}

/* ================= STATS ================= */
async function loadStats() {
  const s = await api("/api/stats");
  const st = await api("/api/status");
  $("s-total").textContent = s.total_events;
  $("s-distinct").textContent = s.distinct_plates;
  $("s-entries").textContent = s.entries;
  $("s-exits").textContent = s.exits;
  $("s-white").textContent = s.whitelist_hits;
  $("s-black").textContent = s.blacklist_hits;
  $("s-engine").textContent = `${st.detector} / ${st.ocr}`;
  $("s-watch").textContent = `${s.watchlist_white} white · ${s.watchlist_black} black`;
  $("s-last").textContent = fmtTime(s.last_seen);
}

/* ---------- status + live socket ---------- */
function setStatus(running, label) {
  ["status-dot", "status-dot-side"].forEach((id) => { const el = $(id); if (el) el.className = "dot" + (running ? " live" : ""); });
  ["status-text", "status-text-side"].forEach((id) => { const el = $(id); if (el) el.textContent = label; });
}
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    setStatus(s.running, s.running ? "Live" : "Idle");
    $("video-fps").textContent = s.running ? `${s.fps} fps` : "— fps";
    $("video-idle").style.display = s.running ? "none" : "flex";
  } catch (_) {}
}
function handleAlert(ev) {
  if (ev.matched_list === "black") toast(`Blacklist alert — ${ev.plate_text} · ${ev.matched_name || "unknown"}`, "black");
  else if (ev.matched_list === "white") toast(`Access granted — ${ev.matched_name || ev.plate_text}`, "white");
}
function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    handleAlert(ev);
    if ($("view-stream").classList.contains("active")) {
      $("stream-log").prepend(eventRow(ev, true));
      $("stream-log-empty").style.display = "none";
    }
    refreshStatus();
  };
  ws.onclose = () => setTimeout(connectWs, 3000);
}

/* ---------- access gate ---------- */
const EMAIL_KEY = "platrix_access_email";
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function initGate() {
  const gate = $("gate");
  if (localStorage.getItem(EMAIL_KEY)) { gate.classList.add("hidden"); return; }
  gate.classList.remove("hidden");
  $("gate-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = $("gate-email").value.trim();
    if (!EMAIL_RE.test(email)) { $("gate-error").textContent = "Please enter a valid email address."; return; }
    $("gate-error").textContent = "";
    try {
      const r = await fetch("/api/access", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.detail || "failed"); }
      localStorage.setItem(EMAIL_KEY, email);
      gate.classList.add("hidden");
    } catch (err) {
      $("gate-error").textContent = "Could not verify email. Please try again.";
    }
  });
}

async function boot() {
  initGate();
  try {
    const h = await api("/api/health");
    $("s-version").textContent = h.version;
    const vs = $("s-version-side"); if (vs) vs.textContent = h.version;
  } catch (_) {}
  refreshStatus();
  connectWs();
  setInterval(refreshStatus, 4000);
}
boot();
