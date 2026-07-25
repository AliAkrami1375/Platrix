// Platrix dashboard client.
"use strict";

const $ = (id) => document.getElementById(id);

// All API calls carry the session token as a Bearer header (token-based API).
const TOKEN_KEY = "platrix_token";
const getToken = () => localStorage.getItem(TOKEN_KEY) || "";
function afetch(url, opts = {}) {
  const headers = Object.assign({}, opts.headers);
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  return fetch(url, Object.assign({}, opts, { headers }));
}
const api = (path, opts) => afetch(path, opts).then((r) => r.json());
// Append the token to <img> URLs (stream / snapshots can't send headers).
const withToken = (url) => url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(getToken());
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
  return i >= 0 ? withToken("/" + path.slice(i)) : null;
}
function toast(msg, kind = "black") {
  const el = $("toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.className = "toast"), 3500);
}

/* ---------- navigation ---------- */
const titles = { image: "Image Detection", stream: "Video Stream", watch: "Watchlist", stats: "Statistics", learn: "Learn / Train" };
const subtitles = { image: "Upload & recognize", stream: "Cameras & live view", watch: "Lists & history search", stats: "System overview", learn: "Teach it new plates" };
function switchView(view) {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("view-" + view).classList.add("active");
  $("appbar-title").textContent = titles[view];
  $("appbar-sub").textContent = subtitles[view];
  if (view === "stream") { loadCameras(); loadStreamLog(); }
  if (view === "watch") { loadWatchlist(); runHistorySearch(); }
  if (view === "stats") loadStats();
  if (view === "learn") initLearn();
}

/* ================= LEARN / TRAIN ================= */
const learn = { img: null, file: null, box: null, drawing: false, dev: "auto", poll: null, inited: false };

function initLearn() {
  loadGpu();
  loadLearnSamples();
  pollTraining();  // resume showing progress if a job is already running
  if (learn.inited) return;
  learn.inited = true;

  $("learn-file").onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    learn.file = f;
    const img = new Image();
    img.onload = () => {
      learn.img = img; learn.box = null;
      const cv = $("learn-canvas");
      const scale = Math.min(1, 900 / img.width);
      cv.width = img.width * scale; cv.height = img.height * scale;
      drawCanvas();
      $("canvas-wrap").classList.remove("hidden");
    };
    img.src = URL.createObjectURL(f);
  };

  const cv = $("learn-canvas");
  const pos = (e) => {
    const r = cv.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return { x: cx * cv.width / r.width, y: cy * cv.height / r.height };
  };
  const down = (e) => { e.preventDefault(); const p = pos(e); learn.box = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; learn.drawing = true; };
  const move = (e) => { if (!learn.drawing) return; const p = pos(e); learn.box.x1 = p.x; learn.box.y1 = p.y; drawCanvas(); };
  const up = () => { learn.drawing = false; };
  cv.onmousedown = down; cv.onmousemove = move; cv.onmouseup = up;
  cv.ontouchstart = down; cv.ontouchmove = move; cv.ontouchend = up;

  $("learn-add").onclick = addLearnSample;

  $("dev-seg").querySelectorAll(".seg-btn").forEach((b) => {
    b.onclick = () => {
      $("dev-seg").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
      b.classList.add("active"); learn.dev = b.dataset.dev;
      $("cuda-row").style.display = learn.dev === "gpu" ? "flex" : "none";
    };
  });
  $("learn-start").onclick = startTraining;
  $("train-apply").onclick = async () => { await afetch("/api/learn/apply", { method: "POST" }); toast("New model applied", "white"); };
}

function drawCanvas() {
  const cv = $("learn-canvas"), ctx = cv.getContext("2d");
  ctx.drawImage(learn.img, 0, 0, cv.width, cv.height);
  if (learn.box) {
    const b = learn.box;
    ctx.strokeStyle = "#2f81f7"; ctx.lineWidth = 2;
    ctx.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
    ctx.fillStyle = "rgba(47,129,247,0.15)";
    ctx.fillRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
  }
}

async function addLearnSample() {
  const plate = $("learn-plate").value.trim();
  if (!learn.file || !learn.box || !plate) { toast("Draw a box and type the plate"); return; }
  const cv = $("learn-canvas");
  const b = learn.box;
  const x = Math.min(b.x0, b.x1) / cv.width, y = Math.min(b.y0, b.y1) / cv.height;
  const w = Math.abs(b.x1 - b.x0) / cv.width, h = Math.abs(b.y1 - b.y0) / cv.height;
  if (w < 0.02 || h < 0.02) { toast("Box too small"); return; }
  const fd = new FormData();
  fd.append("file", learn.file); fd.append("plate", plate);
  fd.append("x", x); fd.append("y", y); fd.append("w", w); fd.append("h", h);
  const r = await afetch("/api/learn/samples", { method: "POST", body: fd });
  if (r.ok) {
    toast("Sample added", "white");
    $("learn-plate").value = ""; learn.box = null;
    $("canvas-wrap").classList.add("hidden"); $("learn-file").value = "";
    loadLearnSamples();
  } else toast("Could not save sample");
}

async function loadLearnSamples() {
  const { samples } = await api("/api/learn/samples");
  $("sample-count").textContent = samples.length;
  const grid = $("learn-samples"); grid.innerHTML = "";
  samples.forEach((s) => {
    const el = document.createElement("div");
    el.className = "learn-samp";
    el.innerHTML = `
      <img src="${withToken("/learn-media/" + s.image_file)}" alt="" />
      <div class="lab">${esc(s.plate_text)}</div>
      <button class="del">✕</button>`;
    el.querySelector(".del").onclick = async () => {
      await afetch("/api/learn/samples/" + s.id, { method: "DELETE" });
      loadLearnSamples();
    };
    grid.appendChild(el);
  });
  $("learn-empty").style.display = samples.length ? "none" : "block";
}

async function loadGpu() {
  try {
    const g = await api("/api/system/gpu");
    const el = $("gpu-banner");
    el.classList.toggle("has-gpu", g.has_gpu);
    el.innerHTML = g.has_gpu
      ? `🖥️ GPU: <b>${esc(g.name || "detected")}</b>${g.driver ? " · driver " + esc(g.driver) : ""} · ${g.cuda_available ? "CUDA ready" : "CUDA not installed"}`
      : `💻 No GPU detected — training runs on CPU.`;
  } catch (_) {}
}

async function startTraining() {
  const epochs = parseInt($("learn-epochs").value) || 15;
  const r = await afetch("/api/learn/train", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs, device: learn.dev, install_cuda: $("learn-cuda").checked }),
  });
  const d = await r.json().catch(() => ({}));
  if (r.ok) { toast("Training started", "white"); $("train-progress").classList.remove("hidden"); pollTraining(); }
  else toast(d.detail || "Could not start training");
}

async function pollTraining() {
  if (learn.poll) { clearInterval(learn.poll); learn.poll = null; }
  const tick = async () => {
    let s;
    try { s = await api("/api/learn/status"); } catch (_) { return; }
    if (!s || s.status === "idle") { $("train-progress").classList.add("hidden"); return; }
    $("train-progress").classList.remove("hidden");
    $("train-step").textContent = s.step || s.status;
    $("train-pct").textContent = (s.progress || 0) + "%";
    $("train-fill").style.width = (s.progress || 0) + "%";
    $("train-device").textContent = "device: " + (s.device || "—");
    $("train-acc").textContent = s.accuracy != null ? "accuracy: " + Math.round(s.accuracy * 100) + "%" : "";
    $("train-log").textContent = (s.log || []).slice(-40).join("\n");
    $("train-log").scrollTop = $("train-log").scrollHeight;
    const done = s.status === "done" || s.status === "error";
    $("train-apply").classList.toggle("hidden", s.status !== "done");
    if (s.status === "error") toast(s.message || "Training failed");
    if (done && learn.poll) { clearInterval(learn.poll); learn.poll = null; loadLearnSamples(); }
  };
  tick();
  learn.poll = setInterval(tick, 2000);
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
      <input class="lbl-plate" dir="ltr" type="text" placeholder="Plate number" value="${esc(read)}" />
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
    const r = await afetch("/api/watchlist", {
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
  const r = await afetch("/api/cameras", {
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
  const stateColors = { online: "var(--white-c)", reconnecting: "var(--exit-c)", connecting: "var(--accent)", error: "var(--black-c)" };
  const dotColor = stateColors[c.status] || "var(--muted)";
  const sub = `${esc(c.url)} · ${c.status || "off"}${c.live_fps ? " · " + c.live_fps + "fps" : ""}`;
  row.innerHTML = `
    <button class="cam-play" title="View live">
      <svg viewBox="0 0 24 24" class="ic-s"><polygon points="6 4 20 12 6 20 6 4"/></svg>
    </button>
    <div class="row-main">
      <div class="cam-name"><span class="cam-dot" style="background:${dotColor}"></span>${esc(c.name)} ${dirBadge}</div>
      <div class="row-sub">${sub}</div>
    </div>
    <label class="switch" title="Always-on monitoring">
      <input type="checkbox" ${c.enabled ? "checked" : ""} /><span class="slider"></span>
    </label>
    <button class="row-del" title="Remove">
      <svg viewBox="0 0 24 24" class="ic-s"><polyline points="3 6 5 6 21 6"/><path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a1 1 0 01-1 1H7a1 1 0 01-1-1V6"/></svg>
    </button>`;
  row.querySelector(".cam-play").onclick = () => viewCamera(c);
  row.querySelector(".switch input").onchange = (e) => toggleCamera(c.id, e.target.checked);
  row.querySelector(".row-del").onclick = async () => {
    await afetch("/api/cameras/" + c.id, { method: "DELETE" });
    loadCameras();
  };
  return row;
}

async function toggleCamera(id, enabled) {
  await afetch("/api/cameras/" + id, {
    method: "PATCH", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  toast(enabled ? "Camera is now always-on" : "Camera monitoring stopped", enabled ? "white" : "black");
  setTimeout(loadCameras, 400);
}

async function viewCamera(c) {
  if (!c.enabled) {
    // ad-hoc view for a camera that isn't in always-on mode
    await api("/api/stream/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: c.url, direction: c.direction }),
    });
    $("live").src = withToken("/api/stream/mjpeg?t=" + Date.now());
  } else {
    $("live").src = withToken(`/api/stream/mjpeg?camera=${c.id}&t=` + Date.now());
  }
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
  const r = await afetch("/api/watchlist", {
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
    await afetch("/api/watchlist/" + e.id, { method: "DELETE" });
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
    const active = s.active_cameras || 0;
    setStatus(s.running, s.running ? (active > 1 ? `${active} cameras` : "Live") : "Idle");
    $("video-fps").textContent = s.running ? `${s.fps} fps` : "— fps";
    $("video-idle").style.display = s.running ? "none" : "flex";
    // keep camera connection dots fresh while viewing the Stream tab
    if ($("view-stream").classList.contains("active")) loadCameras();
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

/* ---------- login gate ---------- */
async function initGate() {
  const gate = $("gate");
  let authed = false;
  try {
    const me = await api("/api/me");
    authed = !me.auth_enabled || me.authenticated;
  } catch (_) {}
  if (authed) { gate.classList.add("hidden"); startApp(); return; }

  gate.classList.remove("hidden");
  $("gate-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("gate-error").textContent = "";
    const r = await afetch("/api/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("gate-user").value.trim(), password: $("gate-pass").value }),
    });
    if (r.ok) {
      const d = await r.json();
      if (d.token) localStorage.setItem(TOKEN_KEY, d.token);
      gate.classList.add("hidden"); startApp();
    } else $("gate-error").textContent = "Invalid username or password.";
  });
}

const _logout = $("btn-logout");
if (_logout) _logout.onclick = async () => {
  await afetch("/api/logout", { method: "POST" });
  localStorage.removeItem(TOKEN_KEY);
  location.reload();
};

/* ---------- settings modal ---------- */
const _settings = $("btn-settings");
if (_settings) _settings.onclick = () => { $("settings").classList.remove("hidden"); loadTokens(); };
if ($("settings-close")) $("settings-close").onclick = () => $("settings").classList.add("hidden");
if ($("settings")) $("settings").addEventListener("click", (e) => { if (e.target.id === "settings") $("settings").classList.add("hidden"); });

if ($("btn-save-creds")) $("btn-save-creds").onclick = async () => {
  const body = {
    new_username: $("set-user").value.trim(),
    current_password: $("set-cur").value,
    new_password: $("set-new").value,
  };
  const r = await afetch("/api/account/password", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (r.ok) {
    if (d.token) localStorage.setItem(TOKEN_KEY, d.token);  // username may have changed
    $("creds-msg").textContent = "Credentials updated ✓";
    $("set-cur").value = ""; $("set-new").value = "";
    toast("Credentials updated", "white");
  } else $("creds-msg").textContent = d.detail || "Could not update.";
};

async function loadTokens() {
  const { tokens } = await api("/api/tokens");
  const list = $("token-list");
  list.innerHTML = "";
  tokens.forEach((t) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `
      <div class="row-main">
        <div class="row-plate" style="font-size:14px">${esc(t.name)}</div>
        <div class="row-sub"><code>${esc(t.prefix)}…</code> · created ${fmtTime(t.created_at)}${t.last_used_at ? " · used " + fmtTime(t.last_used_at) : ""}</div>
      </div>
      <button class="row-del" title="Revoke">
        <svg viewBox="0 0 24 24" class="ic-s"><polyline points="3 6 5 6 21 6"/><path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a1 1 0 01-1 1H7a1 1 0 01-1-1V6"/></svg>
      </button>`;
    row.querySelector(".row-del").onclick = async () => {
      await afetch("/api/tokens/" + t.id, { method: "DELETE" });
      loadTokens();
    };
    list.appendChild(row);
  });
  $("token-empty").style.display = tokens.length ? "none" : "block";
}

if ($("btn-create-token")) $("btn-create-token").onclick = async () => {
  const r = await afetch("/api/tokens", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: $("tok-name").value.trim() || "token" }),
  });
  const d = await r.json();
  if (r.ok && d.token) {
    $("tok-raw").textContent = d.token;
    $("tok-new").classList.remove("hidden");
    $("tok-name").value = "";
    loadTokens();
  }
};
if ($("btn-copy-token")) $("btn-copy-token").onclick = () => {
  const t = $("tok-raw").textContent;
  navigator.clipboard?.writeText(t);
  toast("Token copied", "white");
};

/* export the current detection-history query as CSV */
const _export = $("btn-hist-export");
if (_export) _export.onclick = () => {
  const p = new URLSearchParams();
  const s = $("hist-search").value.trim(); if (s) p.set("plate", s);
  const f = state.histFilter;
  if (f === "entry" || f === "exit") p.set("direction", f);
  if (f === "white" || f === "black") p.set("list_type", f);
  const from = $("hist-from").value, to = $("hist-to").value;
  if (from) p.set("date_from", from + "T00:00:00");
  if (to) p.set("date_to", to + "T23:59:59");
  window.location = "/api/events/export?" + p.toString();
};

let _appStarted = false;
function startApp() {
  if (_appStarted) return;
  _appStarted = true;
  refreshStatus();
  connectWs();
  setInterval(refreshStatus, 4000);
}

async function boot() {
  try {
    const h = await api("/api/health");
    $("s-version").textContent = h.version;
    const vs = $("s-version-side"); if (vs) vs.textContent = h.version;
  } catch (_) {}
  await initGate();   // shows login or, if authed, calls startApp()
}
boot();
