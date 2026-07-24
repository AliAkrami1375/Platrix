// Platrix — mobile-app dashboard client.
"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, opts).then((r) => r.json());

const state = {
  eventFilter: "all",
  watchFilter: "all",
  direction: "unknown",
  watchList: "white",
  search: "",
  events: [],
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
const titles = { live: "Live Monitor", events: "Detections", watch: "Watchlist", stats: "Statistics" };
const subtitles = { live: "Real-time recognition", events: "Detection log & search", watch: "White / black list", stats: "System overview" };

function switchView(view) {
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view)
  );
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $("view-" + view).classList.add("active");
  $("appbar-title").textContent = titles[view];
  $("appbar-sub").textContent = subtitles[view];
  if (view === "events") loadEvents();
  if (view === "watch") loadWatchlist();
  if (view === "stats") loadStats();
}
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.onclick = () => switchView(btn.dataset.view);
});

/* ---------- LIVE ---------- */
$("dir-seg").querySelectorAll(".seg-btn").forEach((b) => {
  b.onclick = () => {
    $("dir-seg").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.direction = b.dataset.dir;
  };
});

$("btn-start").onclick = async () => {
  const source = $("source-input").value.trim() || "0";
  try {
    await api("/api/stream/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, direction: state.direction }),
    });
    $("live").src = "/api/stream/mjpeg?t=" + Date.now();
    refreshStatus();
  } catch (_) {
    toast("Could not start source");
  }
};

$("btn-stop").onclick = async () => {
  await api("/api/stream/stop", { method: "POST" });
  refreshStatus();
};

/* image upload */
const dz = $("dropzone");
["dragover", "dragenter"].forEach((e) =>
  dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("drag"); })
);
["dragleave", "drop"].forEach((e) =>
  dz.addEventListener(e, () => dz.classList.remove("drag"))
);
dz.addEventListener("drop", (ev) => {
  ev.preventDefault();
  if (ev.dataTransfer.files[0]) recognize(ev.dataTransfer.files[0]);
});
$("file-input").onchange = (e) => e.target.files[0] && recognize(e.target.files[0]);

async function recognize(file) {
  $("upload-result").textContent = "Analyzing…";
  const fd = new FormData();
  fd.append("file", file);
  const res = await api(`/api/recognize?direction=${state.direction}`, { method: "POST", body: fd });
  $("upload-result").textContent = `${res.count} plate(s) found`;
  $("upload-preview").innerHTML = `<img src="${res.annotated_image}" alt="result" />`;
  (res.plates || []).forEach(handleAlert);
}

/* ---------- EVENTS ---------- */
$("event-filters").querySelectorAll(".chip").forEach((c) => {
  c.onclick = () => {
    $("event-filters").querySelectorAll(".chip").forEach((x) => x.classList.remove("active"));
    c.classList.add("active");
    state.eventFilter = c.dataset.filter;
    loadEvents();
  };
});

let searchTimer;
$("search-input").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  state.search = e.target.value.trim();
  searchTimer = setTimeout(loadEvents, 250);
});

function eventQuery() {
  const p = new URLSearchParams({ limit: "150" });
  if (state.search) p.set("plate", state.search);
  const f = state.eventFilter;
  if (f === "entry" || f === "exit") p.set("direction", f);
  if (f === "white" || f === "black") p.set("list_type", f);
  return p.toString();
}

async function loadEvents() {
  const { events } = await api("/api/events?" + eventQuery());
  const list = $("events-list");
  list.innerHTML = "";
  events.forEach((ev) => list.appendChild(eventRow(ev)));
  $("events-empty").style.display = events.length ? "none" : "block";
}

function eventRow(ev, flash = false) {
  const row = document.createElement("div");
  row.className = "row" + (ev.matched_list ? " list-" + ev.matched_list : "") + (flash ? " flash" : "");
  const snap = snapUrl(ev.snapshot_path);
  const dirBadge = ev.direction && ev.direction !== "unknown"
    ? `<span class="badge ${ev.direction}">${ev.direction}</span>` : "";
  const listBadge = ev.matched_list
    ? `<span class="badge ${ev.matched_list}">${ev.matched_name || ev.matched_list}</span>` : "";
  row.innerHTML = `
    ${snap ? `<img class="row-snap" src="${snap}" alt="" />` : `<div class="row-snap"></div>`}
    <div class="row-main">
      <div class="row-plate">${ev.plate_text || ev.plate_text_fa || "—"}</div>
      <div class="row-sub">${Math.round((ev.score || 0) * 100)}% · ${(ev.source || "").slice(0, 26)}</div>
    </div>
    <div class="row-side">
      <div class="row-time">${fmtTime(ev.created_at)}</div>
      ${listBadge || dirBadge}
    </div>`;
  return row;
}

/* ---------- WATCHLIST ---------- */
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

$("btn-add-watch").onclick = async () => {
  const plate = $("watch-plate").value.trim();
  if (!plate) { toast("Enter a plate number", "black"); return; }
  const res = await fetch("/api/watchlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      plate,
      name: $("watch-name").value.trim(),
      list_type: state.watchList,
    }),
  });
  if (res.ok) {
    $("watch-plate").value = "";
    $("watch-name").value = "";
    $("watch-result").textContent = "Added ✓";
    toast("Plate added to " + state.watchList + "list", state.watchList);
    loadWatchlist();
  } else {
    const err = await res.json().catch(() => ({}));
    toast(err.detail || "Could not add plate");
  }
};

async function loadWatchlist() {
  const q = state.watchFilter === "all" ? "" : "?list_type=" + state.watchFilter;
  const { entries } = await api("/api/watchlist" + q);
  const list = $("watch-list");
  list.innerHTML = "";
  entries.forEach((e) => list.appendChild(watchRow(e)));
  $("watch-empty").style.display = entries.length ? "none" : "block";
}

function watchRow(e) {
  const row = document.createElement("div");
  row.className = "row list-" + e.list_type;
  row.innerHTML = `
    <div class="row-main">
      <div class="row-plate">${e.plate_text}</div>
      <div class="row-sub"><span class="tag-name">${e.name || "—"}</span> · ${e.note || ""}</div>
    </div>
    <div class="row-side">
      <span class="badge ${e.list_type}">${e.list_type}</span>
    </div>
    <button class="row-del" title="Remove" aria-label="Remove">
      <svg viewBox="0 0 24 24" class="ic-s"><polyline points="3 6 5 6 21 6"/><path d="M8 6V4a1 1 0 011-1h6a1 1 0 011 1v2m2 0v14a1 1 0 01-1 1H7a1 1 0 01-1-1V6"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
    </button>`;
  row.querySelector(".row-del").onclick = async () => {
    await fetch("/api/watchlist/" + e.id, { method: "DELETE" });
    loadWatchlist();
  };
  return row;
}

/* ---------- STATS ---------- */
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
  ["status-dot", "status-dot-side"].forEach((id) => {
    const el = $(id);
    if (el) el.className = "dot" + (running ? " live" : "");
  });
  ["status-text", "status-text-side"].forEach((id) => {
    const el = $(id);
    if (el) el.textContent = label;
  });
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
  if (ev.matched_list === "black")
    toast(`Blacklist alert — ${ev.plate_text} · ${ev.matched_name || "unknown"}`, "black");
  else if (ev.matched_list === "white")
    toast(`Access granted — ${ev.matched_name || ev.plate_text}`, "white");
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    handleAlert(ev);
    // Prepend to events view if visible & unfiltered enough.
    if ($("view-events").classList.contains("active")) {
      const list = $("events-list");
      list.prepend(eventRow(ev, true));
      $("events-empty").style.display = "none";
    }
  };
  ws.onclose = () => setTimeout(connectWs, 3000);
}

async function boot() {
  try {
    const h = await api("/api/health");
    $("s-version").textContent = h.version;
    const vs = $("s-version-side");
    if (vs) vs.textContent = h.version;
  } catch (_) {}
  refreshStatus();
  connectWs();
  setInterval(refreshStatus, 4000);
}
boot();
