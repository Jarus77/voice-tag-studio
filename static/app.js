/* voice-tag-studio — single-page UI. Vanilla JS, no build. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  vid: null, job: null, pollTimer: null, pollFails: 0,
  mix: new Set(),      // lane names currently audible (mixed together)
  tracks: new Map(),   // name -> {path, peaks, canvas, row, audio}
  segments: [], transcripts: {}, alignments: {}, candidates: [],
  loaded: {},          // manifest name -> true once fetched
  playStopAt: null,    // for ±2s tag playback windows
  spkColor: {},
  optimistic: {},      // stage -> "queued" set right after a click, until poll confirms
};
const COLORS = ["#4da3ff", "#3ecf8e", "#f0b13c", "#ef6461", "#b48ead", "#7fd1d8"];
const STAGE_EXPLAIN = {
  ingest: "download audio", diarize: "who speaks when", denoise: "demucs vocals",
  segment: "2–20s utterances", asr: "gemini verbatim", align: "word timings",
};

/* ---------------- boot ---------------- */

async function boot() {
  try {
    const pf = await (await fetch("/api/preflight")).json();
    const bad = Object.entries(pf.checks).filter(([, c]) => !c.ok);
    if (!pf.ok) banner("<b>preflight problems:</b><br>" +
      bad.map(([n, c]) => `${n}: ${c.msg}`).join("<br>"), false);
  } catch { /* handled by poll banner */ }
  await refreshJobList();
  const pick = $("jobPicker");
  if (pick.options.length) selectJob(pick.options[pick.options.length - 1].value);
  else {
    $("jobTitle").textContent = "paste a YouTube URL above to start";
    renderStepper();  // empty skeleton
  }
}

function banner(html, ok) {
  const el = $("banner");
  el.classList.remove("hidden");
  el.classList.toggle("ok", !!ok);
  el.innerHTML = html;
}
function hideBanner() { $("banner").classList.add("hidden"); }

async function refreshJobList() {
  const jobs = await (await fetch("/api/jobs")).json();
  const pick = $("jobPicker");
  const cur = state.vid;
  pick.innerHTML = "";
  jobs.forEach((j) => {
    const o = document.createElement("option");
    o.value = j.video_id;
    o.textContent = (j.title || j.video_id).slice(0, 40);
    pick.appendChild(o);
  });
  if (cur) pick.value = cur;
}

$("urlForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const url = $("urlInput").value.trim();
  if (!url) return;
  const btn = $("runBtn");
  btn.disabled = true; btn.textContent = "starting…";
  try {
    const r = await fetch("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!r.ok) { banner("could not start job: " + ((await r.json()).detail || r.status)); return; }
    const d = await r.json();
    $("urlInput").value = "";
    await refreshJobList();
    selectJob(d.video_id);
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
});

$("jobPicker").addEventListener("change", (ev) => {
  if (ev.target.value) selectJob(ev.target.value);
});

$("showWordTimes").addEventListener("change", () => renderTranscript());

$("deleteJob").addEventListener("click", async () => {
  if (!state.vid) return;
  const title = state.job?.title || state.vid;
  if (!confirm(`Delete "${title}"?\n\nRemoves its audio, speaker lanes, clean ` +
               `lanes, segments, transcripts and manifests. This cannot be undone.`))
    return;
  const btn = $("deleteJob");
  btn.disabled = true;
  const r = await fetch(`/api/jobs/${state.vid}`, { method: "DELETE" });
  btn.disabled = false;
  if (!r.ok) { banner("delete failed: " + ((await r.json()).detail || r.status)); return; }
  const d = await r.json();
  banner(`deleted ${d.deleted} — freed ${d.freed_mb} MB`, true);
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.vid = null; state.job = null;
  state.tracks.forEach((tr) => tr.audio.pause());
  state.tracks.clear();
  ["tracks", "transcript", "hits", "bakeResults"].forEach((id) => ($(id).innerHTML = ""));
  await refreshJobList();
  const pick = $("jobPicker");
  if (pick.options.length) selectJob(pick.options[pick.options.length - 1].value);
  else ["listenSec", "transcriptSec", "tagsSec", "bakeoffSec"]
    .forEach((s) => $(s).classList.add("hidden"));
});

$("uploadBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const btn = $("uploadBtn");
  btn.disabled = true; btn.textContent = "uploading…";
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok) { banner("upload failed: " + ((await r.json()).detail || r.status)); return; }
    const d = await r.json();
    await refreshJobList();
    selectJob(d.video_id);
  } finally {
    btn.disabled = false; btn.textContent = "Upload audio";
    ev.target.value = "";
  }
});

$("runRemaining").addEventListener("click", async () => {
  if (!state.vid || !state.job) return;
  for (const [name, st] of Object.entries(state.job.stages)) {
    if (["done", "running", "queued"].includes(st.status)) continue;
    state.optimistic[name] = true;
    await fetch(`/api/jobs/${state.vid}/stages/${name}?force=false`, { method: "POST" });
  }
  renderStepper();
  poll();
});

/* ---------------- clean-lane builds (SepFormer overlap rescue) ---------------- */

async function isolateSpeaker(speaker) {
  const nOvWin = new Set((state.job.overlaps || []).map(([a]) => Math.floor(a / 10))).size;
  if (!confirm(
    `Build a CLEAN full-length ${speaker.replace("SPEAKER_", "S")} lane?\n\n` +
    `Original audio where they speak alone · SepFormer separation at the ` +
    `~${nOvWin} overlap window${nOvWin === 1 ? "" : "s"} · silence elsewhere.\n\n` +
    `Runs locally (free). Segment then cuts from this lane and keeps the ` +
    `rescued overlap seconds.`)) return;
  const r = await fetch(`/api/jobs/${state.vid}/separate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker, full: true, engine: "sepformer" }),
  });
  if (!r.ok) banner("clean lane: " + ((await r.json()).detail || r.status));
  poll();
}

function renderSepStatus() {
  const d = (state.job.detectors || {}).separate;
  const lane = $("cleanStatus");
  const busy = d && (d.status === "running" || d.status === "queued");
  if (busy || (d && d.status === "error")) {
    lane.classList.remove("hidden");
    const elapsed = d.status === "running" && d.started
      ? ` · ${Math.max(0, Math.round(Date.now() / 1000 - d.started))}s` : "";
    lane.innerHTML = busy
      ? `<span class="icon"></span><span>building clean lane — ${d.msg || d.status}${elapsed}</span>`
      : `<span>✗ clean-lane build failed: ${d.msg || "unknown"} — click <b>clean</b> to retry</span>`;
  } else lane.classList.add("hidden");
  document.querySelectorAll(".samiso").forEach((b) => (b.disabled = !!busy));
}

// diarizer picker (used when the diarize stage is rerun)
(async () => {
  try {
    const models = await (await fetch("/api/diarizers")).json();
    const pick = $("diarPicker");
    models.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.label;
      pick.appendChild(o);
    });
  } catch {}
})();

function selectJob(vid) {
  state.vid = vid;
  state.loaded = {}; state.optimistic = {};
  state.segments = []; state.transcripts = {}; state.alignments = {}; state.candidates = [];
  state.tracks.forEach((tr) => tr.audio.pause());
  state.tracks.clear();
  state.mix = new Set();
  player.removeAttribute("src");
  $("tracks").innerHTML = "";
  $("transcript").innerHTML = "";
  $("hits").innerHTML = "";
  $("jobPicker").value = vid;
  poll();
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 1500);
}

/* ---------------- polling ---------------- */

async function poll() {
  if (!state.vid) return;
  let j;
  try {
    const r = await fetch(`/api/jobs/${state.vid}`);
    if (!r.ok) throw new Error(r.status);
    j = await r.json();
    if (state.pollFails > 2) hideBanner();
    state.pollFails = 0;
  } catch {
    if (++state.pollFails === 3)
      banner("⚠ server unreachable — is <span style='font-family:var(--mono)'>python server.py</span> running? retrying…");
    return;
  }
  state.job = j;
  // clear optimistic flags once the server confirms
  for (const s of Object.keys(state.optimistic))
    if (["queued", "running"].includes(j.stages[s]?.status)) delete state.optimistic[s];
  $("jobTitle").textContent = j.title || j.url || "";
  renderStepper();
  renderActivity();
  renderTracks();
  await loadManifests();
  renderDetectors();
  renderSepStatus();
  renderBakeStatus();
  renderEmptyStates();
}

/* ---------------- pipeline stepper ---------------- */

const ICONS = { done: "✓", error: "✗", queued: "⋯", pending: "○", running: "" };

function renderStepper() {
  const el = $("stepper");
  el.innerHTML = "";
  const stages = state.job?.stages ||
    Object.fromEntries(Object.keys(STAGE_EXPLAIN).map((k) => [k, { status: "pending" }]));
  let firstError = null;
  for (const [name, st] of Object.entries(stages)) {
    let status = st.status || "pending";
    if (state.optimistic[name]) status = "queued";
    const step = document.createElement("div");
    step.className = `step ${status}`;
    const pct = status === "running" ? Math.round((st.progress || 0) * 100) : null;
    // the segment step announces WHICH audio it will cut from, per speaker
    let idle = STAGE_EXPLAIN[name] || "";
    if (name === "segment" && state.job?.stages?.diarize?.status === "done") {
      const have = cleanSpeakers();
      const spks = (state.job.sources || [])
        .filter((s) => s.name.startsWith("SPEAKER_")).map((s) => s.name);
      if (spks.length)
        idle = "cuts " + spks.map((s) =>
          `${s.replace("SPEAKER_", "S")}:${have.has(s) ? "clean" : "orig"}`).join(" ");
    }
    step.innerHTML =
      `<span class="icon">${ICONS[status] ?? "○"}</span>` +
      `<span><span class="name">${name}</span>` +
      `<span class="sub">${status === "running" ? (st.msg || pct + "%")
        : status === "queued" ? "queued"
        : status === "error" ? "failed — click to retry"
        : status === "pending" ? `▶ ${idle || "click to run"}`
        : idle}</span></span>` +
      (status === "running" ? `<span class="stepbar"><i style="width:${pct}%"></i></span>` : "");
    step.title = st.msg || `${name}: ${status}` + " (click to rerun)";
    step.onclick = () => rerunStage(name, status);
    el.appendChild(step);
    if (!firstError && status === "error") firstError = { name, msg: st.msg };
  }
  const errEl = $("stageError");
  if (firstError) {
    errEl.classList.remove("hidden");
    errEl.innerHTML = `<b>${firstError.name} failed:</b> ${firstError.msg || "unknown error"}` +
      ` — <span class="hint">click the step to retry</span>`;
  } else errEl.classList.add("hidden");
}

async function rerunStage(name, status) {
  if (!state.vid || status === "running" || status === "queued") return;
  if (status === "done" && !confirm(`rerun "${name}"? (its output will be rebuilt)`)) return;
  state.optimistic[name] = true;
  renderStepper();
  let q = `force=true`;
  if (name === "diarize" && $("diarPicker").value)
    q += `&engine=${encodeURIComponent($("diarPicker").value)}`;
  if (name === "segment") q += `&guard=${parseFloat($("guardInput").value || "0.15")}`;
  await fetch(`/api/jobs/${state.vid}/stages/${name}?${q}`, { method: "POST" });
  poll();
}

function renderActivity() {
  const el = $("activity");
  const a = state.job?.activity;
  if (!a || (!a.name && !a.queue_len)) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  let txt = "";
  if (a.name) {
    const where = a.video_id === state.vid ? "" :
      ` on <b>${a.video_id}</b> (another job — this one waits its turn)`;
    const pct = a.progress != null ? ` · ${Math.round(a.progress * 100)}%` : "";
    txt = `runner: <b>${a.name}</b>${where}${pct}${a.msg ? " · " + a.msg : ""}`;
  } else txt = "runner busy";
  if (a.queue_len) txt += ` · ${a.queue_len} task${a.queue_len > 1 ? "s" : ""} queued`;
  el.innerHTML = `<span class="icon"></span><span>${txt}</span>`;
}

/* ---------------- listen: stacked multitrack timeline ---------------- */

function spkColor(spk) {
  if (!state.spkColor[spk])
    state.spkColor[spk] = COLORS[Object.keys(state.spkColor).length % COLORS.length];
  return state.spkColor[spk];
}

function laneInfo(name) {
  // friendly labels + lane accent colors, Meta-editor style
  if (name === "original") return { label: "Original sound", color: "#3ecf8e" };
  if (name === "denoised") return { label: "Denoised (demucs)", color: "#4da3ff" };
  let m = name.match(/^sam_speaker_(\d+)_clean(?:_(\w+))?$/);
  if (m) return { label: `Clean S${m[1]}`, color: "#ffd166" };
  m = name.match(/^sam_(.+)_clean(?:_(\w+))?$/);
  if (m) return { label: `Clean ${m[1].replace(/_/g, " ")}`, color: "#ffd166" };
  if (name.startsWith("SPEAKER_"))
    return { label: name.replace("SPEAKER_", "S"), color: spkColor(name) };
  return { label: name, color: "#8595a5" };
}

function renderTracks() {
  ["listenSec", "transcriptSec", "tagsSec"].forEach((s) => $(s).classList.remove("hidden"));
  const holder = $("tracks");
  const sources = state.job.sources || [];
  if (!sources.length) {
    const ing = state.job.stages?.ingest;
    holder.innerHTML = `<p class="empty" style="padding:10px">no audio yet — ingest is ${ing?.status || "pending"}${state.job.activity && state.job.activity.video_id !== state.vid ? " (waiting for the runner to finish the other job)" : ""}…</p>`;
    return;
  }
  if (holder.querySelector("p")) holder.innerHTML = "";
  sources.forEach((s) => {
    if (state.tracks.has(s.name)) return;
    const info = laneInfo(s.name);
    const row = document.createElement("div");
    row.className = "track";
    const label = document.createElement("div");
    label.className = "tlabel";
    const isSpk = s.name.startsWith("SPEAKER_");
    label.innerHTML =
      `<button class="mute" title="toggle this lane in the mix">🔇</button>` +
      `<span class="lname" style="color:${info.color}" ${isSpk ? 'title="click to name this voice (e.g. agent / khushi) — pyannote labels are arbitrary"' : ""}>` +
      `${isSpk ? speakerLabel(s.name) : info.label}</span>` +
      (isSpk
        ? `<button class="samiso" title="build a clean full-length lane for this speaker: original where they speak alone, SepFormer separation at overlap windows, silence elsewhere">clean</button>`
        : "");
    if (isSpk) {
      row.dataset.speaker = s.name;
      label.querySelector(".lname").onclick = async (ev) => {
        ev.stopPropagation();
        const cur = (state.job.speaker_names || {})[s.name] || "";
        const name = prompt(
          `Name this voice (the dataset needs a real identity — ` +
          `"${s.name}" is just pyannote's arbitrary label):`, cur);
        if (name === null) return;
        await fetch(`/api/jobs/${state.vid}/speaker_name`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ speaker: s.name, name }),
        });
        state.tracks.clear();
        $("tracks").innerHTML = "";
        poll();
      };
    }
    label.querySelector(".mute").onclick = (ev) => {
      ev.stopPropagation();
      toggleLane(s.name);
    };
    if (isSpk)
      label.querySelector(".samiso").onclick = (ev) => {
        ev.stopPropagation();
        isolateSpeaker(s.name);
      };
    const canvas = document.createElement("canvas");
    canvas.addEventListener("click", (ev) => {
      const tr = state.tracks.get(s.name);
      if (!tr?.peaks) return;
      // click a lane = solo it and play from that point
      const t = (ev.offsetX / canvas.clientWidth) * tr.peaks.duration;
      state.playStopAt = null;
      soloLane(s.name, t);
    });
    row.append(label, canvas);
    holder.appendChild(row);
    const audio = new Audio(s.path);
    audio.preload = "metadata";
    const tr = { path: s.path, peaks: null, canvas, row, audio };
    state.tracks.set(s.name, tr);
    fetch(`/api/jobs/${state.vid}/peaks/${s.name}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => { tr.peaks = p; drawAll(); })
      .catch(() => {});
  });
  renderOverlaps();
  // transport clock: muted original drives scrubber/duration/events
  const first = sources[0];
  if (!player.src) { player.src = first.path; player.load(); }
  if (!state.mix.size || ![...state.mix].some((n) => state.tracks.has(n)))
    setMix(new Set([first.name]));
  else refreshLaneStyles();
  markCleanLanes();
}

/** which speakers already have a clean lane (derived from the lane names) */
function cleanSpeakers() {
  const out = new Set();
  (state.job.sources || []).forEach((s) => {
    const m = s.name.match(/^sam_(speaker_\d+)_clean/);
    if (m) out.add(m[1].toUpperCase());
  });
  return out;
}

function markCleanLanes() {
  const have = cleanSpeakers();
  state.tracks.forEach((tr, name) => {
    const btn = tr.row.querySelector(".samiso");
    if (!btn) return;
    const built = have.has(name);
    btn.textContent = built ? "clean ✓" : "clean";
    btn.classList.toggle("built", built);
    btn.title = built
      ? "clean lane already built — click to rebuild (segment will cut from it)"
      : "build a clean lane: original where this speaker is alone, SepFormer "
        + "at overlap windows — segment will then keep the rescued overlap";
  });
}

function renderOverlaps() {
  const el = $("overlapStrip");
  const ov = state.job.overlaps || [];
  if (!ov.length) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  if (el.dataset.n == ov.length) return;   // unchanged
  el.dataset.n = ov.length;
  el.innerHTML =
    `<span class="hint">⚠ ${ov.length} overlap region${ov.length > 1 ? "s" : ""} ` +
    `(both speakers at once — dropped unless rescued; click to jump, then ` +
    `<b>clean</b> on a speaker lane to rescue them):</span> ` +
    ov.slice(0, 60).map(([a, b]) =>
      `<button class="ovchip" data-a="${a}" data-b="${b}">${fmtT(a)}·${(b - a).toFixed(1)}s</button>`
    ).join("") + (ov.length > 60 ? ` <span class="hint">+${ov.length - 60} more</span>` : "");
  el.querySelectorAll(".ovchip").forEach((btn) => {
    btn.onclick = () => {
      const a = parseFloat(btn.dataset.a), b = parseFloat(btn.dataset.b);
      // center the overlap in a 10s isolate window + seek just before it
      const dur = parseFloat($("sepDur").value || "10");
      $("sepStart").value = Math.max(0, a - Math.max(0, (dur - (b - a)) / 2)).toFixed(1);
      state.playStopAt = b + 1;
      player.currentTime = Math.max(0, a - 1);
      player.play().catch(() => {});
    };
  });
}

/* ---- lane mixing: the muted transport <audio> is the clock; every unmuted
   lane's own Audio element shadows it (play/pause/seek + drift correction) ---- */

function setMix(names) {
  state.mix = names;
  const wasPlaying = !player.paused;
  const t = player.currentTime || 0;
  state.tracks.forEach((tr, name) => {
    const on = names.has(name);
    if (!on) tr.audio.pause();
    else {
      tr.audio.currentTime = t;
      if (wasPlaying) tr.audio.play().catch(() => {});
    }
  });
  refreshLaneStyles();
}

function toggleLane(name) {
  const next = new Set(state.mix);
  if (next.has(name)) {
    if (next.size === 1) return;           // keep at least one lane audible
    next.delete(name);
  } else next.add(name);
  setMix(next);
}

function soloLane(name, seekTo) {
  setMix(new Set([name]));
  if (seekTo != null) player.currentTime = seekTo;
  player.play().catch(() => {});
}

function refreshLaneStyles() {
  const labels = [];
  state.tracks.forEach((tr, name) => {
    const on = state.mix.has(name);
    tr.row.classList.toggle("audible", on);
    const btn = tr.row.querySelector(".mute");
    if (btn) btn.textContent = on ? "🔊" : "🔇";
    if (on) labels.push(laneInfo(name).label);
  });
  $("audibleLabel").textContent = labels.join(" + ") || "–";
}

const player = $("player");
player.muted = true;   // silent clock — audible sound comes from lane audios

function eachLive(fn) {
  state.mix.forEach((n) => { const tr = state.tracks.get(n); if (tr) fn(tr.audio); });
}
player.addEventListener("play", () =>
  eachLive((a) => { a.currentTime = player.currentTime; a.play().catch(() => {}); }));
player.addEventListener("pause", () => eachLive((a) => a.pause()));
player.addEventListener("seeked", () => eachLive((a) => { a.currentTime = player.currentTime; }));
player.addEventListener("ratechange", () => eachLive((a) => { a.playbackRate = player.playbackRate; }));
player.addEventListener("timeupdate", () => {
  drawAll();
  highlightPlaying();
  eachLive((a) => {                        // drift correction
    if (Math.abs(a.currentTime - player.currentTime) > 0.1)
      a.currentTime = player.currentTime;
  });
  if (state.playStopAt != null && player.currentTime >= state.playStopAt) {
    player.pause();
    state.playStopAt = null;
  }
});

/** karaoke: mark the segment row and word under the playhead */
let _lastRow = null;
function highlightPlaying() {
  const t = player.currentTime;
  const rows = document.querySelectorAll("#transcript .seg");
  let active = null;
  for (const r of rows) {
    if (t >= parseFloat(r.dataset.start) && t <= parseFloat(r.dataset.end)) {
      active = r; break;
    }
  }
  if (_lastRow && _lastRow !== active) {
    _lastRow.classList.remove("playing");
    _lastRow.querySelectorAll(".w.now").forEach((w) => w.classList.remove("now"));
  }
  if (!active) { _lastRow = null; return; }
  active.classList.add("playing");
  active.querySelectorAll(".w").forEach((w) => {
    const on = t >= parseFloat(w.dataset.a) && t <= parseFloat(w.dataset.b);
    w.classList.toggle("now", on);
  });
  const cv = active.querySelector(".segwave");
  if (cv?._peaks) drawSegWave(cv);
  _lastRow = active;
}

function drawAll() {
  const segById = Object.fromEntries(state.segments.map((s) => [s.seg_id, s]));
  state.tracks.forEach((tr, name) => drawTrack(tr, name, segById));
}

function drawTrack(tr, name, segById) {
  const canvas = tr.canvas;
  const W = (canvas.width = canvas.clientWidth * devicePixelRatio);
  const H = (canvas.height = canvas.clientHeight * devicePixelRatio);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (!tr.peaks || !tr.peaks.bins.length) {
    ctx.fillStyle = "#3a4a5c";
    ctx.font = `${11 * devicePixelRatio}px monospace`;
    ctx.fillText("loading waveform…", 8, H / 2);
    return;
  }
  const bins = tr.peaks.bins;
  const mid = H / 2, bw = W / bins.length;
  ctx.fillStyle = name === "original" ? "#3a4a5c" : laneInfo(name).color + "99";
  bins.forEach(([mn, mx], i) => {
    const y0 = mid + mn * mid * 0.92, y1 = mid + mx * mid * 0.92;
    ctx.fillRect(i * bw, Math.min(y0, y1), Math.max(1, bw * 0.8), Math.abs(y1 - y0) || 1);
  });
  const dur = tr.peaks.duration;
  // overlap shading (amber strip along the bottom)
  ctx.fillStyle = "#f0b13c55";
  (state.job.overlaps || []).forEach(([a, b]) => {
    const x0 = (a / dur) * W, x1 = (b / dur) * W;
    ctx.fillRect(x0, H - 4 * devicePixelRatio, Math.max(1.5, x1 - x0), 4 * devicePixelRatio);
  });
  state.candidates.forEach((c) => {
    const seg = segById[c.seg_id];
    if (!seg || c.position_s == null) return;
    if (name !== seg.speaker && name !== "original") return;
    const x = ((seg.start + c.position_s) / dur) * W;
    ctx.fillStyle = name === seg.speaker ? "#ff7ab0" : "#ff7ab066";
    const r = 4 * devicePixelRatio;
    ctx.beginPath();
    ctx.moveTo(x, 2); ctx.lineTo(x + r, 2 + r);
    ctx.lineTo(x, 2 + 2 * r); ctx.lineTo(x - r, 2 + r);
    ctx.fill();
  });
  const x = (player.currentTime / dur) * W;
  ctx.fillStyle = "#e8eef4";
  ctx.fillRect(x, 0, Math.max(1.5, devicePixelRatio), H);
}

/* ---------------- manifests + transcript ---------------- */

async function loadManifests() {
  const st = state.job.stages;
  const mt = state.job.manifest_mtimes || {};
  const get = async (name) => {
    const r = await fetch(`/api/jobs/${state.vid}/manifests/${name}`);
    return r.ok ? r.json() : null;
  };
  // refetch whenever a manifest's mtime changes (i.e. its stage was re-run)
  const stale = (file, key) =>
    state.loaded[key] !== (mt[file] ?? null);
  const mark = (file, key) => (state.loaded[key] = mt[file] ?? null);

  let dirty = false;
  if (st.segment?.status === "done" && stale("segments.jsonl", "segments")) {
    state.segments = (await get("segments.jsonl")) || [];
    mark("segments.jsonl", "segments"); dirty = true;
  }
  if (st.asr?.status === "done" && stale("transcripts.jsonl", "transcripts")) {
    const rows = (await get("transcripts.jsonl")) || [];
    state.transcripts = Object.fromEntries(rows.map((r) => [r.seg_id, r]));
    mark("transcripts.jsonl", "transcripts"); dirty = true;
  }
  if (st.align?.status === "done" && stale("alignments.jsonl", "alignments")) {
    const rows = (await get("alignments.jsonl")) || [];
    state.alignments = Object.fromEntries(rows.map((r) => [r.seg_id, r]));
    mark("alignments.jsonl", "alignments"); dirty = true;
  }
  if (st.segment?.status === "done") $("bakeoffSec").classList.remove("hidden");
  if ((state.job.detectors || {}).bakeoff?.status === "done"
      && stale("asr_bakeoff.jsonl", "bakeoff")) {
    const rows = await get("asr_bakeoff.jsonl");
    if (rows) { renderBakeoff(rows); mark("asr_bakeoff.jsonl", "bakeoff"); }
  }
  const dets = state.job.detectors || {};
  for (const [name, d] of Object.entries(dets)) {
    if (name === "bakeoff" || name === "separate") continue;
    const file = `candidates_${name}.jsonl`;
    if (d.status === "done" && stale(file, "cand_" + name)) {
      state.candidates = state.candidates
        .filter((c) => c.detector_name !== name)
        .concat(((await get(file)) || [])
          .map((c) => ({ ...c, detector_name: name })));
      mark(file, "cand_" + name); dirty = true;
      renderHits();
    }
  }
  if (dirty) { renderTranscript(); drawAll(); }
}

function renderEmptyStates() {
  const st = state.job.stages;
  const waitTxt = (need) => {
    const s = st[need]?.status || "pending";
    if (s === "done") return "";
    if (s === "error") return `blocked: <b>${need}</b> failed — click its step above to retry`;
    if (s === "running") return `waiting for <b>${need}</b> (running now — watch the pipeline above)…`;
    return `appears after the <b>${need}</b> stage (${s})`;
  };
  $("transcriptEmpty").innerHTML = state.segments.length
    ? (Object.keys(state.transcripts).length ? "" : waitTxt("asr"))
    : waitTxt("segment");
  const nImpure = state.segments.filter((s) => s.purity != null && !s.purity_pass).length;
  const nClip = Object.values(state.alignments).filter((a) => a.clipped?.length).length;
  const nResc = state.segments.filter((s) => s.separated).length;
  $("transcriptBadge").textContent = state.segments.length
    ? `${state.segments.length} segments · ${new Set(state.segments.map((s) => s.speaker)).size} speakers`
      + (nResc ? ` · ${nResc} rescued` : "")
      + (nImpure ? ` · ${nImpure} impure` : "")
      + (nClip ? ` · ${nClip} clipped` : "")
    : "";
  const ranDets = Object.entries(state.job.detectors || {})
    .filter(([n, d]) => d.status === "done" && n !== "separate" && n !== "bakeoff")
    .map(([n]) => n);
  $("hitsEmpty").innerHTML = state.candidates.length ? "" :
    (!state.segments.length ? waitTxt("segment")
      : ranDets.length
        ? `<b>${ranDets.join(", ")}</b> ran and found <b>nothing</b> in this audio — ` +
          `that's a real result, not an error (this material may simply not contain ` +
          `those events). Try another detector, or audio you know contains them.`
        : "no candidates yet — pick a detector and press <b>Run detector</b>");
  $("tagsBadge").textContent = state.candidates.length
    ? `${state.candidates.length} candidates` : "";
}

function fmtT(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
}

function speakerLabel(spk) {
  const nm = (state.job?.speaker_names || {})[spk];
  return nm ? `${nm} (${spk.replace("SPEAKER_", "S")})` : spk.replace("SPEAKER_", "S");
}

/** lazy mini-waveform for a segment row (drawn once it scrolls into view) */
const segPeakObserver = new IntersectionObserver((entries) => {
  entries.forEach((e) => {
    if (!e.isIntersecting) return;
    segPeakObserver.unobserve(e.target);
    const cv = e.target;
    fetch(`/api/jobs/${state.vid}/segpeaks/${cv.dataset.seg}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => { if (p) { cv._peaks = p; drawSegWave(cv); } })
      .catch(() => {});
  });
}, { rootMargin: "300px" });

function drawSegWave(cv) {
  const p = cv._peaks;
  const W = (cv.width = cv.clientWidth * devicePixelRatio);
  const H = (cv.height = cv.clientHeight * devicePixelRatio);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (!p?.bins?.length) return;
  const mid = H / 2, bw = W / p.bins.length;
  ctx.fillStyle = cv.dataset.color + "aa";
  p.bins.forEach(([mn, mx], i) => {
    const y0 = mid + mn * mid * 0.9, y1 = mid + mx * mid * 0.9;
    ctx.fillRect(i * bw, Math.min(y0, y1), Math.max(1, bw * 0.85), Math.abs(y1 - y0) || 1);
  });
  // playhead while this segment is the one playing
  const s = parseFloat(cv.dataset.start), d = parseFloat(cv.dataset.dur);
  const t = player.currentTime;
  if (t >= s && t <= s + d) {
    ctx.fillStyle = "#e8eef4";
    ctx.fillRect(((t - s) / d) * W, 0, Math.max(1.5, devicePixelRatio), H);
  }
}

function renderTranscript() {
  const el = $("transcript");
  el.innerHTML = "";
  const candBySeg = {};
  state.candidates.forEach((c) =>
    (candBySeg[c.seg_id] = candBySeg[c.seg_id] || []).push(c));

  [...state.segments].sort((a, b) => a.start - b.start).forEach((seg) => {
    const tr = state.transcripts[seg.seg_id];
    const al = state.alignments[seg.seg_id];
    const row = document.createElement("div");
    row.className = "seg";
    row.dataset.seg = seg.seg_id;
    row.dataset.start = seg.start;
    row.dataset.end = seg.end;

    // ---- left column: identity, timing, mini waveform, play button ----
    const meta = document.createElement("div");
    meta.className = "meta";
    const col = spkColor(seg.speaker);
    meta.innerHTML =
      `<div class="segid"><span class="dot" style="background:${col}"></span>` +
      `<b>${speakerLabel(seg.speaker)}</b></div>` +
      `<div class="segtime">${fmtT(seg.start)} → ${fmtT(seg.end)} · ${seg.dur.toFixed(1)}s</div>` +
      `<canvas class="segwave" data-seg="${seg.seg_id}" data-start="${seg.start}" ` +
      `data-dur="${seg.dur}" data-color="${col}"></canvas>` +
      `<div class="segbtns"><button class="segplay">▶ play clip</button>` +
      (seg.purity != null ? `<span class="hint" title="voice match to this speaker's own voiceprint">purity ${seg.purity.toFixed(2)}</span>` : "") +
      `</div>`;
    const cv = meta.querySelector(".segwave");
    segPeakObserver.observe(cv);
    cv.onclick = (ev) => {
      ev.stopPropagation();
      const f = ev.offsetX / cv.clientWidth;
      state.playStopAt = seg.end;
      player.currentTime = seg.start + f * seg.dur;
      player.play().catch(() => {});
    };
    meta.querySelector(".segplay").onclick = (ev) => {
      ev.stopPropagation();
      state.playStopAt = seg.end;
      player.currentTime = seg.start;
      player.play().catch(() => {});
    };
    row.appendChild(meta);

    const text = document.createElement("div");
    text.className = "text";
    const cands = candBySeg[seg.seg_id] || [];

    if (al && al.words && al.words.length && tr?.text) {
      const showTimes = $("showWordTimes").checked;
      // Render EVERY token of the transcript. MMS can only align letters, so
      // digits/symbols have no timing — they must still be shown (they are
      // real speech), just not clickable.
      const tokens = tr.text.split(/\s+/).filter(Boolean);
      const timed = {};
      if (al.words[0].i !== undefined) {
        al.words.forEach((w) => (timed[w.i] = w));
      } else {                       // alignments written before `i` existed
        let k = 0;
        tokens.forEach((tok, i) => {
          if (k < al.words.length && al.words[k].w === tok) timed[i] = al.words[k++];
        });
      }
      tokens.forEach((tok, i) => {
        const w = timed[i];
        const span = document.createElement("span");
        const esc = tok.replace(/&/g, "&amp;").replace(/</g, "&lt;");
        if (!w) {                    // unalignable (digits, ₹, symbols)
          span.className = "w untimed";
          span.innerHTML = esc + " ";
          span.title = "no forced-alignment timing (digits and symbols can't be "
            + "aligned) — the audio is there, only the timestamp is unknown";
          text.appendChild(span);
          return;
        }
        span.className = "w";
        const abs = seg.start + w.start;
        span.dataset.a = abs;
        span.dataset.b = seg.start + w.end;
        span.innerHTML = esc +
          (showTimes ? `<sub class="wt">${abs.toFixed(2)}</sub>` : "") + " ";
        span.title = `${abs.toFixed(2)}s → ${(seg.start + w.end).toFixed(2)}s ` +
          `(${(w.end - w.start).toFixed(2)}s) · segment-relative ` +
          `${w.start.toFixed(2)}–${w.end.toFixed(2)}`;
        span.onclick = (ev) => { ev.stopPropagation(); seekPlay(abs); };
        text.appendChild(span);
        cands.forEach((c) => {
          if (c.position_s != null && c.position_s >= w.start && c.position_s <= w.end)
            text.appendChild(makeChip(c, seg));
        });
      });
    } else if (tr?.text) {
      text.textContent = tr.text;
    } else if (tr?.error) {
      text.innerHTML = `<span class="hint" style="color:var(--err)">⚠ ${tr.error}</span>`;
    } else {
      const st = state.job?.stages?.asr?.status || "pending";
      text.innerHTML = `<span class="hint">no transcript yet — ` +
        (st === "done" ? "this clip returned empty"
         : st === "running" ? "asr is running…"
         : `click the <b>asr</b> step to transcribe (${st})`) + `</span>`;
    }
    cands.forEach((c) => {
      const placed = al?.words?.some((w) =>
        c.position_s != null && c.position_s >= w.start && c.position_s <= w.end);
      if (!placed) text.appendChild(makeChip(c, seg));
    });
    // script-mix indicator: catches an ASR drifting to all-Devanagari
    if (tr?.script && tr.script.chars > 8) {
      const chip = document.createElement("span");
      const lat = tr.script.latin_frac;
      chip.className = "chip status";
      chip.textContent = `deva ${Math.round(tr.script.deva_frac * 100)}% / lat ${Math.round(lat * 100)}%`;
      chip.title = "script mix — Hinglish should keep English words in Latin; "
        + "0% Latin on English-heavy speech means the ASR transliterated them";
      text.appendChild(chip);
    }
    if (al?.status && al.status !== "clean") {
      const chip = document.createElement("span");
      chip.className = `chip status ${al.status}`;
      chip.textContent = al.status;
      chip.title = `unaligned speech islands: ${JSON.stringify(al.islands || [])} — listen here for untranscribed sounds`;
      text.appendChild(chip);
    }
    if (seg.over_len) {
      const chip = document.createElement("span");
      chip.className = "chip over";
      chip.textContent = ">20s";
      text.appendChild(chip);
    }
    if (seg.separated) {
      const chip = document.createElement("span");
      chip.className = "chip status";
      chip.style.borderColor = "#ffd166";
      chip.style.color = "#ffd166";
      const pct = Math.round((seg.separated_frac || 0) * 100);
      chip.textContent = `rescued ${pct}%`;
      chip.title = `${(seg.separated_s || 0).toFixed(2)}s of this ${seg.dur.toFixed(1)}s ` +
        `clip is separated overlap audio (${seg.source}); the other ` +
        `${(100 - pct)}% is untouched original`;
      text.appendChild(chip);
    }
    if (seg.purity != null && !seg.purity_pass) {
      const chip = document.createElement("span");
      chip.className = "chip over";
      chip.textContent = `impure ${seg.purity.toFixed(2)}`;
      chip.title = "this clip's voice doesn't match the speaker's own " +
        "voiceprint — likely crosstalk or a diarization error. Listen before trusting it.";
      text.appendChild(chip);
    }
    (al?.clipped || []).forEach((side) => {
      const chip = document.createElement("span");
      chip.className = "chip over";
      chip.textContent = `✂ ${side}`;
      chip.title = `the ${side === "start" ? "first" : "last"} word runs into the ` +
        `clip edge — probably cut in half by segmentation`;
      text.appendChild(chip);
    });
    row.appendChild(text);
    row.onclick = () => seekPlay(seg.start);
    el.appendChild(row);
  });
}

function makeChip(c, seg) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = `[${c.tag}] ${c.score.toFixed(2)}`;
  chip.title = "click to play ±2s around the hit\n" + JSON.stringify(c.evidence);
  chip.onclick = (ev) => { ev.stopPropagation(); playWindow(seg.start + (c.position_s ?? 0)); };
  return chip;
}

function seekPlay(t) {
  state.playStopAt = null;
  player.currentTime = Math.max(0, t);
  player.play().catch(() => {});
}

function playWindow(t) {
  player.currentTime = Math.max(0, t - 2);
  state.playStopAt = t + 2;
  player.play().catch(() => {});
}

/* ---------------- ASR bake-off (Phase 1) ---------------- */

const FILLER_RE = /(उह+|हम्+म?|अं+|अच्छा|मतलब|यानी|हाँ|\b(?:uh+|um+|hmm+|haan|like)\b)/gi;
const BAKE_ENGINES = ["srota", "sarvam", "gemini"];

$("bakeRun").addEventListener("click", async () => {
  const btn = $("bakeRun");
  btn.disabled = true; btn.textContent = "Running…";
  state.loaded.bakeoff = -1;
  const r = await fetch(`/api/jobs/${state.vid}/bakeoff`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit: parseInt($("bakeLimit").value, 10) || 12 }),
  });
  if (!r.ok) {
    banner("bake-off: " + ((await r.json()).detail || r.status));
    btn.disabled = false; btn.textContent = "Run bake-off";
  }
  poll();
});

function renderBakeStatus() {
  $("bakeOf").textContent = state.segments.length
    ? `of ${state.segments.length} segments (evenly spaced across the call)` : "";
  const d = (state.job.detectors || {}).bakeoff;
  const el = $("bakeStatus");
  const btn = $("bakeRun");
  if (!d) { el.textContent = ""; return; }
  const busy = d.status === "running" || d.status === "queued";
  const elapsed = d.status === "running" && d.started
    ? ` · ${Math.max(0, Math.round(Date.now() / 1000 - d.started))}s` : "";
  el.className = `detStatus ${d.status}`;
  el.innerHTML = busy
    ? `<span class="icon"></span> ${d.status}${d.msg ? " — " + d.msg : ""}${elapsed}`
    : d.status === "error" ? `✗ ${d.msg || "failed"}` : `✓ ${d.msg || "done"}`;
  btn.disabled = !!busy;
  btn.textContent = busy ? "Running…" : "Run bake-off";
}

function markFillers(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return esc.replace(FILLER_RE, "<mark>$1</mark>");
}

function renderBakeoff(rows) {
  const el = $("bakeResults");
  if (!rows || !rows.length) { el.innerHTML = ""; return; }
  const fillerCount = (t) => (t.match(FILLER_RE) || []).length;
  const totals = {};
  BAKE_ENGINES.forEach((e) => {
    totals[e] = {
      miss: rows.filter((r) => !(r.engines[e]?.text)).length,
      fillers: rows.reduce((n, r) => n + fillerCount(r.engines[e]?.text || ""), 0),
    };
  });
  let html = `<table class="baketab"><tr><th>seg</th>` +
    BAKE_ENGINES.map((e) =>
      `<th>${e}<span class="hint"> · ∅${totals[e].miss} · fillers ${totals[e].fillers}</span></th>`
    ).join("") + "</tr>";
  rows.forEach((r) => {
    html += `<tr><td class="bseg" data-s="${r.start}" data-e="${r.start + r.dur}">` +
      `▶ ${fmtT(r.start)}<br><span class="hint">${(r.speaker || "").replace("SPEAKER_", "S")} · ${r.dur.toFixed(1)}s</span></td>`;
    BAKE_ENGINES.forEach((e) => {
      const c = r.engines[e] || {};
      html += c.text
        ? `<td>${markFillers(c.text)}</td>`
        : `<td class="bmiss">∅ ${c.error ? c.error.slice(0, 60) : "missed"}</td>`;
    });
    html += "</tr>";
  });
  el.innerHTML = html + "</table>";
  el.querySelectorAll(".bseg").forEach((td) => {
    td.style.cursor = "pointer";
    td.onclick = () => {
      state.playStopAt = parseFloat(td.dataset.e);
      player.currentTime = parseFloat(td.dataset.s);
      player.play().catch(() => {});
    };
  });
}

/* ---------------- detectors ---------------- */

function renderDetectors() {
  const pick = $("detPicker");
  const names = state.job.detector_names || [];
  if (pick.childElementCount !== names.length) {
    pick.innerHTML = "";
    names.forEach((n) => {
      const o = document.createElement("option");
      o.value = n; o.textContent = `[${n}]`;
      pick.appendChild(o);
    });
  }
  const d = (state.job.detectors || {})[pick.value];
  const el = $("detStatus");
  const btn = $("detRun");
  if (!d) { el.textContent = ""; el.className = "detStatus"; btn.disabled = false; btn.textContent = "Run detector"; return; }
  el.className = `detStatus ${d.status}`;
  el.innerHTML = (d.status === "running" || d.status === "queued")
    ? `<span class="icon"></span> ${d.status}${d.msg ? " — " + d.msg : ""}`
    : `${d.status === "done" ? "✓ done" : d.status === "error" ? "✗ " + (d.msg || "failed") : d.status}${d.status === "done" && d.msg ? " — " + d.msg : ""}`;
  const busy = d.status === "running" || d.status === "queued";
  btn.disabled = busy;
  btn.textContent = busy ? "Running…" : "Run detector";
}

$("detRun").addEventListener("click", async () => {
  const det = $("detPicker").value;
  if (!det) return;
  const btn = $("detRun");
  btn.disabled = true; btn.textContent = "Running…";
  state.loaded["cand_" + det] = -1;
  const r = await fetch(`/api/jobs/${state.vid}/detect`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detector: det }),
  });
  if (!r.ok) {
    banner("detector: " + ((await r.json()).detail || r.status));
    btn.disabled = false; btn.textContent = "Run detector";
  }
  poll();
});

function renderHits() {
  const el = $("hits");
  if (!state.candidates.length) { el.innerHTML = ""; return; }
  const segById = Object.fromEntries(state.segments.map((s) => [s.seg_id, s]));
  const rows = [...state.candidates].sort((a, b) => b.score - a.score);
  el.innerHTML =
    "<table><tr><th>#</th><th>tag</th><th>score</th><th>speaker</th><th>at</th><th>listen</th></tr>" +
    rows.map((c, i) => {
      const seg = segById[c.seg_id] || {};
      const t = (seg.start ?? 0) + (c.position_s ?? 0);
      return `<tr><td>${i + 1}</td><td>[${c.tag}]</td>` +
        `<td class="score">${c.score.toFixed(3)}</td>` +
        `<td>${(seg.speaker || "?").replace("SPEAKER_", "S")}</td>` +
        `<td style="font-family:var(--mono)">${fmtT(t)}</td>` +
        `<td><button onclick="playWindow(${t})">▶ ±2s</button></td></tr>`;
    }).join("") + "</table>";
}
window.playWindow = playWindow;

boot();
