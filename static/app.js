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
  segment: "2–20s utterances", asr: "srota transcription", align: "word timings",
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

/* ---------------- SAM-Audio isolate panel ---------------- */

$("sepAtPlayhead").addEventListener("click", () => {
  $("sepStart").value = (player.currentTime || 0).toFixed(1);
});
$("sepRun").addEventListener("click", async () => {
  const prompt = $("sepPrompt").value.trim();
  if (!prompt) { $("sepPrompt").focus(); return; }
  const btn = $("sepRun");
  btn.disabled = true; btn.textContent = "Queued…";
  const r = await fetch(`/api/jobs/${state.vid}/separate`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      start: parseFloat($("sepStart").value || "0"),
      dur: parseFloat($("sepDur").value || "10"),
      model: $("sepModel").value,
      reranking: parseInt($("sepRerank").value, 10),
    }),
  });
  if (!r.ok) {
    banner("isolate: " + ((await r.json()).detail || r.status));
    btn.disabled = false; btn.textContent = "Isolate";
  }
  poll();
});

function renderSepStatus() {
  const d = (state.job.detectors || {}).separate;
  const el = $("sepStatus");
  const btn = $("sepRun");
  if (!d) { el.textContent = ""; return; }
  const busy = d.status === "running" || d.status === "queued";
  el.className = `detStatus ${d.status}`;
  el.innerHTML = busy
    ? `<span class="icon"></span> ${d.status}${d.msg ? " — " + d.msg : ""}`
    : d.status === "error" ? `✗ ${d.msg || "failed"}`
    : `✓ ${d.msg || "done"} — new lanes above`;
  btn.disabled = busy;
  btn.textContent = busy ? "Isolating…" : "Isolate";
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
    step.innerHTML =
      `<span class="icon">${ICONS[status] ?? "○"}</span>` +
      `<span><span class="name">${name}</span>` +
      `<span class="sub">${status === "running" ? (st.msg || pct + "%")
        : status === "queued" ? "queued"
        : status === "error" ? "failed — click to retry"
        : STAGE_EXPLAIN[name] || ""}</span></span>` +
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
  let m = name.match(/^sam_(.+)_target$/);
  if (m) return { label: `Isolated: ${m[1].replace(/_/g, " ")}`, color: "#ff7ab0" };
  m = name.match(/^sam_(.+)_residual$/);
  if (m) return { label: `Without: ${m[1].replace(/_/g, " ")}`, color: "#7fd1d8" };
  if (name.startsWith("SPEAKER_"))
    return { label: name.replace("SPEAKER_", "S"), color: spkColor(name) };
  return { label: name, color: "#8595a5" };
}

function renderTracks() {
  ["listenSec", "sepSec", "transcriptSec", "tagsSec"].forEach((s) => $(s).classList.remove("hidden"));
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
    label.innerHTML =
      `<button class="mute" title="toggle this lane in the mix">🔇</button>` +
      `<span class="lname" style="color:${info.color}">${info.label}</span>`;
    label.querySelector(".mute").onclick = (ev) => {
      ev.stopPropagation();
      toggleLane(s.name);
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
  // transport clock: muted original drives scrubber/duration/events
  const first = sources[0];
  if (!player.src) { player.src = first.path; player.load(); }
  if (!state.mix.size || ![...state.mix].some((n) => state.tracks.has(n)))
    setMix(new Set([first.name]));
  else refreshLaneStyles();
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
  eachLive((a) => {                        // drift correction
    if (Math.abs(a.currentTime - player.currentTime) > 0.1)
      a.currentTime = player.currentTime;
  });
  if (state.playStopAt != null && player.currentTime >= state.playStopAt) {
    player.pause();
    state.playStopAt = null;
  }
});

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
  const get = async (name) => {
    const r = await fetch(`/api/jobs/${state.vid}/manifests/${name}`);
    return r.ok ? r.json() : null;
  };
  let dirty = false;
  if (st.segment?.status === "done" && !state.loaded.segments) {
    state.segments = (await get("segments.jsonl")) || [];
    state.loaded.segments = true; dirty = true;
  }
  if (st.asr?.status === "done" && !state.loaded.transcripts) {
    const rows = (await get("transcripts.jsonl")) || [];
    state.transcripts = Object.fromEntries(rows.map((r) => [r.seg_id, r]));
    state.loaded.transcripts = true; dirty = true;
  }
  if (st.align?.status === "done" && !state.loaded.alignments) {
    const rows = (await get("alignments.jsonl")) || [];
    state.alignments = Object.fromEntries(rows.map((r) => [r.seg_id, r]));
    state.loaded.alignments = true; dirty = true;
  }
  const dets = state.job.detectors || {};
  for (const [name, d] of Object.entries(dets)) {
    if (d.status === "done" && !state.loaded["cand_" + name]) {
      state.candidates = state.candidates
        .filter((c) => c.detector_name !== name)
        .concat(((await get(`candidates_${name}.jsonl`)) || [])
          .map((c) => ({ ...c, detector_name: name })));
      state.loaded["cand_" + name] = true; dirty = true;
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
  $("transcriptBadge").textContent = state.segments.length
    ? `${state.segments.length} segments · ${new Set(state.segments.map((s) => s.speaker)).size} speakers`
    : "";
  $("hitsEmpty").innerHTML = state.candidates.length ? "" :
    (state.segments.length
      ? "no candidates yet — pick a detector and press <b>Run detector</b>"
      : waitTxt("segment"));
  $("tagsBadge").textContent = state.candidates.length
    ? `${state.candidates.length} candidates` : "";
}

function fmtT(t) {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1);
  return `${m}:${s.padStart(4, "0")}`;
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

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML =
      `<span class="dot" style="background:${spkColor(seg.speaker)}"></span>` +
      `${seg.speaker.replace("SPEAKER_", "S")}<br>${fmtT(seg.start)} · ${seg.dur.toFixed(1)}s`;
    row.appendChild(meta);

    const text = document.createElement("div");
    text.className = "text";
    const cands = candBySeg[seg.seg_id] || [];

    if (al && al.words && al.words.length) {
      al.words.forEach((w) => {
        const span = document.createElement("span");
        span.className = "w";
        span.textContent = w.w + " ";
        span.onclick = (ev) => { ev.stopPropagation(); seekPlay(seg.start + w.start); };
        text.appendChild(span);
        cands.forEach((c) => {
          if (c.position_s != null && c.position_s >= w.start && c.position_s <= w.end)
            text.appendChild(makeChip(c, seg));
        });
      });
    } else {
      text.textContent = tr?.text || (tr?.error ? `⚠ ${tr.error}` : "…");
    }
    cands.forEach((c) => {
      const placed = al?.words?.some((w) =>
        c.position_s != null && c.position_s >= w.start && c.position_s <= w.end);
      if (!placed) text.appendChild(makeChip(c, seg));
    });
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
  state.loaded["cand_" + det] = false;
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
