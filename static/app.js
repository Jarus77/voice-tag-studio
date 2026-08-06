/* voice-tag-studio — single-page UI. Vanilla JS, no build. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  vid: null, job: null, pollTimer: null,
  audible: "original",
  tracks: new Map(),   // name -> {path, peaks, canvas, row}
  segments: [], transcripts: {}, alignments: {}, candidates: [],
  loaded: {},          // manifest name -> true once fetched
  playStopAt: null,    // for ±2s tag playback windows
  spkColor: {},
};
const COLORS = ["#4da3ff", "#3ecf8e", "#f0b13c", "#ef6461", "#b48ead", "#7fd1d8"];

/* ---------------- boot ---------------- */

async function boot() {
  const pf = await (await fetch("/api/preflight")).json();
  const bad = Object.entries(pf.checks).filter(([, c]) => !c.ok);
  const el = $("preflight");
  el.classList.remove("hidden");
  if (pf.ok) {
    el.classList.add("ok");
    el.textContent = "preflight OK" + (bad.length
      ? "  (optional missing: " + bad.map(([n, c]) => `${n} — ${c.msg}`).join(" · ") + ")"
      : "");
  } else {
    el.innerHTML = "<b>preflight problems:</b><br>" + bad
      .map(([n, c]) => `${n}: ${c.msg}`).join("<br>");
  }
  const jobs = await (await fetch("/api/jobs")).json();
  const pick = $("jobPicker");
  jobs.forEach((j) => {
    const o = document.createElement("option");
    o.value = j.video_id;
    o.textContent = `${j.video_id} ${j.title || ""}`.slice(0, 60);
    pick.appendChild(o);
  });
  if (jobs.length) selectJob(jobs[jobs.length - 1].video_id);
}

$("urlForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const url = $("urlInput").value.trim();
  if (!url) return;
  const r = await fetch("/api/jobs", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!r.ok) { alert((await r.json()).detail || r.status); return; }
  const d = await r.json();
  selectJob(d.video_id);
});

$("jobPicker").addEventListener("change", (ev) => {
  if (ev.target.value) selectJob(ev.target.value);
});

function selectJob(vid) {
  state.vid = vid;
  state.loaded = {};
  state.segments = []; state.transcripts = {}; state.alignments = {}; state.candidates = [];
  state.tracks.clear();
  $("tracks").innerHTML = "";
  poll();
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 1500);
}

/* ---------------- polling + stage pills ---------------- */

async function poll() {
  if (!state.vid) return;
  const r = await fetch(`/api/jobs/${state.vid}`);
  if (!r.ok) return;
  state.job = await r.json();
  renderPills();
  renderTracks();
  await loadManifests();
  renderDetectors();
}

function renderPills() {
  const el = $("stagePills");
  el.innerHTML = "";
  for (const [name, st] of Object.entries(state.job.stages)) {
    const pill = document.createElement("span");
    pill.className = `pill ${st.status || "pending"}`;
    let extra = "";
    if (st.status === "running")
      extra = ` ${Math.round((st.progress || 0) * 100)}%`;
    pill.innerHTML = `${name}${extra}<small>${st.status === "running" ? (st.msg || "") : st.status === "error" ? "✗" : st.status === "done" ? "✓" : ""}</small>`;
    pill.title = (st.msg || "") + "  (click to rerun)";
    pill.onclick = async () => {
      if (st.status === "running" || st.status === "queued") return;
      if (!confirm(`rerun stage "${name}"?`)) return;
      await fetch(`/api/jobs/${state.vid}/stages/${name}?force=true`, { method: "POST" });
      poll();
    };
    el.appendChild(pill);
  }
}

/* ---------------- listen: stacked multitrack timeline ---------------- */

function spkColor(spk) {
  if (!state.spkColor[spk])
    state.spkColor[spk] = COLORS[Object.keys(state.spkColor).length % COLORS.length];
  return state.spkColor[spk];
}

function renderTracks() {
  const secs = ["listenSec", "transcriptSec", "tagsSec"];
  secs.forEach((s) => $(s).classList.remove("hidden"));
  const holder = $("tracks");
  const sources = state.job.sources || [];
  if (!sources.length) {
    holder.innerHTML = "<p class='hint' style='padding:8px'>no audio yet — waiting for ingest…</p>";
    return;
  }
  if (holder.querySelector("p")) holder.innerHTML = "";
  sources.forEach((s) => {
    if (state.tracks.has(s.name)) return;
    const row = document.createElement("div");
    row.className = "track" + (s.name === state.audible ? " audible" : "");
    const label = document.createElement("div");
    label.className = "tlabel";
    const isSpk = s.name.startsWith("SPEAKER_");
    label.innerHTML =
      (isSpk ? `<span class="dot" style="background:${spkColor(s.name)}"></span>` : "") +
      `${s.name.replace("SPEAKER_", "S")} <span class="hint">🔊</span>`;
    label.title = "click to make this the audible version (position kept)";
    label.onclick = () => makeAudible(s.name);
    const canvas = document.createElement("canvas");
    canvas.addEventListener("click", (ev) => {
      const tr = state.tracks.get(s.name);
      if (!tr?.peaks) return;
      player.currentTime = (ev.offsetX / canvas.clientWidth) * tr.peaks.duration;
      state.playStopAt = null;
      player.play().catch(() => {});
    });
    row.append(label, canvas);
    holder.appendChild(row);
    const tr = { path: s.path, peaks: null, canvas, row };
    state.tracks.set(s.name, tr);
    fetch(`/api/jobs/${state.vid}/peaks/${s.name}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => { tr.peaks = p; drawAll(); })
      .catch(() => {});
  });
  if (!state.tracks.get(state.audible)) makeAudible(sources[0].name);
  else if (!player.src) makeAudible(state.audible);
}

function makeAudible(name) {
  const tr = state.tracks.get(name);
  if (!tr) return;
  const t = player.currentTime || 0;
  const wasPlaying = !player.paused;
  state.audible = name;
  player.src = tr.path;
  player.currentTime = t;
  if (wasPlaying) player.play().catch(() => {});
  state.tracks.forEach((v, k) => v.row.classList.toggle("audible", k === name));
}

const player = $("player");
player.addEventListener("timeupdate", () => {
  drawAll();
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
    ctx.fillText("loading…", 8, H / 2);
    return;
  }
  const bins = tr.peaks.bins;
  const mid = H / 2, bw = W / bins.length;
  const isSpk = name.startsWith("SPEAKER_");
  ctx.fillStyle = isSpk ? spkColor(name) + "99" : "#3a4a5c";
  bins.forEach(([mn, mx], i) => {
    const y0 = mid + mn * mid * 0.92, y1 = mid + mx * mid * 0.92;
    ctx.fillRect(i * bw, Math.min(y0, y1), Math.max(1, bw * 0.8), Math.abs(y1 - y0) || 1);
  });
  // tag markers: on the owning speaker's lane + faintly on original
  const dur = tr.peaks.duration;
  state.candidates.forEach((c) => {
    const seg = segById[c.seg_id];
    if (!seg || c.position_s == null) return;
    const onLane = name === seg.speaker || name === "original";
    if (!onLane) return;
    const x = ((seg.start + c.position_s) / dur) * W;
    ctx.fillStyle = name === seg.speaker ? "#ff7ab0" : "#ff7ab066";
    ctx.beginPath();  // ◆ diamond
    const r = 4 * devicePixelRatio;
    ctx.moveTo(x, 2);
    ctx.lineTo(x + r, 2 + r);
    ctx.lineTo(x, 2 + 2 * r);
    ctx.lineTo(x - r, 2 + r);
    ctx.fill();
  });
  // playhead
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
      chip.title = `islands: ${JSON.stringify(al.islands || [])}`;
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
  chip.title = JSON.stringify(c.evidence);
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
      o.value = n; o.textContent = n;
      pick.appendChild(o);
    });
  }
  const d = (state.job.detectors || {})[pick.value];
  $("detStatus").textContent = d
    ? `${d.status}${d.msg ? " — " + d.msg : ""}` : "";
}

$("detRun").addEventListener("click", async () => {
  const det = $("detPicker").value;
  if (!det) return;
  state.loaded["cand_" + det] = false;
  const r = await fetch(`/api/jobs/${state.vid}/detect`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ detector: det }),
  });
  if (!r.ok) alert((await r.json()).detail || r.status);
});

function renderHits() {
  const el = $("hits");
  if (!state.candidates.length) { el.innerHTML = "<p class='hint'>no candidates yet</p>"; return; }
  const segById = Object.fromEntries(state.segments.map((s) => [s.seg_id, s]));
  const rows = [...state.candidates].sort((a, b) => b.score - a.score);
  el.innerHTML =
    "<table><tr><th>#</th><th>tag</th><th>score</th><th>speaker</th><th>at</th><th></th></tr>" +
    rows.map((c, i) => {
      const seg = segById[c.seg_id] || {};
      const t = (seg.start ?? 0) + (c.position_s ?? 0);
      return `<tr><td>${i + 1}</td><td>[${c.tag}]</td>` +
        `<td class="score">${c.score.toFixed(3)}</td>` +
        `<td>${(seg.speaker || "?").replace("SPEAKER_", "S")}</td>` +
        `<td>${fmtT(t)}</td>` +
        `<td><button onclick="playWindow(${t})">▶ ±2s</button></td></tr>`;
    }).join("") + "</table>";
}
window.playWindow = playWindow;

boot();
