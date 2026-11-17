"use strict";
/* Ballrain training dashboard: polls /api/status every 60 s, fetches the map / curve payloads when their
   version changes, renders everything client-side with ECharts (maps on a "geo" component built from
   the level outlines, so pan / zoom / tooltips / legend toggles come for free). */

const POLL_S = 60;
const RUN_COLORS = { r1: "#9aa0a6", r2: "#6fbf73", r3: "#5aa9e6", r4: "#b58cf0", r5: "#ffb347", r6: "#ef6f6c", r7: "#4fd1c5", r8: "#f6e05e" };
const OUT = { finish: { c: "#7ed69a", n: "通关" }, gameover: { c: "#ff6b5e", n: "命用完" }, stopped: { c: "#8f9bff", n: "提前结束（卡住 / 超时）" } };
const INK = "#ebe3cf", MUTED = "#a9a089", BRASS = "#b8893a", AMBER = "#ffcf5a";
const coarse = matchMedia("(pointer: coarse)").matches;
// Ballance terms in the Chinese community's wording (one place, so they are easy to rename)
const TERMS = { checkpoint: "盘点", sector: "小节", fan: "风扇", airship: "飞船" };

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t == null ? "" : t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 0) => (v == null || isNaN(v) ? "—" : Number(v).toLocaleString("zh-CN", { maximumFractionDigits: d, minimumFractionDigits: d }));
const pct = (v) => (v == null || isNaN(v) ? "—" : (100 * v).toFixed(0) + "%");
const sec = (ms) => (ms == null ? "—" : (ms / 1000).toFixed(3) + " s");
const runColor = (r) => RUN_COLORS[r] || "#ddd";

let S = null;           // last /api/status
let sel = null;         // selected run
const P = {};           // payload cache: name -> {v, data}
const charts = {};
let nextAt = 0, timer = null;
try { sel = localStorage.getItem("dash-run"); } catch (e) { /* private mode */ }

/* ------------------------------------------------------------------ game bitmap font (Font_1) */
const FONT = { img: null, meta: null };
(async () => {
  try {
    const meta = await (await fetch("/assets/font.json")).json();
    const img = new Image();
    img.src = "/assets/font.png";
    await img.decode();
    FONT.img = img; FONT.meta = meta;
    document.querySelectorAll(".gt").forEach((el) => gameText(el, el.dataset.text || el.textContent, +el.dataset.h || 22, el.dataset.tone));
  } catch (e) { document.body.classList.add("no-font"); }
})();
{ const probe = new Image(); probe.onerror = () => document.body.classList.add("no-assets"); probe.src = "/assets/button.png"; }

function gameText(el, text, h, tone) {
  el.dataset.text = text; el.dataset.h = h; if (tone) el.dataset.tone = tone;
  if (!FONT.img) { el.textContent = text; return; }
  const g = FONT.meta.glyphs, cell = FONT.meta.cell, top = 2, bot = 30;
  const scale = h / (bot - top), dpr = window.devicePixelRatio || 1;
  const norm = String(text).replace(/[—–]/g, "-");
  // the game's menu font has Latin glyphs only: anything it cannot draw stays in the page font, never "?"
  if ([...norm].some((ch) => !g[ch.charCodeAt(0)])) { el.textContent = norm; el.classList.add("gt-plain"); return; }
  const chars = [...norm].map((ch) => g[ch.charCodeAt(0)]);
  const gap = 2;
  const w = chars.reduce((a, c) => a + (c ? c[2] + gap : 8), 0);
  const cv = document.createElement("canvas");
  cv.width = Math.ceil((w + 4) * scale * dpr); cv.height = Math.ceil((h + 4) * dpr);
  cv.style.width = (w + 4) * scale + "px"; cv.style.height = h + 4 + "px";
  const x = cv.getContext("2d");
  x.scale(dpr, dpr);
  const ink = document.createElement("canvas");
  ink.width = cv.width; ink.height = cv.height;
  const y = ink.getContext("2d");
  y.scale(dpr, dpr);
  let px = 2;
  for (const c of chars) {
    if (c) y.drawImage(FONT.img, c[0], c[1] + top, c[2], bot - top, px * scale, 2, c[2] * scale, h);
    px += (c ? c[2] : 6) + gap;
  }
  y.globalCompositeOperation = "source-in";
  const gr = y.createLinearGradient(0, 2, 0, h + 2);
  if (tone === "dark") { gr.addColorStop(0, "#5a3a12"); gr.addColorStop(1, "#2a1806"); }
  else { gr.addColorStop(0, "#fff0a8"); gr.addColorStop(0.5, "#ffcf5a"); gr.addColorStop(1, "#e88a12"); }
  y.fillStyle = gr; y.fillRect(0, 0, cv.width, cv.height);
  x.save(); x.setTransform(1, 0, 0, 1, 0, 0);
  if (tone !== "dark") { x.shadowColor = "rgba(20,8,0,.95)"; x.shadowBlur = 3 * dpr; x.shadowOffsetY = 1 * dpr; }
  x.drawImage(ink, 0, 0); x.restore();
  cv.setAttribute("role", "img"); cv.setAttribute("aria-label", text);
  el.replaceChildren(cv);
}

/* ------------------------------------------------------------------------------- data loading */
const inflight = {};
async function getPayload(name) {
  const v = (S.data || {})[name];
  if (!v) return P[name] ? P[name].data : null;
  if (P[name] && P[name].v === v) return P[name].data;
  const key = name + "@" + v;
  if (!inflight[key]) {
    inflight[key] = (async () => {
      try {
        const r = await fetch(`/api/data/${name}.json?v=${v}`);
        if (r.ok) P[name] = { v, data: await r.json() };
      } catch (e) { /* keep the last good payload */ }
      delete inflight[key];
      return P[name] ? P[name].data : null;
    })();
  }
  return inflight[key];
}

/* first draw / reset: full replace; data updates: keep pan / zoom / legend state */
function apply(ch, opt, full) {
  if (full || !ch._drawn) { ch.setOption(opt, true); ch._drawn = true; return; }
  const o = { ...opt };
  delete o.geo; delete o.dataZoom;
  ch.setOption(o, { replaceMerge: ["series"] });
}

async function load() {
  clearTimeout(timer);
  let wait = POLL_S;
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    const j = await r.json();
    if (r.ok && Array.isArray(j.runs)) { S = j; $("errs").textContent = ""; }
    else $("errs").textContent = "服务器暂时出错，显示的是上一次的数据。";
  } catch (e) {
    $("errs").textContent = "暂时连不上服务器，显示的是上一次的数据。";
  }
  try {
    if (S && S.gen && S.gen.running) wait = 10;
    nextAt = Date.now() + wait * 1000;
    if (S) await render();
  } catch (e) {
    console.error(e);
  } finally {
    clearTimeout(timer);
    timer = setTimeout(load, wait * 1000);
  }
}

/* ------------------------------------------------------------------------------------ header */
function renderHeader() {
  if (!sel || !S.runs.includes(sel)) sel = S.default_run || S.runs[S.runs.length - 1];
  const nav = $("runs");
  if (nav.dataset.runs !== S.runs.join(",") + "|" + sel) {
    nav.innerHTML = S.runs.map((r) => `<button type="button" class="mbtn${r === sel ? " on" : ""}" data-r="${esc(r)}"><span class="gt" data-h="17">${esc(r)}</span></button>`).join("");
    nav.dataset.runs = S.runs.join(",") + "|" + sel;
    nav.querySelectorAll("button").forEach((b) => {
      b.onclick = () => { sel = b.dataset.r; try { localStorage.setItem("dash-run", sel); } catch (e) { /* */ } render(); };
      const sp = b.querySelector(".gt"); gameText(sp, sp.textContent, 17);
    });
  }
  document.querySelectorAll(".rn").forEach((e) => (e.textContent = sel));
  $("milestone").textContent = S.milestone || "M2";
  const tb = $("tb");
  if (S.tensorboard_url) { tb.href = S.tensorboard_url; tb.classList.remove("off"); tb.title = ""; tb.querySelector("span").textContent = "TensorBoard"; }
  else { tb.removeAttribute("href"); tb.classList.add("off"); tb.querySelector("span").textContent = "TensorBoard 暂未开放"; }
  const g = S.gen || {};
  const errs = Object.entries(g.errors || {});
  $("errs").textContent = errs.length ? "部分内容计算出错：" + errs.map(([k, v]) => `${k}: ${v}`).join("；") : "";
  updateMeta();
  $("foot").textContent = `时间均为 ${S.tz}。训练机：${S.machines.train}；评估机：${S.machines.eval}。数据只包含训练本身（课程、成功率、轨迹、曲线、SR 估计与原版评估）。`;
}

function updateMeta() {
  if (!S) return;
  const g = S.gen || {};
  const left = Math.max(0, Math.round((nextAt - Date.now()) / 1000));
  $("meta").textContent = `现在 ${S.now.slice(5, 16)}（${S.tz}）· 图表数据每 ${Math.round(S.interval_s / 60)} 分钟重算，上次 ${g.generated_at ? g.generated_at.slice(0, 11) : "—"}` +
    (g.seconds != null ? `（${g.seconds} 秒）` : "") + (g.running ? " · 正在重算…" : "") + ` · 页面 ${left} 秒后自动更新`;
}
setInterval(updateMeta, 1000);

$("refresh").onclick = async () => {
  const b = $("refresh"), sp = b.querySelector("span");
  b.disabled = true;
  try { sp.textContent = (await (await fetch("/api/refresh", { method: "POST" })).json()).msg; } catch (e) { sp.textContent = "请求失败"; }
  setTimeout(() => { sp.textContent = "立即刷新"; b.disabled = false; }, 6000);
  load();
};

/* ------------------------------------------------------------------------------------ status */
function tile(k, v, s, gt) {
  return `<div class="tile"><div class="k">${esc(k)}</div><div class="v">${gt ? `<span class="gt" data-h="20" data-tone="dark">${esc(v)}</span>` : esc(v)}</div>${s ? `<div class="s">${esc(s)}</div>` : ""}</div>`;
}

function renderStatus() {
  const s = S.status[sel] || {};
  $("state").textContent = s.state || "";
  const src = s.source_share || s.backend_share;
  const det = (s.source_share && s.source_detail) || {};
  const share = src ? Object.entries(src).map(([k, v]) => `<div class="row"><span class="gt">${esc(k)}</span><span class="gt">${esc(v + "%")}</span></div>` +
    (det[k] ? `<div class="sub">${esc(det[k].join(" + "))}</div>` : "")).join("") : "—";
  $("tiles").innerHTML = [
    tile("run", sel, s.machine ? `训练机：${s.machine}` : "", true),
    tile(`最后记录（${S.tz}）`, s.last_log || "—", s.last_log_age_min != null ? `${num(s.last_log_age_min, 0)} 分钟前；开始于 ${s.started || "—"}` : "", true),
    tile("更新次数", num(s.update), "", true),
    tile("训练步数", s.steps != null ? (s.steps / 1e6).toFixed(2) + " M" : "—", "", true),
    tile("每秒步数", num(s.steps_per_s), `最近 20 次平均 ${num(s.steps_per_s_avg20)}`, true),
    tile("课程开局点（TAS 帧）", num(s.frontier), [s.frontier_sector != null ? `第 ${num(s.frontier_sector)} 小节` : "", s.covered_pct != null ? `智能体已能接手全程的 ${s.covered_pct}%` : ""].filter(Boolean).join("，"), true),
    tile("开局区间成功率", pct(s.band_success), `推进阈值 60%；最近 20 次平均 ${pct(s.band_success_avg20)}`, true),
    tile("回合通关率", pct(s.finish_tas), `最近 20 次平均 ${pct(s.finish_tas_avg20)}`, true),
    tile("策略熵 / KL", `${num(s.u_entropy, 3)} / ${num(s.u_approx_kl, 4)}`, "随机策略的熵为 2.20", true),
    `<div class="tile"><div class="k">训练数据来自哪台机器（按步数，近 100 次）</div><div class="v split">${share}</div></div>`,
    tile("最佳 SR（含 TAS 前缀）", sec(s.best_sr_any), s.best_sr_k0 != null ? `从头开始最佳 ${sec(s.best_sr_k0)}` : "从 TAS 中途接手时的整关计时", true),
    tile("记录了轨迹的回合", num((S.traced_counts || {})[sel]), "抽样记录，供轨迹地图使用", true),
  ].join("");
  $("tiles").querySelectorAll(".gt").forEach((el) => gameText(el, el.textContent, el.closest(".split") ? 18 : 20, "dark"));
  const c = s.curriculum || [];
  $("curr").innerHTML = c.length ? c.map((l) => `<li><time>${esc(l.slice(0, 11))}</time>${esc(l.slice(13).replace("curriculum: ", ""))}</li>`).join("") : "<li>（无）</li>";
}

/* ---------------------------------------------------------------------------- progress report */
function renderReport() {
  const R = S.reports || [];
  $("s-report").hidden = !R.length;
  if (!R.length) return;
  const one = (r) => `<div class="rtitle">${esc(r.title)}</div><ul class="ritems">${r.items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` +
    (r.next ? `<div class="rnext"><b>下一步</b>${esc(r.next)}</div>` : "");
  const [cur, ...old] = R;
  $("r-at").textContent = `${cur.at.slice(5)}（${cur.tz}）`;
  const open = $("report").querySelector("details")?.open;
  $("report").innerHTML = one(cur) + (old.length ? `<details class="rold"${open ? " open" : ""}><summary>更早的报告（${old.length}）</summary>` +
    old.map((r) => `<div class="rprev"><time>${esc(r.at.slice(5))}（${esc(r.tz)}）</time>${one(r)}</div>`).join("") + `</details>` : "");
}

/* --------------------------------------------------------------------------- highlight replay */
let hlOverlay = true;
try { hlOverlay = localStorage.getItem("dash-hl-ov") !== "0"; } catch (e) { /* private mode */ }
let hlShown = null;  // the recording on screen
function renderHighlight() {
  let h = S.highlight;
  $("s-hl").hidden = !h;
  if (!h) return;
  const vid = $("hl-video");
  // a new hourly recording never replaces the one someone is watching: it waits until playback is paused (the
  // server keeps the previous set for exactly this); caption and video stay those of the recording on screen
  const pending = hlShown && String(hlShown.v) !== String(h.v) && !vid.paused && !vid.ended;
  if (pending) {
    h = hlShown;
    if (!vid.dataset.waitPause) { vid.dataset.waitPause = "1"; vid.addEventListener("pause", () => { delete vid.dataset.waitPause; renderHighlight(); }, { once: true }); }
  }
  hlShown = h;
  // file names of this recording's set from the server (a server of the first version sends a flag, fixed names)
  const files = { video: typeof h.video === "string" ? h.video : "highlight.mp4",
    overlay: typeof h.overlay === "string" ? h.overlay : (h.overlay ? "highlight-overlay.mp4" : null) };
  const ov = hlOverlay && !!files.overlay;
  const src = `/media/${encodeURIComponent(ov ? files.overlay : files.video)}?v=${h.v}`;
  document.querySelector(".hlbox").style.setProperty("--hl-ar", ov ? "1366 / 768" : "1024 / 768");
  if (vid.dataset.src !== src) {
    const same = vid.dataset.v === String(h.v);  // same recording, other version: keep the position
    const t = same ? vid.currentTime : 0, playing = same && !vid.paused;
    vid.dataset.src = src; vid.dataset.v = String(h.v);
    vid.src = src;
    if (t > 0) vid.addEventListener("loadedmetadata", () => { vid.currentTime = Math.min(t, vid.duration || t); if (playing) vid.play().catch(() => {}); }, { once: true });
  }
  const tools = $("hl-tools");
  tools.style.display = files.overlay ? "" : "none";  // .maptools sets display: flex, which beats [hidden]
  tools.querySelectorAll("button").forEach((b) => {
    b.classList.toggle("on", (b.dataset.ov === "1") === !!ov);
    b.onclick = () => { hlOverlay = b.dataset.ov === "1"; try { localStorage.setItem("dash-hl-ov", hlOverlay ? "1" : "0"); } catch (e) { /* */ } renderHighlight(); };
  });
  $("hl-at").textContent = (h.recorded_at ? `录制于 ${h.recorded_at.slice(5)}（${h.tz}）` : "") + (pending ? " · 有新的回放，暂停后切换" : "");
  const parts = [`<b>${esc(h.run)}</b>`, `第 ${num(h.update)} 次更新`, h.k > 0 ? `从 TAS 第 ${num(h.k)} 帧接手` : "从头开始",
    `整关 <b>${(h.sr_ms / 1000).toFixed(3)} 秒</b>${h.tas_ms ? `（TAS ${(h.tas_ms / 1000).toFixed(3)} 秒）` : ""}`,
    `原始尝试：${esc(h.backend || "—")}${h.episode_at ? `，${esc(h.episode_at.slice(5))}` : ""}`,
    h.matched ? "原版游戏复现一致" : "原版游戏复现不一致"];
  $("hl-cap").innerHTML = parts.join(" · ") + `<br><span class="note">上一小时内成绩最好的一次尝试（无死亡通关，优先选接手最早的），在原版游戏（2002 物理）里按原输入重放、用游戏自己的画面录制：从出生到上${esc(TERMS.airship)}，TAS 部分加智能体部分，不剪辑。</span>`;
}

/* ---------------------------------------------------------------------------------- SR chart */
function renderSR() {
  const r = (S.sr || {})[sel];
  if (!r) { $("sr-head").innerHTML = `<p class="note">尚未计算（启动后约半分钟）。</p>`; return; }
  const b = r.benchmarks || {};
  const extra = r.alt ? r.shortcut_extra_s : 0;
  const est = (ratio) => (ratio == null ? null : +(b.tas / 1000 * ratio + extra * ratio).toFixed(2));
  let h = "";
  if (r.estimate_s) {
    h += `<div class="big"><span class="gt" data-h="30">${r.estimate_s[0].toFixed(1)} - ${r.estimate_s[1].toFixed(1)} s</span>
      <span class="note" style="margin:0">中位比率 ${r.ratio_range[0].toFixed(3)} – ${r.ratio_range[1].toFixed(3)}${r.alt ? `，已含鬼躲石变约 ${r.shortcut_extra_s} s × 比率` : ""}</span></div>`;
  } else {
    h += `<p class="note">样本不足，暂无整关估计。</p>`;
  }
  h += `<div class="bench"><span>TAS <b>${sec(b.tas)}</b></span><span>人类纪录 <b>${sec(b.human_wr)}</b></span><span>精品线 <b>${sec(b.featured)}</b></span><span>上榜线 <b>${sec(b.threshold)}</b></span></div>`;
  h += `<div class="note" style="margin:0">样本：${esc((r.runs_used || []).join(" + "))} 最近 ${r.window} 次更新（${r.update_range ? r.update_range.join(" – ") : "—"}）中，从 TAS 中途接手（k &gt; 0）且没有死亡的通关局，共 ${num(r.n)} 局${r.n_retail ? `（其中原版 2002 ${num(r.n_retail)} 局）` : ""}。</div>`;
  $("sr-head").innerHTML = h;
  $("sr-head").querySelectorAll(".gt").forEach((el) => gameText(el, el.textContent, 30));

  const bins = r.bins || [];
  const cats = bins.map((x) => `${x.lo}–${x.hi}`);
  const ys = bins.map((x) => est(x.median));
  const lo = Math.min(b.tas / 1000, ...ys.filter((v) => v != null), ...bins.map((x) => est(x.p10)).filter((v) => v != null)) - 4;
  const hi = Math.max(b.threshold / 1000, ...bins.filter((x) => x.in_range).map((x) => est(x.p90))) + 5;
  if (window.echarts) {
  const ch = chart("c-sr");
  const benchLine = (v, name, color, pos) => ({ yAxis: v / 1000, name, lineStyle: { color, type: "dashed", width: 1.4 }, label: { formatter: `${name} ${(v / 1000).toFixed(1)}`, color, position: pos, fontSize: 11, textBorderColor: "#000", textBorderWidth: 2 } });
  ch.setOption({
    animation: false, textStyle: { color: INK },
    grid: { left: 48, right: 16, top: 16, bottom: 58 },
    tooltip: {
      trigger: "axis", axisPointer: { type: "shadow" }, backgroundColor: "rgba(20,14,6,.95)", borderColor: BRASS, textStyle: { color: INK },
      formatter: (ps) => {
        const x = bins[ps[0].dataIndex];
        if (!x) return "";
        return `<b>开局帧 ${x.lo} – ${x.hi}</b><br>TAS 剩余段约 ${x.tas_seg_s} s · ${x.n} 局` +
          (x.median != null ? `<br>中位比率 ${x.median.toFixed(3)} → 整关约 ${est(x.median).toFixed(1)} s<br>p10 ${x.p10.toFixed(3)} → ${est(x.p10).toFixed(1)} s · p90 ${x.p90.toFixed(3)} → ${est(x.p90).toFixed(1)} s` : "") +
          (x.in_range ? "<br><span style='color:#ffcf5a'>计入估计范围</span>" : "<br><span style='color:#a9a089'>样本少或剩余段太短，不计入范围</span>");
      },
    },
    xAxis: { type: "category", data: cats, axisLabel: { color: MUTED, rotate: 35, fontSize: 10.5 }, axisLine: { lineStyle: { color: BRASS } }, name: "开局帧（TAS）", nameLocation: "middle", nameGap: 44, nameTextStyle: { color: MUTED } },
    yAxis: { type: "value", min: Math.floor(lo), max: Math.ceil(hi), name: "整关估计（秒）", nameTextStyle: { color: MUTED }, axisLabel: { color: MUTED }, splitLine: { lineStyle: { color: "rgba(255,220,150,.08)" } } },
    series: [
      {
        type: "bar", name: "中位比率估计", barMaxWidth: 34,
        data: bins.map((x, i) => ({ value: ys[i], itemStyle: { color: x.in_range ? "#d79b32" : "rgba(184,137,58,.35)", borderColor: "#2b1a0a" } })),
        markLine: { symbol: "none", silent: true, data: [benchLine(b.tas, "TAS", "#7ed69a", "insideStartBottom"), benchLine(b.human_wr, "人类纪录", "#5aa9e6", "insideEndTop"), benchLine(b.featured, "精品线", "#b58cf0", "insideEndTop"), benchLine(b.threshold, "上榜线", "#ff7a66", "insideEndTop")] },
        markArea: r.estimate_s ? { silent: true, itemStyle: { color: "rgba(255,207,90,.10)" }, data: [[{ yAxis: r.estimate_s[0] }, { yAxis: r.estimate_s[1] }]] } : undefined,
      },
      {
        type: "custom", name: "p10–p90", silent: true, clip: true,
        renderItem: (params, api) => {
          const x = bins[params.dataIndex];
          if (!x || x.p10 == null) return null;
          const a = api.coord([params.dataIndex, est(x.p10)]), c = api.coord([params.dataIndex, est(x.p90)]);
          const st = { stroke: "#f3e3b5", lineWidth: 1.3 };
          return { type: "group", children: [
            { type: "line", shape: { x1: a[0], y1: a[1], x2: c[0], y2: c[1] }, style: st },
            { type: "line", shape: { x1: a[0] - 5, y1: a[1], x2: a[0] + 5, y2: a[1] }, style: st },
            { type: "line", shape: { x1: c[0] - 5, y1: c[1], x2: c[0] + 5, y2: c[1] }, style: st }] };
        },
        data: bins.map((x, i) => i),
      },
    ],
  }, true);
  }

  let t = `<table><thead><tr><th>开局帧</th><th>TAS 剩余段</th><th>局数</th><th>中位比率</th><th>p10</th><th>p90</th><th>整关估计</th></tr></thead><tbody>`;
  for (const x of bins) {
    t += `<tr class="${x.in_range ? "" : "dim"}"><td>${x.lo} – ${x.hi}</td><td>${x.tas_seg_s} s</td><td>${x.n}</td><td>${x.median != null ? x.median.toFixed(3) : "—"}</td><td>${x.p10 != null ? x.p10.toFixed(3) : "—"}</td><td>${x.p90 != null ? x.p90.toFixed(3) : "—"}</td><td>${x.median != null ? est(x.median).toFixed(1) + " s" : "—"}</td></tr>`;
  }
  $("sr-table").innerHTML = t + "</tbody></table>";

  const k0 = (S.k0 || []).filter((x) => x.run === sel || (r.runs_used || []).includes(x.run));
  if (!k0.length) $("k0").innerHTML = `<p class="note" style="margin:0">这个 run 还没有从头开始的原版评估。</p>`;
  else {
    let k = `<table><thead><tr><th>run</th><th>时间</th><th>更新</th><th>游戏</th><th>采样</th><th>局数</th><th>通关率</th><th>无死亡通关</th><th>每局死亡</th><th>SR 中位</th><th>SR 最佳</th></tr></thead><tbody>`;
    for (const x of k0) k += `<tr><td>${esc(x.run)}</td><td>${esc(x.t || "")}</td><td>${num(x.update)}</td><td>${esc(x.backend)}</td><td>${x.greedy ? "贪心" : "温度 " + esc(x.temperature)}</td><td>${num(x.runs)}</td><td>${pct(x.success)}</td><td>${num(x.finished_no_death)}</td><td>${num(x.deaths_per_run, 1)}</td><td>${sec(x.sr_ms_median)}</td><td>${sec(x.sr_ms_min)}</td></tr>`;
    $("k0").innerHTML = k + "</tbody></table>";
  }
  $("sr-note").textContent = `怎么算的：比率 = 智能体从接手帧 k 跑到终点的用时 ÷ TAS 同一段的用时（TAS 在第 k 帧的计时 = (k − c0) × 1000 / 264，c0 = 20483 − 74446 × 0.264）。` +
    `整关估计 = ${(b.tas / 1000).toFixed(3)} s × 比率${r.alt ? ` + 鬼躲石变约 ${r.shortcut_extra_s} s × 比率` : ""}；范围取局数 ≥ ${r.min_n} 且 TAS 剩余段 ≥ ${r.min_seg_s} s 的分箱的中位比率最小值到最大值（柱状图中亮色的柱子，竖线为 p10–p90）。` +
    `这是外推：只统计无死亡的通关局，前段还没练到的部分按后段的比率估计，近终点分箱的段很短、噪声大，无头后端与原版游戏也略有差别；最终以从头开始的原版评估为准。`;
}

/* --------------------------------------------------------------------------------- the maps */
let geoReady = false;
async function ensureGeo() {
  if (geoReady) return true;
  const g = await getPayload("geometry");
  if (!g) return false;
  const ring = (r) => { const p = []; for (let i = 0; i < r.length; i += 2) p.push([r[i], r[i + 1]]); p.push(p[0]); return [p]; };
  const fc = { type: "FeatureCollection", features: ["walk", "steep", "rail"].map((k) => ({ type: "Feature", properties: { name: k }, geometry: { type: "MultiPolygon", coordinates: (g[k] || []).map(ring) } })) };
  echarts.registerMap("L1", fc);
  geoReady = true;
  return true;
}

function geoBase(roam) {
  return {
    map: "L1", roam: roam ? true : false, aspectScale: 1, silent: true, zoom: 1.05, scaleLimit: { min: 0.8, max: 60 },
    layoutCenter: ["50%", "50%"], layoutSize: "96%",
    itemStyle: { areaColor: "#cbbf9f", borderColor: "#7d6f52", borderWidth: 0.4 },
    regions: [{ name: "steep", itemStyle: { areaColor: "#5f6f88" } }, { name: "rail", itemStyle: { areaColor: "#c08a3e", borderColor: "#6b4a17" } }],
    emphasis: { disabled: true }, select: { disabled: true },
  };
}

const TT = { trigger: "item", backgroundColor: "rgba(20,14,6,.95)", borderColor: BRASS, textStyle: { color: INK, fontSize: 12.5 }, confine: true };
const LEG = { type: "scroll", top: 4, left: 6, right: 6, textStyle: { color: INK, fontSize: 12 }, pageTextStyle: { color: INK }, inactiveColor: "#555", pageIconColor: AMBER };

function chart(id) {
  if (!charts[id]) {
    charts[id] = echarts.init($(id), null, { renderer: "canvas" });
    new ResizeObserver(() => charts[id].resize()).observe($(id));
  }
  return charts[id];
}

const roamOn = {};
function roamToggle(id, onChange) {
  if (roamOn[id] == null) roamOn[id] = !coarse;
  const fs = $(id).closest(".panel").classList.contains("fs");
  return `<button type="button" class="tbtn${roamOn[id] ? " on" : ""}" data-roam="${id}">${roamOn[id] ? "拖动 / 缩放：开" : "拖动 / 缩放：关"}</button> <button type="button" class="tbtn" data-reset="${id}">复位视图</button> <button type="button" class="tbtn fsbtn" data-fs="${id}">${fs ? "退出全屏" : "全屏"}</button>`;
}

/* Fullscreen for a map panel: the Fullscreen API where it exists, else a fixed overlay (iPhone Safari). */
function setFsLabels() {
  document.querySelectorAll("[data-fs]").forEach((b) => { b.textContent = $(b.dataset.fs).closest(".panel").classList.contains("fs") ? "退出全屏" : "全屏"; });
}
function exitPseudoFs() {
  document.querySelectorAll(".panel.fs").forEach((p) => p.classList.remove("fs"));
  document.body.classList.remove("fs-lock");
  setFsLabels();
}
const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
function toggleFs(id) {
  const panel = $(id).closest(".panel");
  if (panel.classList.contains("fs")) {
    if (fsElement()) (document.exitFullscreen || document.webkitExitFullscreen).call(document);
    else exitPseudoFs();
    return;
  }
  panel.classList.add("fs");
  document.body.classList.add("fs-lock");
  setFsLabels();
  const req = panel.requestFullscreen || panel.webkitRequestFullscreen;
  if (req) { try { const r = req.call(panel); if (r && r.catch) r.catch(() => {}); } catch (e) { /* overlay stays */ } }
}
["fullscreenchange", "webkitfullscreenchange"].forEach((ev) => document.addEventListener(ev, () => { if (!fsElement()) exitPseudoFs(); }));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !fsElement() && document.querySelector(".panel.fs")) exitPseudoFs(); });
function wireRoam(box, id, redraw) {
  box.querySelector(`[data-roam="${id}"]`).onclick = () => { roamOn[id] = !roamOn[id]; chart(id).setOption({ geo: { roam: roamOn[id] } }); const b = box.querySelector(`[data-roam="${id}"]`); b.classList.toggle("on", roamOn[id]); b.textContent = roamOn[id] ? "拖动 / 缩放：开" : "拖动 / 缩放：关"; };
  box.querySelector(`[data-reset="${id}"]`).onclick = () => redraw(true);
  box.querySelector(`[data-fs="${id}"]`).onclick = () => toggleFs(id);
}

function tasSeries(F) {
  return { type: "lines", name: "TAS 轨迹", coordinateSystem: "geo", polyline: true, silent: true, z: 2, data: [{ coords: F.tas }], lineStyle: { color: "#e9edf7", width: 1, type: "dashed", opacity: 0.7 } };
}

async function renderFrontier(force, full) {
  const F = await getPayload("frontier");
  if (!F || !(await ensureGeo())) return;
  const id = "m-frontier", ch = chart(id), box = $("t-frontier");
  if (!box.dataset.wired) { box.innerHTML = roamToggle(id); wireRoam(box, id, () => renderFrontier(true, true)); box.dataset.wired = 1; }
  if (!force && ch._v === P.frontier.v) return;
  ch._v = P.frontier.v;
  const runs = [...new Set(F.history.map((h) => h.run))];
  const pt = (arr, extra) => arr.map((p) => ({ value: [p[0], p[1]], frame: p[2], label: p[3], ...extra }));
  const series = [
    tasSeries(F),
    { type: "lines", name: "已覆盖段", coordinateSystem: "geo", polyline: true, silent: true, z: 3, data: [{ coords: F.covered }], lineStyle: { color: "#7ed69a", width: 3.2, opacity: 0.9 } },
    { type: "scatter", name: "TAS 帧", coordinateSystem: "geo", z: 4, symbolSize: 5, itemStyle: { color: "rgba(233,237,247,.55)" },
      data: F.tas_pts.map((p) => ({ value: [p[0], p[1]], frame: p[2] })), tooltip: { formatter: (p) => `TAS 第 ${p.data.frame} 帧` } },
    { type: "scatter", name: "盘点", coordinateSystem: "geo", z: 6, symbol: "rect", symbolSize: 11, itemStyle: { color: "#1b1206", borderColor: AMBER, borderWidth: 1.5 },
      data: [...pt(F.checkpoints), ...pt([F.start], { symbol: "triangle" }), ...pt([F.end], { symbol: "diamond", symbolSize: 16, itemStyle: { color: "#ff9f1c" } })],
      label: { show: true, formatter: (p) => p.data.label, color: INK, fontSize: 11, position: "right", textBorderColor: "#000", textBorderWidth: 2 },
      tooltip: { formatter: (p) => `${p.data.label}<br>TAS 第 ${p.data.frame} 帧` } },
    ...runs.map((r) => ({
      type: "scatter", name: `${r} 开局点`, coordinateSystem: "geo", z: 7, symbolSize: 8,
      itemStyle: { color: runColor(r), borderColor: "#000", borderWidth: 0.6 },
      data: F.history.filter((h) => h.run === r).map((h) => ({ value: [h.x, h.z], ...h })),
      tooltip: { formatter: (p) => `<b>${p.data.run}</b> 课程开局点<br>第 ${p.data.frame} 帧<br>${p.data.t}（${S.tz}）` },
    })),
    { type: "scatter", name: "当前开局点", coordinateSystem: "geo", z: 9, symbolSize: 18, itemStyle: { color: "#ff6b2d", borderColor: "#fff1c4", borderWidth: 2 },
      data: [{ value: [F.current.x, F.current.z], ...F.current }],
      label: { show: true, formatter: (p) => `当前 ${p.data.frame}`, color: AMBER, fontWeight: "bold", fontSize: 13, position: "right", textBorderColor: "#000", textBorderWidth: 3 },
      tooltip: { formatter: (p) => `<b>当前（${p.data.run}）</b><br>第 ${p.data.frame} 帧<br>智能体已能接手全程的 ${p.data.covered_pct}%` } },
  ];
  if (F.alt) series.splice(2, 0, { type: "lines", name: "鬼躲石变参考线", coordinateSystem: "geo", polyline: true, silent: true, z: 3, data: [{ coords: F.alt }], lineStyle: { color: "#7ed69a", width: 3, type: "dotted" } });
  apply(ch, { animation: false, tooltip: TT, legend: { ...LEG, data: series.map((s) => s.name) }, geo: { ...geoBase(roamOn[id]), top: 40 }, series }, full);
}

function trajTooltip(d) {
  const o = OUT[d.end] || OUT.stopped;
  return `<b>${d.run}</b> · <span style="color:${o.c}">${o.n}</span><br>开始 ${d.t}（${S.tz}）<br>接手帧 ${d.k}，结束帧 ${d.end_frame}` +
    (d.sr ? `<br>SR ${(d.sr / 1000).toFixed(3)} s` : "") + `<br>死亡 ${d.deaths.length} 次 · ${d.backend}`;
}

function plasma(t) {
  const stops = [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const c = stops[i].map((v, j) => Math.round(v + (stops[i + 1][j] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const mapState = { "m-attempts": { picked: null, outcomes: { finish: true, gameover: true, stopped: true } }, "m-history": { picked: null, outcomes: { finish: true, gameover: true, stopped: true }, t0: 0, t1: 1 } };

function trajSeries(trs, colorOf, groupOf, groups) {
  const series = [];
  for (const [gname, gcolor] of groups) {
    const items = [];
    for (const d of trs.filter((x) => groupOf(x) === gname)) {
      const col = colorOf(d);
      for (const s of d.segs) items.push({ coords: s, d, lineStyle: { color: col } });
    }
    series.push({ type: "lines", name: gname, coordinateSystem: "geo", polyline: true, z: 4, data: items, color: gcolor,
      lineStyle: { width: 1.2, opacity: 0.6 }, emphasis: { focus: "none", lineStyle: { width: 3, opacity: 1 } },
      tooltip: { formatter: (p) => trajTooltip(p.data.d) } });
  }
  series.push({ type: "scatter", name: "接手点", coordinateSystem: "geo", z: 6, symbolSize: 6, itemStyle: { color: "#fff1c4", borderColor: "#000", borderWidth: 0.6 },
    data: trs.map((d) => ({ value: d.start, d })), tooltip: { formatter: (p) => `接手点：第 ${p.data.d.k} 帧<br>` + trajTooltip(p.data.d) } });
  series.push({ type: "scatter", name: "死亡", coordinateSystem: "geo", z: 7, symbol: "path://M0,0L10,10M10,0L0,10", symbolSize: 8, itemStyle: { color: "#ff6b5e", borderColor: "#ff6b5e", borderWidth: 2 },
    data: trs.flatMap((d) => d.deaths.map((x) => ({ value: [x[0], x[1]], frame: x[2], d }))), tooltip: { formatter: (p) => `死亡：第 ${p.data.frame} 帧<br>` + trajTooltip(p.data.d) } });
  return series;
}

function pickedSeries(st) {
  const d = st.picked;
  return { type: "lines", id: "picked", name: "选中", coordinateSystem: "geo", polyline: true, z: 10, silent: true,
    data: d ? d.segs.map((s) => ({ coords: s })) : [], lineStyle: { color: "#fff6c2", width: 3.4, opacity: 1, shadowColor: "#000", shadowBlur: 4 } };
}

function outcomeChecks(id) {
  const st = mapState[id];
  return Object.entries(OUT).map(([k, o]) => `<label><input type="checkbox" data-o="${k}"${st.outcomes[k] ? " checked" : ""}><span style="color:${o.c}">■</span>${o.n}</label>`).join("");
}

function wirePick(ch, id, pickBox) {
  if (ch._wired) return;
  ch._wired = true;
  ch.on("click", (p) => {
    const d = p.data && p.data.d;
    const st = mapState[id];
    st.picked = d && (!st.picked || st.picked.id !== d.id) ? d : null;
    ch.setOption({ series: [pickedSeries(st)] });
    $(pickBox).innerHTML = st.picked ? "选中：" + trajTooltip(st.picked).replace(/<br>/g, " · ") : "点击一条轨迹可高亮并查看详情。";
  });
}

async function renderAttempts(force, full) {
  const name = "attempts_" + sel;
  const A = await getPayload(name);
  const F = await getPayload("frontier");
  const id = "m-attempts", ch = chart(id), box = $("t-attempts"), st = mapState[id];
  if (!A || !F || !(await ensureGeo())) { ch.clear(); ch._drawn = false; ch._run = null; box.innerHTML = "<span>这个 run 没有记录轨迹。</span>"; box.dataset.wired = ""; return; }
  if (!force && box.dataset.shown === name + P[name].v) return;
  box.dataset.shown = name + P[name].v;
  box.innerHTML = `<span>最新 ${A.traj.length} 局 / 共记录 ${A.total} 局</span>${outcomeChecks(id)}${roamToggle(id)}`;
  wireRoam(box, id, () => renderAttempts(true, true));
  box.querySelectorAll("[data-o]").forEach((c) => (c.onchange = () => { st.outcomes[c.dataset.o] = c.checked; box.dataset.shown = ""; renderAttempts(true); }));
  const trs = A.traj.filter((d) => st.outcomes[d.end]);
  const groups = Object.entries(OUT).filter(([k]) => st.outcomes[k]).map(([k, o]) => [o.n, o.c]);
  const series = [tasSeries(F), ...trajSeries(trs, (d) => (OUT[d.end] || OUT.stopped).c, (d) => (OUT[d.end] || OUT.stopped).n, groups), pickedSeries(st)];
  if (ch._run !== sel) { st.picked = null; $("p-attempts").textContent = "点击一条轨迹可高亮并查看详情。"; series[series.length - 1] = pickedSeries(st); }
  apply(ch, { animation: false, tooltip: TT, legend: { ...LEG, data: series.filter((s) => s.name !== "选中").map((s) => s.name) }, geo: { ...geoBase(roamOn[id]), top: 40 }, series }, full || ch._run !== sel);
  ch._run = sel;
  wirePick(ch, id, "p-attempts");
}

async function renderHistory(force, full) {
  const H = await getPayload("history");
  const F = await getPayload("frontier");
  const id = "m-history", ch = chart(id), box = $("t-history"), st = mapState[id];
  if (!H || !F || !(await ensureGeo())) return;
  if (!force && box.dataset.shown === String(P.history.v)) return;
  box.dataset.shown = String(P.history.v);
  const ts = H.traj.map((d) => d.ts);
  const tmin = Math.min(...ts), tmax = Math.max(...ts), span = Math.max(1, tmax - tmin);
  const hrs = (t) => ((t - tmin) / 3600).toFixed(1);
  const lo = tmin + st.t0 * span, hi = tmin + st.t1 * span;
  box.innerHTML = `<span>按时间抽 ${H.traj.length} 局 / 共记录 ${H.total} 局</span>${outcomeChecks(id)}
    <label>时间从 <input type="range" min="0" max="1000" value="${Math.round(st.t0 * 1000)}" data-t="0"> 到 <input type="range" min="0" max="1000" value="${Math.round(st.t1 * 1000)}" data-t="1"></label>
    <span id="h-range">${hrs(lo)} – ${hrs(hi)} 小时</span>${roamToggle(id)}`;
  wireRoam(box, id, () => renderHistory(true, true));
  box.querySelectorAll("[data-o]").forEach((c) => (c.onchange = () => { st.outcomes[c.dataset.o] = c.checked; box.dataset.shown = ""; renderHistory(true); }));
  box.querySelectorAll("[data-t]").forEach((c) => {
    c.oninput = () => { const v = c.value / 1000; if (c.dataset.t === "0") st.t0 = Math.min(v, st.t1); else st.t1 = Math.max(v, st.t0); $("h-range").textContent = `${hrs(tmin + st.t0 * span)} – ${hrs(tmin + st.t1 * span)} 小时`; };
    c.onchange = () => { box.dataset.shown = ""; renderHistory(true); };
  });
  const trs = H.traj.filter((d) => st.outcomes[d.end] && d.ts >= lo - 1 && d.ts <= hi + 1);
  const groups = H.runs.map((r) => [r, runColor(r)]);
  const series = [tasSeries(F), ...trajSeries(trs, (d) => plasma((d.ts - tmin) / span), (d) => d.run, groups), pickedSeries(st)];
  apply(ch, {
    animation: false, tooltip: TT, legend: { ...LEG, data: series.filter((s) => s.name !== "选中").map((s) => s.name) }, geo: { ...geoBase(roamOn[id]), top: 40 }, series,
    visualMap: { type: "continuous", show: true, min: 0, max: +(span / 3600).toFixed(1), seriesIndex: [], calculable: false, orient: "horizontal", right: 10, bottom: 6, itemWidth: 10, itemHeight: 120,
      text: ["新", "旧（小时）"], textStyle: { color: INK, fontSize: 11 }, inRange: { color: [0, 0.25, 0.5, 0.75, 1].map(plasma) } },
  }, full);
  wirePick(ch, id, "p-history");
}

/* ----------------------------------------------------------------------------------- curves */
let curveMode = { log: true };
async function renderCurves(force, full) {
  const C = await getPayload("curves");
  const F = await getPayload("frontier");
  if (!C) return;
  const id = "c-curves", ch = chart(id), box = $("t-curves");
  const runs = Object.keys(C);
  if (!box.dataset.wired) {
    box.innerHTML = `<button type="button" class="tbtn${curveMode.log ? " on" : ""}" data-c="log">横轴对数</button>
      <button type="button" class="tbtn" data-c="all">全部</button><button type="button" class="tbtn" data-c="run">当前 run</button><button type="button" class="tbtn" data-c="recent">最近 1500 次更新</button>
      <span>拖动下方滑条或滚轮缩放；点击图例开关 run；虚线为“区间开局通关率”。</span>`;
    box.querySelectorAll("[data-c]").forEach((b) => (b.onclick = () => {
      const c = b.dataset.c;
      if (c === "log") { curveMode.log = !curveMode.log; b.classList.toggle("on", curveMode.log); renderCurves(true, true); return; }
      zoomCurves(c);
    }));
    box.dataset.wired = 1;
  }
  if (!force && box.dataset.shown === String(P.curves.v)) return;
  box.dataset.shown = String(P.curves.v);
  const panels = [
    ["frontier", "课程开局点（TAS 帧）"], ["band", "开局区间成功率"], ["finish", "回合通关率"], ["entropy", "策略熵"], ["sps", "每秒步数"],
  ];
  const n = panels.length, top0 = 58, gap = 26, bottom = 64;
  const H = $(id).clientHeight || 860;
  const ph = (H - top0 - bottom - gap * (n - 1)) / n;
  const grids = panels.map((p, i) => ({ left: 58, right: 14, top: top0 + i * (ph + gap), height: ph }));
  const xAxis = panels.map((p, i) => ({ type: curveMode.log ? "log" : "value", gridIndex: i, min: "dataMin", max: "dataMax", axisLabel: { show: i === n - 1, color: MUTED, formatter: (v) => (v >= 1 ? (+v).toFixed(0) : (+v).toPrecision(1)) + (i === n - 1 ? " M" : "") }, axisLine: { lineStyle: { color: BRASS } }, splitLine: { show: false }, name: i === n - 1 ? "训练步数（百万）" : "", nameLocation: "middle", nameGap: 26, nameTextStyle: { color: MUTED } }));
  const yAxis = panels.map((p, i) => ({ type: "value", gridIndex: i, name: p[1], nameTextStyle: { color: AMBER, align: "left", fontSize: 12 }, nameGap: 6, axisLabel: { color: MUTED, fontSize: 10.5 }, splitLine: { lineStyle: { color: "rgba(255,220,150,.07)" } },
    ...(i === 0 ? { min: 0, max: 20500 } : i === 1 || i === 2 ? { min: 0, max: 1 } : { scale: true }) }));
  const series = [];
  panels.forEach(([k], i) => {
    for (const r of runs) {
      const d = C[r];
      if (!d[k]) continue;
      const pts = d.steps.map((s, j) => [s, d[k][j], d.update[j]]).filter((p) => p[1] != null && p[0] > 0);
      series.push({ type: "line", name: r, xAxisIndex: i, yAxisIndex: i, data: pts, showSymbol: false, sampling: "lttb", step: k === "frontier" ? "end" : undefined, lineStyle: { width: 1.6 }, color: runColor(r), emphasis: { disabled: true },
        markLine: i === 0 && r === runs[0] && F ? { symbol: "none", silent: true, lineStyle: { color: "rgba(255,255,255,.35)", type: "dashed" }, label: { color: MUTED, fontSize: 10, position: "insideStartTop" },
          data: [...F.checkpoints.map((c) => ({ yAxis: c[2], name: c[3], label: { formatter: c[3] } })), { yAxis: F.start[2], label: { formatter: "出生点" } }] } :
          i === 1 && r === runs[0] ? { symbol: "none", silent: true, lineStyle: { color: "#ff7a66", type: "dotted" }, label: { formatter: "推进阈值 0.6", color: "#ff7a66", fontSize: 10, position: "insideStartTop" }, data: [{ yAxis: 0.6 }] } : undefined });
      if (k === "finish" && d.band_finish) {
        const pb = d.steps.map((s, j) => [s, d.band_finish[j], d.update[j]]).filter((p) => p[1] != null && p[0] > 0);
        series.push({ type: "line", name: r, xAxisIndex: i, yAxisIndex: i, data: pb, showSymbol: false, sampling: "lttb", lineStyle: { width: 1.1, type: "dashed" }, color: runColor(r), emphasis: { disabled: true }, extra: "band_finish" });
      }
    }
  });
  const xi = panels.map((p, i) => i);
  apply(ch, {
    animation: false, textStyle: { color: INK }, grid: grids, xAxis, yAxis, series,
    legend: { ...LEG, data: runs },
    axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: AMBER } },
    tooltip: { trigger: "axis", backgroundColor: "rgba(20,14,6,.95)", borderColor: BRASS, textStyle: { color: INK, fontSize: 12 }, confine: true,
      formatter: (ps) => {
        if (!ps.length) return "";
        const lines = ps.map((p) => {
          const s = series[p.seriesIndex], i = s.xAxisIndex, v = p.value[1];
          const lab = `${panels[i][1].replace(/（.*）/, "")}${s.extra === "band_finish" ? "（区间开局）" : ""}`;
          return `<span style="color:${p.color}">●</span> ${p.seriesName} · ${lab}：<b>${i === 0 || i === 4 ? num(v) : (+v).toFixed(3)}</b>`;
        });
        return `<b>${(+ps[0].value[0]).toFixed(2)} M 步</b> <span style="color:${MUTED}">（第 ${num(ps[0].value[2])} 次更新）</span><br>` + lines.join("<br>");
      } },
    dataZoom: [{ type: "inside", xAxisIndex: xi, filterMode: "none" }, { type: "slider", xAxisIndex: xi, filterMode: "none", bottom: 6, height: 18, borderColor: BRASS, textStyle: { color: MUTED }, fillerColor: "rgba(255,207,90,.15)", handleStyle: { color: AMBER } }],
  }, full);
}

function zoomCurves(mode) {
  const C = P.curves && P.curves.data, ch = charts["c-curves"];
  if (!C || !ch) return;
  if (mode === "all") { ch.dispatchAction({ type: "dataZoom", start: 0, end: 100 }); return; }
  const d = C[sel];
  if (!d) return;
  let s0 = d.steps[0];
  if (mode === "recent") {
    const r = (S.sr || {})[sel];
    const u0 = r && r.update_range ? r.update_range[0] : d.update[0];
    for (const rn of Object.keys(C)) { const x = C[rn]; const j = x.update.findIndex((u) => u >= u0); if (j >= 0) { s0 = Math.min(s0, x.steps[j]); break; } }
  }
  const s1 = d.steps[d.steps.length - 1];
  ch.dispatchAction({ type: "dataZoom", startValue: s0, endValue: s1 });
}

/* ------------------------------------------------------------------------------------ render */
const hasCharts = typeof window.echarts !== "undefined";
async function render() {
  for (const f of [renderHeader, renderReport, renderHighlight, renderStatus]) { try { f(); } catch (e) { console.error(e); } }
  if (!hasCharts) {
    document.querySelectorAll(".map, .chart").forEach((el) => { el.style.height = "auto"; el.innerHTML = '<p class="note" style="padding:12px">图表库没能加载（需要能访问 cdn.jsdelivr.net），只显示文字与表格。</p>'; });
  }
  const jobs = [() => renderSR(), () => renderFrontier(), () => renderAttempts(), () => renderCurves(), () => renderHistory()];
  for (const j of jobs) { try { if (hasCharts || j === jobs[0]) await j(); } catch (e) { console.error(e); } }
}

load();
