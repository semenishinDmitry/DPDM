"use strict";

// ====================================================================== utils
const $ = (s, el = document) => el.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
const SVG = "http://www.w3.org/2000/svg";
const sv = (tag, attrs = {}) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok || j.error) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
function fmt(v, p = 3) {
  if (v == null || v === "") return "—";
  v = Number(v);
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e-3 && a < 1e5) return String(+v.toPrecision(p));
  const e = Math.floor(Math.log10(a));
  const m = +(v / 10 ** e).toPrecision(p);
  const es = String(e).split("").map(c => SUP[c]).join("");
  return (m === 1 ? "" : m === -1 ? "−" : m + "·") + "10" + es;
}
function fmtYears(t) {
  if (t == null || !isFinite(t)) return { v: "—", u: "" };
  if (t >= 1e6) return { v: fmt(t / 1e6, 3), u: "млн лет" };
  if (t >= 1e4) return { v: fmt(t / 1e3, 3), u: "тыс. лет" };
  return { v: fmt(t, 4), u: "лет" };
}
const store = {
  get(k) { try { return JSON.parse(localStorage.getItem("dustppd." + k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem("dustppd." + k, JSON.stringify(v)); } catch { /* ignore */ } },
};
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ====================================================================== charts
const tooltip = () => $("#tooltip");

function niceLinearTicks(lo, hi, n = 6) {
  const span = hi - lo;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
function logTicks(lo, hi) {
  const a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi));
  if (b - a >= 2) {
    const step = Math.max(1, Math.ceil((b - a) / 6));
    const out = [];
    for (let k = a; k <= b; k += step) { const v = 10 ** k; if (v >= lo * 0.999 && v <= hi * 1.001) out.push({ v, dec: k }); }
    return out;
  }
  const out = [];
  for (let k = a - 1; k <= b; k++) for (const m of [1, 2, 5]) {
    const v = m * 10 ** k;
    if (v >= lo * 0.999 && v <= hi * 1.001) out.push({ v, dec: m === 1 ? k : null });
  }
  return out.length >= 2 ? out : niceLinearTicks(lo, hi).map(v => ({ v, dec: null }));
}

class Chart {
  constructor(parent, { title = "", options = null, value = null, onSelect = null, height = 270 } = {}) {
    this.height = height;
    this.zoom = null;
    this.data = null;
    this.root = el("div", { class: "chart" });
    const head = el("div", { class: "chart-head" });
    if (options) {
      this.sel = el("select", { "aria-label": "что показать" });
      for (const [k, o] of Object.entries(options)) this.sel.append(el("option", { value: k }, o.title));
      this.sel.value = value;
      this.sel.addEventListener("change", () => { this.zoom = null; onSelect && onSelect(this.sel.value); });
      head.append(this.sel, el("div", { class: "ct" }));
    } else {
      head.append(el("div", { class: "ct" }, title));
    }
    head.append(el("button", { class: "btn small ghost", type: "button", title: "Сбросить масштаб (двойной щелчок по графику)", onclick: () => { this.zoom = null; this.render(); } }, "⤢"));
    head.append(el("button", { class: "btn small ghost", type: "button", title: "Скачать SVG", onclick: () => this.download() }, "SVG"));
    this.svg = sv("svg");
    this.root.append(head, this.svg);
    parent.append(this.root);
    this._bind();
  }

  set(data) { this.data = data; this.zoom = null; this.render(); }

  _domain(vals, log) {
    let lo = Infinity, hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) return null;
    if (log) {
      if (lo === hi) { lo /= 2; hi *= 2; }
      const p = (Math.log10(hi) - Math.log10(lo)) * 0.04;
      return [lo / 10 ** p, hi * 10 ** p];
    }
    if (lo === hi) { const d = Math.abs(lo) || 1; lo -= d * 0.5; hi += d * 0.5; }
    const p = (hi - lo) * 0.05;
    return [lo - p, hi + p];
  }

  render() {
    const svg = this.svg, d = this.data;
    const W = Math.max(280, this.root.clientWidth - 20), H = this.height;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    svg.replaceChildren();
    const C = { ink: css("--ink"), muted: css("--muted"), line: css("--line"), grid: css("--grid"), accent: css("--accent"), panel: css("--panel") };
    svg.append(sv("rect", { x: 0, y: 0, width: W, height: H, fill: C.panel }));
    if (!d) return;
    const M = { l: 62, r: 26, t: 12, b: 38 };
    const pw = W - M.l - M.r, ph = H - M.t - M.b;
    this.geom = { M, pw, ph };

    // clean series
    const series = d.series.map(s => {
      const xs = [], ys = [];
      for (let i = 0; i < s.x.length; i++) {
        let x = s.x[i], y = s.y[i];
        if (x == null || y == null || !isFinite(x) || !isFinite(y)) { xs.push(null); ys.push(null); continue; }
        if (s.abs) y = Math.abs(y);
        if ((d.xlog && x <= 0) || (d.ylog && y <= 0)) { xs.push(null); ys.push(null); continue; }
        xs.push(x); ys.push(y);
      }
      return { ...s, xs, ys };
    });
    if (d.ylog && d.yminRel) {  // hide far tails of log-scaled densities
      const top = Math.max(...series.flatMap(s => s.ys.filter(v => v != null)));
      for (const s of series) s.ys.forEach((v, i) => { if (v != null && v < top * d.yminRel) { s.ys[i] = null; s.xs[i] = null; } });
    }
    const allx = series.flatMap(s => s.xs.filter(v => v != null));
    const ally = series.flatMap(s => s.ys.filter(v => v != null));
    let xd = this._domain(allx, d.xlog), yd = this._domain(ally, d.ylog);
    if (!xd || !yd) {
      const t = sv("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: C.muted, "font-size": 12 });
      t.textContent = d.empty || "нет данных для этого графика";
      svg.append(t);
      this.scr = null;
      return;
    }
    if (this.zoom) { xd = this.zoom.x; yd = this.zoom.y; }
    const fx = d.xlog ? v => Math.log10(v) : v => v, fy = d.ylog ? v => Math.log10(v) : v => v;
    const [x0, x1] = xd.map(fx), [y0, y1] = yd.map(fy);
    const X = v => M.l + (fx(v) - x0) / (x1 - x0) * pw;
    const Y = v => M.t + ph - (fy(v) - y0) / (y1 - y0) * ph;
    this.inv = {
      x: px => { const u = x0 + (px - M.l) / pw * (x1 - x0); return d.xlog ? 10 ** u : u; },
      y: py => { const u = y0 + (M.t + ph - py) / ph * (y1 - y0); return d.ylog ? 10 ** u : u; },
    };

    const defs = sv("defs");
    const clip = sv("clipPath", { id: "c" + Math.random().toString(36).slice(2) });
    clip.append(sv("rect", { x: M.l, y: M.t, width: pw, height: ph }));
    defs.append(clip);
    svg.append(defs);

    // grid + ticks
    const g = sv("g", { "font-family": "ui-monospace, Menlo, monospace", "font-size": 11 });
    const xt = d.xlog ? logTicks(...xd) : niceLinearTicks(...xd).map(v => ({ v, dec: null }));
    const yt = d.ylog ? logTicks(...yd) : niceLinearTicks(...yd, 5).map(v => ({ v, dec: null }));
    const label = (t, x, y, anchor) => {
      const e = sv("text", { x, y, "text-anchor": anchor, fill: C.muted });
      if (t.dec != null && (d.xlog || d.ylog) && Math.abs(t.dec) >= 3) {
        e.append(document.createTextNode("10"));
        const s = sv("tspan", { dy: -5, "font-size": 8.5 }); s.textContent = String(t.dec).replace("-", "−"); e.append(s);
      } else e.textContent = fmt(t.v, 3);
      return e;
    };
    for (const t of xt) {
      const x = X(t.v); if (x < M.l - 1 || x > M.l + pw + 1) continue;
      g.append(sv("line", { x1: x, x2: x, y1: M.t, y2: M.t + ph, stroke: C.grid }));
      g.append(label(t, x, M.t + ph + 15, "middle"));
    }
    for (const t of yt) {
      const y = Y(t.v); if (y < M.t - 1 || y > M.t + ph + 1) continue;
      g.append(sv("line", { x1: M.l, x2: M.l + pw, y1: y, y2: y, stroke: C.grid }));
      g.append(label(t, M.l - 6, y + 4, "end"));
    }
    g.append(sv("rect", { x: M.l, y: M.t, width: pw, height: ph, fill: "none", stroke: C.line }));
    if (!d.ylog && yd[0] < 0 && yd[1] > 0) g.append(sv("line", { x1: M.l, x2: M.l + pw, y1: Y(0), y2: Y(0), stroke: C.muted, "stroke-opacity": .5 }));
    svg.append(g);
    const xl = sv("text", { x: M.l + pw / 2, y: H - 4, "text-anchor": "middle", fill: C.muted, "font-size": 11.5 });
    xl.textContent = d.xlabel; svg.append(xl);
    const yl = sv("text", { x: 12, y: M.t + ph / 2, "text-anchor": "middle", fill: C.muted, "font-size": 11.5, transform: `rotate(-90 12 ${M.t + ph / 2})` });
    yl.textContent = d.ylabel; svg.append(yl);

    // markers (vertical lines)
    const plot = sv("g", { "clip-path": `url(#${clip.id})` });
    for (const m of d.markers || []) {
      if (m.x == null || (d.xlog && m.x <= 0)) continue;
      const x = X(m.x);
      plot.append(sv("line", { x1: x, x2: x, y1: M.t, y2: M.t + ph, stroke: m.color || C.accent, "stroke-dasharray": "4 3", "stroke-width": 1.2 }));
      const t = sv("text", { x: x + 4, y: M.t + 12, fill: m.color || C.accent, "font-size": 11 });
      t.textContent = m.label; plot.append(t);
    }
    // lines
    this.scr = [];
    series.forEach((s, k) => {
      const color = s.color || css(`--s${(k % 5) + 1}`);
      let path = "", pen = false;
      const sx = [], sy = [];
      for (let i = 0; i < s.xs.length; i++) {
        if (s.xs[i] == null) { pen = false; continue; }
        const px = X(s.xs[i]), py = Y(s.ys[i]);
        sx.push(px); sy.push(py);
        path += (pen ? "L" : "M") + px.toFixed(1) + " " + py.toFixed(1);
        pen = true;
      }
      if (s.points) {
        for (let i = 0; i < sx.length; i++) plot.append(sv("circle", { cx: sx[i], cy: sy[i], r: 2.4, fill: color }));
      } else {
        plot.append(sv("path", { d: path, fill: "none", stroke: color, "stroke-width": s.width || 1.7, "stroke-linejoin": "round", "stroke-dasharray": s.dash || "" }));
      }
      this.scr.push({ name: s.name, color, sx, sy, x: s.xs.filter(v => v != null), y: s.ys.filter(v => v != null) });
    });
    svg.append(plot);

    // legend
    const named = series.filter(s => s.name);
    if (named.length > 1 || (named.length === 1 && d.legend)) {
      const lg = sv("g");
      const lw = 8 + Math.max(...named.map(s => s.name.length)) * 6.6 + 30, lh = named.length * 16 + 6;
      const pos = d.legend || "tr";
      const bx = pos.endsWith("l") ? M.l + 6 : M.l + pw - lw - 6, by = pos.startsWith("b") ? M.t + ph - lh - 6 : M.t + 4;
      let y = by + 4;
      lg.append(sv("rect", { x: bx, y: by, width: lw, height: lh, rx: 6, fill: C.panel, "fill-opacity": .88, stroke: C.line }));
      named.forEach((s) => {
        const k = series.indexOf(s);
        const color = s.color || css(`--s${(k % 5) + 1}`);
        const x = bx + 6;
        lg.append(sv("line", { x1: x, x2: x + 20, y1: y + 5, y2: y + 5, stroke: color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }));
        const t = sv("text", { x: x + 26, y: y + 9, fill: C.ink, "font-size": 11.5 }); t.textContent = s.name; lg.append(t);
        y += 16;
      });
      svg.append(lg);
    }
    this.cross = sv("g", { visibility: "hidden" });
    this.crossV = sv("line", { stroke: C.muted, "stroke-dasharray": "2 3" });
    this.crossDot = sv("circle", { r: 4, fill: C.panel, "stroke-width": 2 });
    this.cross.append(this.crossV, this.crossDot);
    this.zbox = sv("rect", { visibility: "hidden", fill: css("--blue"), "fill-opacity": .12, stroke: css("--blue"), "stroke-dasharray": "3 3" });
    svg.append(this.cross, this.zbox);
  }

  _pt(ev) {
    const r = this.svg.getBoundingClientRect();
    const vb = this.svg.viewBox.baseVal;
    return { x: (ev.clientX - r.left) * vb.width / r.width, y: (ev.clientY - r.top) * vb.height / r.height };
  }

  _bind() {
    let drag = null;
    this.svg.addEventListener("mousemove", ev => {
      if (!this.scr || !this.geom) return;
      const p = this._pt(ev), { M, pw, ph } = this.geom;
      if (drag) {
        const x = Math.min(drag.x, p.x), y = Math.min(drag.y, p.y);
        Object.entries({ x, y, width: Math.abs(p.x - drag.x), height: Math.abs(p.y - drag.y), visibility: "visible" }).forEach(([k, v]) => this.zbox.setAttribute(k, v));
      }
      if (p.x < M.l || p.x > M.l + pw || p.y < M.t || p.y > M.t + ph) { this._hide(); return; }
      let best = null;
      for (const s of this.scr) for (let i = 0; i < s.sx.length; i++) {
        const dd = (s.sx[i] - p.x) ** 2 + 0.25 * (s.sy[i] - p.y) ** 2;
        if (!best || dd < best.dd) best = { dd, s, i };
      }
      if (!best) return;
      const { s, i } = best;
      this.cross.setAttribute("visibility", "visible");
      Object.entries({ x1: s.sx[i], x2: s.sx[i], y1: M.t, y2: M.t + ph }).forEach(([k, v]) => this.crossV.setAttribute(k, v));
      this.crossDot.setAttribute("cx", s.sx[i]); this.crossDot.setAttribute("cy", s.sy[i]); this.crossDot.setAttribute("stroke", s.color);
      const tt = tooltip();
      tt.hidden = false;
      tt.textContent = (s.name ? s.name + "\n" : "") + `${this.data.xshort || "x"} = ${fmt(s.x[i], 5)}\n${this.data.yshort || "y"} = ${fmt(s.y[i], 5)}`;
      tt.style.left = Math.min(ev.clientX + 14, innerWidth - tt.offsetWidth - 8) + "px";
      tt.style.top = (ev.clientY + 14) + "px";
    });
    this.svg.addEventListener("mouseleave", () => { this._hide(); drag = null; this.zbox && this.zbox.setAttribute("visibility", "hidden"); });
    this.svg.addEventListener("mousedown", ev => { if (this.scr) { drag = this._pt(ev); ev.preventDefault(); } });
    this.svg.addEventListener("mouseup", ev => {
      if (!drag) return;
      const p = this._pt(ev);
      if (Math.abs(p.x - drag.x) > 8 && Math.abs(p.y - drag.y) > 8) {
        const xs = [this.inv.x(drag.x), this.inv.x(p.x)].sort((a, b) => a - b);
        const ys = [this.inv.y(drag.y), this.inv.y(p.y)].sort((a, b) => a - b);
        this.zoom = { x: xs, y: ys };
        this.render();
      }
      drag = null;
      this.zbox.setAttribute("visibility", "hidden");
    });
    this.svg.addEventListener("dblclick", () => { this.zoom = null; this.render(); });
  }

  _hide() { if (this.cross) this.cross.setAttribute("visibility", "hidden"); tooltip().hidden = true; }

  download() {
    const s = new XMLSerializer().serializeToString(this.svg).replace("<svg", `<svg xmlns="${SVG}"`);
    const a = el("a", { href: URL.createObjectURL(new Blob([s], { type: "image/svg+xml" })), download: (this.data && this.data.file || "dustppd_plot") + ".svg" });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
}

// ====================================================================== trajectory plots
const TRAJ = {
  r_t:   { title: "r(t) — радиус орбиты",       x: "t_yr", y: "r_au",  xlog: true,  ylog: true,  xl: "t, лет", yl: "r, а.е." },
  z_r:   { title: "z(r) — высота над экватором", x: "r_au", y: "z_au",  xlog: true,  ylog: false, xl: "r, а.е.", yl: "z, а.е." },
  z_t:   { title: "z(t) — оседание",            x: "t_yr", y: "z_au",  xlog: true,  ylog: false, xl: "t, лет", yl: "z, а.е." },
  vr_r:  { title: "|v_r|(r) — скорость дрейфа",  x: "r_au", y: "vr_cms", xlog: true, ylog: true,  xl: "r, а.е.", yl: "|v_r|, м/с", abs: true, scale: 0.01 },
  vr_t:  { title: "v_r(t) — радиальная скорость", x: "t_yr", y: "vr_cms", xlog: true, ylog: false, xl: "t, лет", yl: "v_r, м/с", scale: 0.01 },
  a_r:   { title: "a(r) — размер пылинки",       x: "r_au", y: "a_cm",  xlog: true,  ylog: true,  xl: "r, а.е.", yl: "a, см" },
  a_t:   { title: "a(t) — рост",                x: "t_yr", y: "a_cm",  xlog: true,  ylog: true,  xl: "t, лет", yl: "a, см" },
  St_r:  { title: "St(r) — число Стокса",        x: "r_au", y: "St",    xlog: true,  ylog: true,  xl: "r, а.е.", yl: "St" },
  dv_r:  { title: "|v − u|(r) — относительная скорость", x: "r_au", y: "dv_cms", xlog: true, ylog: true, xl: "r, а.е.", yl: "|v − u|, м/с", scale: 0.01 },
  Re_r:  { title: "Re(r) — число Рейнольдса",    x: "r_au", y: "Re",    xlog: true,  ylog: true,  xl: "r, а.е.", yl: "Re" },
  beta_r:{ title: "β_K(r) вдоль траектории",     x: "r_au", y: "beta",  xlog: true,  ylog: false, xl: "r, а.е.", yl: "β_K" },
  hall_r:{ title: "Γ(r) — параметр Холла",       x: "r_au", y: "hall",  xlog: true,  ylog: true,  xl: "r, а.е.", yl: "Γ" },
  Z_r:   { title: "Z(r) — заряд, e",             x: "r_au", y: "Z_charge", xlog: true, ylog: false, xl: "r, а.е.", yl: "Z" },
};

const app = { schema: null, job: null, result: null, charts: [], diskCharts: [], mixCharts: [], polling: null };

function trajData(key) {
  const p = TRAJ[key], c = app.result.columns;
  const sc = p.scale || 1;
  const y = c[p.y].map(v => v == null ? null : v * sc);
  const markers = [];
  const s = app.result.summary;
  if (p.x === "r_au" && s.trapped) markers.push({ x: s.r_trap_au, label: "ловушка" });
  return { series: [{ x: c[p.x], y, abs: p.abs }], xlog: p.xlog, ylog: p.ylog, xlabel: p.xl, ylabel: p.yl,
           xshort: p.xl.split(",")[0], yshort: p.yl.split(",")[0], markers, file: "dustppd_" + key,
           empty: key === "hall_r" || key === "Z_r" ? "включите заряд пылинки (OML)" : undefined };
}

function defaultCharts(p) {
  return ["r_t", p.sedimentation ? "z_r" : "vr_t", "vr_r", p.coagulation ? "a_r" : (p.charge === "oml" ? "hall_r" : "St_r")];
}

function buildTrajCharts() {
  const host = $("#charts");
  host.replaceChildren();
  const saved = store.get("charts") || defaultCharts(getParams());
  app.charts = saved.map((k, i) => {
    const ch = new Chart(host, { options: TRAJ, value: TRAJ[k] ? k : "r_t", onSelect: v => {
      store.set("charts", app.charts.map(c => c.sel.value));  // the user's choice now overrides the defaults
      if (app.result) ch.set(trajData(v));
    } });
    return ch;
  });
}

function renderTrajectory(params) {
  if (!store.get("charts")) defaultCharts(params).forEach((k, i) => { app.charts[i].sel.value = k; });
  for (const ch of app.charts) ch.set(trajData(ch.sel.value));
}

function renderDisk(disk, summary, params) {
  const host = $("#disk-charts");
  host.replaceChildren();
  app.diskCharts = [];
  if (!disk) return;
  const r = disk.r_au;
  const traps = (disk.traps || []).map(t => ({ x: t.r_au, label: "ловушка" }));
  const zero = (disk.headwind_zero_au || []).map(x => ({ x, label: "max p", color: css("--s3") }));
  const mk = (title, data) => { const c = new Chart(host, { title }); app.diskCharts.push(c); c.set(data); };
  mk("β_K(r) — встречный ветер", { series: [{ x: r, y: disk.beta, name: "β_K" }], xlog: true, ylog: false,
       xlabel: "r, а.е.", ylabel: "β_K", xshort: "r", yshort: "β_K", markers: [...zero], file: "beta_r" });
  mk(`v_r стационарная для a = ${fmt(params.a_init_cm)} см`, { series: [{ x: r, y: disk.vr_stat_cms.map(v => v == null ? null : v / 100), name: "v_r" }],
       xlog: true, ylog: false, xlabel: "r, а.е.", ylabel: "v_r, м/с (> 0 — наружу)", xshort: "r", yshort: "v_r", markers: traps, file: "vr_stationary" });
  mk("Давление на экваторе p(r)", { series: [{ x: r, y: disk.p_mid }], xlog: true, ylog: true,
       xlabel: "r, а.е.", ylabel: "p, дин/см²", xshort: "r", yshort: "p", markers: zero, file: "p_mid" });
  const hasB = disk.Bz_G.some(v => v && Math.abs(v) > 0);
  if (hasB) {
    mk("Остаточное поле", { series: [{ x: r, y: disk.Bz_G, name: "B_z", abs: true }, { x: r, y: disk.Br_G, name: "|B_r|", abs: true, dash: "5 3" }],
         xlog: true, ylog: true, xlabel: "r, а.е.", ylabel: "B, Гс", xshort: "r", yshort: "B", file: "field" });
  } else {
    mk("Температура и толщина диска", { series: [{ x: r, y: disk.T_K, name: "T, К" }, { x: r, y: disk.H_au.map(v => v * 1000), name: "H, 10⁻³ а.е.", dash: "5 3" }],
         xlog: true, ylog: true, xlabel: "r, а.е.", ylabel: "T [К], H [10⁻³ а.е.]", xshort: "r", yshort: "значение", file: "T_H" });
  }
}

function card(k, v, u = "", d = "", cls = "") {
  return el("div", { class: "card " + cls }, el("div", { class: "k" }, k),
    el("div", { class: "v", html: `${v}${u ? `<small>${u}</small>` : ""}` }), d ? el("div", { class: "d" }, d) : null);
}

const STATUS_RU = {
  success: "завершён", terminal_event: "завершён", cancelled: "остановлен", max_steps_exceeded: "лимит шагов",
  step_size_underflow: "шаг исчез", time_resolution_lost: "потеря точности t", non_finite_state: "NaN / ∞",
  newton_failure: "Ньютон не сошёлся", linear_solver_failure: "ошибка линейного решателя",
  domain_violation: "выход из области", invalid_input: "неверные данные",
};

function renderSummary(res, params) {
  const s = res.summary, st = s.stats;
  const cards = $("#cards");
  cards.replaceChildren();
  cards.append(card("Статус", STATUS_RU[s.status] || s.status, "", s.message || s.final_method, s.success ? "good" : "bad"));
  if (s.reached_inner_edge) {
    const y = fmtYears(s.t_drift_yr); cards.append(card("Время дрейфа до r_in", y.v, y.u, `до ${fmt(params.r_in_au)} а.е.`, "acc"));
  } else {
    const y = fmtYears(s.t_final_yr); cards.append(card("Конец расчёта", y.v, y.u, `r = ${fmt(s.r_final_au, 4)} а.е.`));
  }
  if (s.trapped) cards.append(card("Ловушка", fmt(s.r_trap_au, 4), "а.е.", "частица остановилась", "good"));
  if (params.sedimentation) {
    const y = fmtYears(s.t_settle_first_yr);
    cards.append(card("Оседание к экватору", y.v, y.u, `последний вход: ${fmtYears(s.t_settle_last_yr).v} ${fmtYears(s.t_settle_last_yr).u}`));
  }
  cards.append(card("max |v_r|", fmt(s.vr_max_cms / 100, 3), "м/с", `при r = ${fmt(s.r_at_vr_max_au, 3)} а.е.`));
  if (params.coagulation) cards.append(card("Максимальный размер", fmt(s.a_max_cm, 3), "см", `конечный ${fmt(s.a_final_cm, 3)} см`));
  cards.append(card("Решатель", fmt(st.n_steps, 6), "шагов", `${(s.wall_time_s * 1000).toFixed(1)} мс · ${s.final_method}`));

  const tb = el("table");
  const rows = [
    ["итоговый метод", s.final_method], ["шагов", st.n_steps], ["отклонено (ошибка / Ньютон)", `${st.n_reject_error} / ${st.n_reject_newton}`],
    ["вычислений правой части", st.n_rhs], ["якобианов / LU", `${st.n_jac} / ${st.n_lu}`], ["итераций Ньютона", st.n_newton_iter],
    ["h_min … h_max (код. ед.)", `${fmt(st.h_min)} … ${fmt(st.h_max)}`], ["режим Стокса встречался", s.stokes_regime_seen ? `да (max Re ${fmt(s.max_Re)})` : "нет"],
    ["точек траектории", res.n_points], ["время счёта", `${(s.wall_time_s * 1000).toFixed(2)} мс`],
  ];
  for (const [k, v] of rows) tb.append(el("tr", {}, el("td", {}, k), el("td", {}, String(v))));
  const box = $("#solver-stats");
  box.replaceChildren(tb);
  const ul = el("ul");
  const t0 = 0.15915494309189535;
  for (const w of st.switches || []) ul.append(el("li", {}, `переключение ${w.from} → ${w.to} при t = ${fmt(w.t * t0, 4)} лет: ${w.reason}`));
  if (ul.children.length) box.append(ul);
}

// ====================================================================== form
function fieldId(k) { return "f_" + k; }

function buildForm(schema) {
  const form = $("#form");
  form.replaceChildren();
  schema.groups.forEach((g, gi) => {
    const det = el("details", { class: "group" });
    if (gi < 2 || (store.get("open") || []).includes(g.title)) det.open = true;
    det.addEventListener("toggle", () => {
      const open = [...form.querySelectorAll("details.group")].filter(d => d.open).map(d => $("summary", d).textContent);
      store.set("open", open);
    });
    det.append(el("summary", {}, g.title));
    const box = el("div", { class: "fields" });
    for (const f of g.fields) {
      let input, row;
      const unit = f.unit ? el("i", {}, f.unit) : null;
      if (f.kind === "bool") {
        input = el("input", { type: "checkbox", id: fieldId(f.key), name: f.key });
        row = el("label", { class: "field check", "data-key": f.key }, input, el("span", {}, f.label));
      } else if (f.kind === "choice") {
        input = el("select", { id: fieldId(f.key), name: f.key });
        for (const c of f.choices) input.append(el("option", { value: c }, c));
        row = el("label", { class: "field", "data-key": f.key }, el("span", {}, f.label, unit), input);
      } else {
        input = el("input", { id: fieldId(f.key), name: f.key, spellcheck: "false", inputmode: f.kind === "path" ? "text" : "decimal" });
        if (f.kind === "path") input.placeholder = "/путь/к/таблице.csv";
        row = el("label", { class: "field" + (f.kind === "path" ? " wide" : ""), "data-key": f.key }, el("span", {}, f.label, unit), input);
      }
      input.addEventListener("input", onFormChange);
      input.addEventListener("change", onFormChange);
      box.append(row);
    }
    det.append(box);
    form.append(det);
  });
}

function setParams(p) {
  for (const g of app.schema.groups) for (const f of g.fields) {
    const inp = document.getElementById(fieldId(f.key));
    if (!inp || !(f.key in p)) continue;
    if (f.kind === "bool") inp.checked = !!p[f.key];
    else inp.value = p[f.key] == null ? "" : String(p[f.key]);
    inp.classList.remove("bad");
  }
  updateDim();
  store.set("params", getParams());
}

function getParams() {
  const out = {};
  for (const g of app.schema.groups) for (const f of g.fields) {
    const inp = document.getElementById(fieldId(f.key));
    if (!inp) continue;
    if (f.kind === "bool") out[f.key] = inp.checked;
    else if (f.kind === "float" || f.kind === "int") out[f.key] = inp.value.trim().replace(",", ".");
    else out[f.key] = inp.value;
  }
  return out;
}

function updateDim() {
  const p = getParams();
  const dim = (k, on) => { const r = document.querySelector(`[data-key="${k}"]`); if (r) r.classList.toggle("dim", !on); };
  const tab = p.disk_model === "tabulated" || p.disk_model === "tabulated_beta";
  dim("table_path", tab);
  dim("vphi2vk", p.disk_model === "analytic");
  const pert = p.perturbation !== "none";
  ["bump_amplitude", "bump_r_au", "bump_width_au"].forEach(k => dim(k, pert));
  const fieldUsed = p.charge === "oml" || (p.disk_model === "self_consistent" && p.magnetic_tension);
  ["B0_G", "field_exponent", "br_over_bz"].forEach(k => dim(k, fieldUsed));
  dim("magnetic_tension", p.disk_model === "self_consistent");
  dim("ion_mass_amu", p.charge === "oml");
  dim("linear_solver", p.method === "auto" || p.method === "bdf");
  dim("settle_eps", p.sedimentation);
}

function onFormChange(ev) {
  const t = ev.target;
  if (t.tagName === "INPUT" && t.type !== "checkbox" && t.inputMode === "decimal") {
    const v = t.value.trim().replace(",", ".");
    t.classList.toggle("bad", v === "" || !isFinite(Number(v)));
  }
  updateDim();
  store.set("params", getParams());
  $("#preset").value = "";
}

// ====================================================================== run
function setBusy(busy, progress) {
  $("#btn-run").disabled = busy;
  $("#btn-cancel").disabled = !busy;
  const pr = $("#progress");
  pr.classList.toggle("indet", busy && !(progress > 0));
  $(".bar", pr).style.width = busy ? (progress > 0 ? `${Math.round(progress * 100)}%` : "") : "0";
}

function showAlert(id, msg, warn = false) {
  const a = $(id);
  if (!msg) { a.hidden = true; return; }
  a.hidden = false; a.textContent = msg; a.classList.toggle("warn", warn);
}

async function runTrajectory() {
  if (app.polling) return;
  const params = getParams();
  showAlert("#alert", null);
  $("#status").textContent = "Запуск…";
  setBusy(true, 0);
  try {
    const { id } = await api("/api/run", { params });
    app.job = id;
    const t0 = performance.now();
    app.polling = true;
    while (true) {
      await new Promise(r => setTimeout(r, 120));
      const st = await api(`/api/status?id=${id}`);
      if (!st.done) {
        setBusy(true, st.progress);
        $("#status").textContent = `Считаем… ${st.progress > 0 ? Math.round(st.progress * 100) + "%" : ""} (${((performance.now() - t0) / 1000).toFixed(1)} с)`;
        continue;
      }
      if (st.error) throw new Error(st.error);
      app.result = st.result;
      const s = st.result.summary;
      $("#status").textContent = s.success ? `Готово: ${fmt(s.stats.n_steps, 6)} шагов, ${(s.wall_time_s * 1000).toFixed(1)} мс` : `Расчёт не завершён: ${s.status}`;
      if (!s.success && s.status !== "cancelled") showAlert("#alert", `Интегрирование не завершено (${s.status}): ${s.message}`);
      else if (s.warnings && s.warnings.length) showAlert("#alert", "Предупреждения:\n• " + s.warnings.join("\n• "), true);
      if (st.result.disk_error) showAlert("#alert", "Профиль диска не построен: " + st.result.disk_error, true);
      $("#page-run .content").classList.add("has-result");
      renderSummary(st.result, params);
      renderTrajectory(params);
      renderDisk(st.result.disk, s, params);
      $("#btn-csv").disabled = false;
      break;
    }
  } catch (e) {
    showAlert("#alert", e.message);
    $("#status").textContent = "Ошибка";
  } finally {
    app.polling = null;
    setBusy(false);
  }
}

// ====================================================================== mixing
async function runMixing() {
  const fd = new FormData($("#mix-form"));
  const options = Object.fromEntries([...fd.entries()].map(([k, v]) => [k, String(v).trim().replace(",", ".")]));
  store.set("mix", options);
  showAlert("#mix-alert", null);
  const btn = $("#btn-mix");
  btn.disabled = true;
  $("#mix-status").textContent = "Моделируем траектории…";
  try {
    const { id } = await api("/api/diffusion", { params: getParams(), options });
    let st;
    while (true) {
      await new Promise(r => setTimeout(r, 150));
      st = await api(`/api/status?id=${id}`);
      if (st.done) break;
    }
    if (st.error) throw new Error(st.error);
    const d = st.result;
    $("#mix-status").textContent = `Готово: ${d.n_paths} траекторий за ${d.wall_time_s.toFixed(2)} с`;
    if (d.warnings && d.warnings.length) showAlert("#mix-alert", "Предупреждения:\n• " + d.warnings.join("\n• "), true);
    $("#page-mix .content").classList.add("has-result");
    const cards = $("#mix-cards");
    const dub = (d.Hd_dubrulle / d.Hd_exact - 1) * 100;
    cards.replaceChildren(
      card("H_d / H — СДУ", fmt(d.Hd_sim, 4), "", `± ${fmt(d.Hd_err, 2)} (1σ)`, "acc"),
      card("H_d / H — точно", fmt(d.Hd_exact, 4), "", "стационарное решение"),
      card("H_d / H — Дюбрюль", fmt(d.Hd_dubrulle, 4), "", `${dub >= 0 ? "+" : ""}${dub.toFixed(1)} % к точному`, Math.abs(dub) > 5 ? "bad" : "good"),
      card("St у экватора", fmt(d.St_mid, 3), "", `α_t = ${fmt(d.alpha_t)}`),
      card("H газа", fmt(d.H_au, 3), "а.е.", `r = ${fmt(d.r_au)} а.е., a = ${fmt(d.a_cm)} см`),
    );
    const host = $("#mix-charts");
    host.replaceChildren();
    const c1 = new Chart(host, { title: "Установление толщины слоя" });
    c1.set({ series: [{ x: d.t_yr, y: d.z_rms_H, name: "СДУ: √⟨z²⟩ / H" },
                      { x: [d.t_yr[0], d.t_yr[d.t_yr.length - 1]], y: [d.Hd_exact, d.Hd_exact], name: "точно", dash: "5 3" },
                      { x: [d.t_yr[0], d.t_yr[d.t_yr.length - 1]], y: [d.Hd_dubrulle, d.Hd_dubrulle], name: "Дюбрюль", dash: "2 3" }],
             xlog: false, ylog: false, xlabel: "t, лет", ylabel: "H_d / H", xshort: "t", yshort: "H_d/H", file: "mixing_relaxation", legend: "br" });
    const hx = [], hy = [];
    const w = d.hist_z.length > 1 ? d.hist_z[1] - d.hist_z[0] : 0.1;
    d.hist_z.forEach((z, i) => { hx.push(z - w / 2, z + w / 2); hy.push(d.hist[i] || null, d.hist[i] || null); });
    const c2 = new Chart(host, { title: "Распределение пыли по высоте" });
    c2.set({ series: [{ x: hx, y: hy, name: "СДУ (гистограмма)" }, { x: d.z_H, y: d.density_exact, name: "точное", dash: "5 3" },
                      { x: d.z_H, y: d.density_dubrulle, name: "гаусс Дюбрюля", dash: "2 3" }],
             xlog: false, ylog: true, yminRel: 1e-4, xlabel: "z / H", ylabel: "плотность вероятности", xshort: "z/H", yshort: "p", file: "mixing_distribution", legend: "bl" });
    app.mixCharts = [c1, c2];
  } catch (e) {
    showAlert("#mix-alert", e.message);
    $("#mix-status").textContent = "Ошибка";
  } finally {
    btn.disabled = false;
  }
}

// ====================================================================== init
async function init() {
  document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
    document.querySelectorAll(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + t.dataset.tab));
    store.set("tab", t.dataset.tab);
    requestAnimationFrame(() => [...app.charts, ...app.diskCharts, ...app.mixCharts].forEach(c => c.data && c.render()));
  }));
  const tab = store.get("tab");
  if (tab) { const b = document.querySelector(`.tab[data-tab="${tab}"]`); b && b.click(); }

  try {
    app.schema = await api("/api/schema");
  } catch (e) {
    showAlert("#alert", "Не удалось связаться с сервером dustppd: " + e.message);
    return;
  }
  $("#meta").textContent = `v${app.schema.version} · SIMD ${app.schema.backend}`;
  const ping = () => fetch("/api/ping", { method: "POST", body: "{}" }).catch(() => {});
  ping();
  setInterval(ping, 10000);
  addEventListener("pagehide", () => navigator.sendBeacon("/api/bye", "{}"));
  if (app.schema.app_mode) {
    const q = $("#btn-quit");
    q.hidden = false;
    q.addEventListener("click", async () => {
      try { await api("/api/quit", {}); } catch { /* server already gone */ }
      document.body.replaceChildren(el("div", { class: "bye" },
        el("h2", {}, "dustppd закрыт"), el("p", {}, "Эту вкладку можно закрыть. Чтобы начать снова, запустите приложение.")));
    });
  }
  const ps = $("#preset");
  ps.append(el("option", { value: "" }, "— свои параметры —"));
  for (const p of app.schema.presets) ps.append(el("option", { value: p.id }, p.title));
  ps.addEventListener("change", async () => {
    if (!ps.value) return;
    try { setParams((await api("/api/preset", { id: ps.value })).params); store.set("preset", ps.value); ps.value = ps.value; }
    catch (e) { showAlert("#alert", e.message); }
  });
  buildForm(app.schema);
  const saved = store.get("params");
  setParams(saved ? { ...app.schema.defaults, ...saved } : app.schema.defaults);
  ps.value = saved ? (store.get("preset") || "") : "reference";
  buildTrajCharts();

  const mix = store.get("mix");
  if (mix) for (const [k, v] of Object.entries(mix)) { const i = $(`#mix-form [name="${k}"]`); if (i) i.value = v; }

  $("#btn-run").addEventListener("click", runTrajectory);
  $("#btn-cancel").addEventListener("click", () => app.job && api("/api/cancel", { id: app.job }).catch(() => {}));
  $("#btn-csv").addEventListener("click", () => { if (app.job) location.href = `/api/csv?id=${app.job}`; });
  $("#btn-mix").addEventListener("click", runMixing);
  $("#btn-save").addEventListener("click", async () => {
    try {
      const { text } = await api("/api/to_toml", { params: getParams() });
      const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/plain" })), download: "dustppd_config.toml" });
      a.click();
    } catch (e) { showAlert("#alert", e.message); }
  });
  $("#btn-load").addEventListener("click", () => $("#file-toml").click());
  $("#file-toml").addEventListener("change", async ev => {
    const f = ev.target.files[0];
    if (!f) return;
    try { setParams((await api("/api/parse_toml", { text: await f.text() })).params); $("#preset").value = ""; $("#status").textContent = `Загружено: ${f.name}`; }
    catch (e) { showAlert("#alert", "Не удалось прочитать TOML: " + e.message); }
    ev.target.value = "";
  });
  document.addEventListener("keydown", ev => {
    if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") { ev.preventDefault(); $("#page-mix").classList.contains("active") ? runMixing() : runTrajectory(); }
  });
  let rt;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => [...app.charts, ...app.diskCharts, ...app.mixCharts].forEach(c => c.data && c.render()), 120); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => [...app.charts, ...app.diskCharts, ...app.mixCharts].forEach(c => c.data && c.render()));
}

init();
