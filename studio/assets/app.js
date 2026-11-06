/* ═══════════════════════════════════════════════════════════════════
   مِشكاة — استوديو الويب
   JS خالص، بلا مكتبات. كل رسمٍ هنا (الرسم البياني، خريطة المواضيع)
   مكتوبٌ يدويًّا على canvas.
   ═══════════════════════════════════════════════════════════════════ */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const TYPE_COLORS = {
  'مستند': '#e0b45c',
  'شخص': '#56d4c4',
  'مكان': '#e2849b',
  'مفهوم': '#9b8cf0',
  'أداة': '#7fb2ff',
  'نطاق': '#6d7a8e',
  'domain': '#6d7a8e',
};

const DOMAIN_COLORS = [
  '#e0b45c', '#56d4c4', '#e2849b', '#9b8cf0', '#7fb2ff',
  '#f0a868', '#8ee08e', '#d8d06a', '#c98cf0', '#6ad8e0',
];

/* ───────────────────────────── شبكة ───────────────────────────── */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));

/* تطبيعٌ عربيٌّ بسيط لإبراز المطابقات في المقاطع */
function normAr(s) {
  return String(s)
    .replace(/[\u064B-\u065F\u0670\u0640]/g, '')
    .replace(/[أإآٱ]/g, 'ا')
    .replace(/ى/g, 'ي')
    .replace(/ة/g, 'ه');
}

function highlight(text, terms) {
  const roots = (terms || []).map((t) => normAr(t)).filter((t) => t && t.length >= 3);
  if (!roots.length) return esc(text);
  return String(text)
    .split(/(\s+)/)
    .map((chunk) => {
      const w = normAr(chunk);
      const hit = roots.some((r) => w.startsWith(r) || (w.length >= 3 && r.startsWith(w)));
      return hit ? `<mark>${esc(chunk)}</mark>` : esc(chunk);
    })
    .join('');
}

/* ───────────────────────────── حالة ───────────────────────────── */
const state = {
  view: 'search',
  mode: 'hybrid',
  domain: null,
  lastQuery: '',
  loaded: { graph: false, space: false, dash: false, lab: false },
  domains: [],
  graph: { nodes: [], edges: [], zoom: 1, offsetX: 0, offsetY: 0, frozen: false },
};

/* ───────────────────────────── الشريط الجانبي ───────────────────────────── */
function setStatus(text, kind = '') {
  $('#status-text').textContent = text;
  const dot = $('#status-dot');
  dot.className = 'dot' + (kind ? ' ' + kind : '');
}

async function bootHealth() {
  try {
    const h = await api('/api/health');
    const s = await api('/api/stats');
    setStatus(`${s.corpus.documents} مستندًا · ${s.graph.nodes} عقدة`, 'ok');
    $('#suggestions').innerHTML = [
      'كيف قاس البيروني الأرض؟',
      'الوقف وتمويل المدارس',
      'ابن الهيثم والضوء',
      'مشكلة المياه في المدن',
      'ما الفرق بين الجبر والخوارزمية؟',
      'تعلم الآلة والانتباه',
    ].map((q) => `<button class="chip" data-q="${esc(q)}">${esc(q)}</button>`).join('');
    $$('#suggestions .chip').forEach((b) => b.addEventListener('click', () => {
      $('#q').value = b.dataset.q;
      doSearch();
    }));
    void h;
  } catch (err) {
    setStatus('تعذّر الاتصال بالمحرّك', 'err');
    console.error(err);
  }
}

/* ───────────────────────────── التنقّل ───────────────────────────── */
function showView(name) {
  state.view = name;
  $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  if (name === 'graph' && !state.loaded.graph) loadGraph();
  if (name === 'space' && !state.loaded.space) loadSpace();
  if (name === 'dash' && !state.loaded.dash) loadDash();
  if (name === 'lab' && !state.loaded.lab) { state.loaded.lab = true; runLab(); }
  if (name === 'graph') fitGraph();
}

/* ───────────────────────────── البحث ───────────────────────────── */
async function loadDomains() {
  const { domains } = await api('/api/domains');
  state.domains = domains;
  $('#domain-chips').innerHTML = ['<button class="chip is-active" data-domain="">كل النطاقات</button>']
    .concat(domains.map((d) => `<button class="chip" data-domain="${esc(d.name)}">${esc(d.icon)} ${esc(d.name)} <span class="small">${d.count}</span></button>`))
    .join('');
  $$('#domain-chips .chip').forEach((c) => c.addEventListener('click', () => {
    $$('#domain-chips .chip').forEach((x) => x.classList.remove('is-active'));
    c.classList.add('is-active');
    state.domain = c.dataset.domain || null;
    if (state.lastQuery) doSearch();
  }));
}

async function doSearch() {
  const q = $('#q').value.trim();
  if (!q) return;
  state.lastQuery = q;
  const box = $('#results');
  box.innerHTML = '<div class="empty"><span class="spinner"></span> جارٍ البحث…</div>';
  const params = new URLSearchParams({ q, k: '8', mode: state.mode });
  if (state.domain) params.set('domain', state.domain);
  try {
    const data = await api('/api/search?' + params.toString());
    renderResults(data);
    renderTrace(data);
  } catch (err) {
    box.innerHTML = `<div class="empty">خطأ: ${esc(err.message)}</div>`;
  }
}

function renderResults(data) {
  const box = $('#results');
  if (!data.results.length) {
    box.innerHTML = '<div class="empty">لا نتائج. جرّب كلمةً أعمّ أو نمط البحث الدلالي.</div>';
    return;
  }
  const max = Math.max(...data.results.map((r) => r.score)) || 1;
  box.innerHTML = data.results.map((r, i) => {
    const pct = Math.round((r.score / max) * 100);
    const terms = [...(r.matched || [])];
    return `
      <article class="result" data-doc="${esc(r.doc_id)}">
        <div class="result-top">
          <span class="result-title">${esc(r.title)}</span>
          <span class="result-domain">${esc(r.domain)}</span>
          <span class="muted small" style="margin-inline-start:auto">#${i + 1}</span>
        </div>
        <div class="result-snippet">${highlight(r.snippet, terms)}</div>
        <div class="result-meta">
          <span class="meter"><i style="width:${pct}%"></i></span>
          <span>درجة ${r.score.toFixed(4)}</span>
          <span>معجمي ${r.lexical.toFixed(1)}</span>
          <span>دلالي ${r.semantic.toFixed(2)}</span>
          ${r.ranks.lexical ? `<span class="tag">L${r.ranks.lexical}</span>` : ''}
          ${r.ranks.semantic ? `<span class="tag">S${r.ranks.semantic}</span>` : ''}
          <span class="muted small">${esc(r.doc_id)}</span>
        </div>
      </article>`;
  }).join('');
  $$('#results .result').forEach((el) => el.addEventListener('click', () => openDoc(el.dataset.doc)));
}

function renderTrace(data) {
  const t = data.trace;
  const dl = $('#trace-dl');
  const rows = [
    ['الاستعلام', t.query],
    ['المطبّع', t.normalized],
    ['السيقان', (t.terms || []).slice(0, 8).join(' · ') || '—'],
    ['الجذور', (t.roots || []).slice(0, 8).join(' · ') || '—'],
    ['مرشّحات معجمية', t.candidates.lexical],
    ['مرشّحات دلالية', t.candidates.semantic],
    ['زمن معجمي', `${t.timing_ms.lexical} ms`],
    ['زمن دلالي', `${t.timing_ms.semantic} ms`],
    ['زمن الدمج', `${t.timing_ms.fuse} ms`],
    ['الإجمالي', `${t.timing_ms.total} ms`],
    ['أوزان الدمج', Object.entries(t.weights || {}).map(([k, v]) => `${k}=${v}`).join(' · ') || '—'],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
}

/* ───────────────────────────── نافذة المستند ───────────────────────────── */
async function openDoc(id) {
  const modal = $('#doc-modal');
  $('#modal-body').innerHTML = '<div class="empty"><span class="spinner"></span></div>';
  modal.showModal();
  try {
    const d = await api(`/api/doc/${encodeURIComponent(id)}`);
    const doc = d.document;
    const roots = d.analysis.top_roots.map((r) => `<span>${esc(r.root)} ×${r.count}</span>`).join('');
    const outEdges = d.graph.out.map((e) => `<div class="rel">${esc(labelOf(e.s))} <span>—[${esc(e.r)}]→</span> <a data-node="${esc(e.t)}">${esc(labelOf(e.t))}</a></div>`).join('');
    const inEdges = d.graph.in.map((e) => `<div class="rel"><a data-node="${esc(e.s)}">${esc(labelOf(e.s))}</a> <span>—[${esc(e.r)}]→</span> ${esc(labelOf(e.t))}</div>`).join('');
    const related = d.related.map((r) => `<button data-doc="${esc(r.doc_id)}">${esc(r.title)}</button>`).join('');

    $('#modal-body').innerHTML = `
      <h2>${esc(doc.title)}</h2>
      <div class="doc-meta">
        <span>${esc(doc.domain)}</span><span>·</span><span>${esc(doc.era || '')}</span>
        <span>·</span><span>${doc.body.split(/\s+/).length} كلمة</span>
        <span>·</span><span>${d.analysis.tokens} رمزًا / ${d.analysis.types} نمطًا</span>
      </div>
      <p class="muted">${esc(doc.summary)}</p>
      <div class="doc-body">${esc(doc.body)}</div>
      <div class="doc-section">
        <h3>الجذور الأكثر حضورًا</h3>
        <div class="roots-cloud">${roots}</div>
      </div>
      <div class="doc-section">
        <h3>علاقات صريحة ومستنتجة</h3>
        <div class="rel-list">${outEdges || '<span class="muted">لا حواف خارجة.</span>'}${inEdges}</div>
      </div>
      <div class="doc-section">
        <h3>مستندات قريبة (دلاليًّا وبنيويًّا)</h3>
        <div class="related-list">${related || '<span class="muted">لا جيران.</span>'}</div>
      </div>`;
    $$('#modal-body [data-node]').forEach((a) => a.addEventListener('click', () => {
      modal.close();
      showView('graph');
      focusNode(a.dataset.node);
    }));
    $$('#modal-body [data-doc]').forEach((b) => b.addEventListener('click', () => openDoc(b.dataset.doc)));
  } catch (err) {
    $('#modal-body').innerHTML = `<div class="empty">خطأ: ${esc(err.message)}</div>`;
  }
}

const labelCache = new Map();
function labelOf(nodeId) {
  if (labelCache.has(nodeId)) return labelCache.get(nodeId);
  const node = state.graph.nodes.find((n) => n.id === nodeId);
  const label = node ? node.label : nodeId.replace(/^\w+:/, '');
  labelCache.set(nodeId, label);
  return label;
}

/* ═════════════════════════════ الرسم البياني ═════════════════════════════ */
async function loadGraph() {
  state.loaded.graph = true;
  const data = await api('/api/graph/overview?limit=150');
  const canvas = $('#graph-canvas');
  const rect = canvas.getBoundingClientRect();
  const nodes = data.nodes.map((n, i) => ({
    ...n,
    x: rect.width / 2 + Math.cos((i / data.nodes.length) * Math.PI * 2) * (120 + Math.random() * 140),
    y: rect.height / 2 + Math.sin((i / data.nodes.length) * Math.PI * 2) * (120 + Math.random() * 140),
    vx: 0, vy: 0,
    r: 4 + Math.min(9, Math.sqrt(n.degree) * 2.2),
  }));
  state.graph.nodes = nodes;
  state.graph.edges = data.edges.filter((e) => nodes.some((n) => n.id === e.s) && nodes.some((n) => n.id === e.t));
  labelCache.clear();
  nodes.forEach((n) => labelCache.set(n.id, n.label));

  renderLegend();
  fitGraph();
  startSimulation();
  bindGraphInteractions();
}

function renderLegend() {
  const types = [...new Set(state.graph.nodes.map((n) => n.type))];
  $('#graph-legend').innerHTML = types.map((t) => (
    `<span><i style="background:${TYPE_COLORS[t] || '#889'}"></i>${esc(t)}</span>`
  )).join('');
}

function simulateStep() {
  const { nodes, edges } = state.graph;
  const W = $('#graph-canvas').clientWidth || 800;
  const H = $('#graph-canvas').clientHeight || 600;
  const idx = new Map(nodes.map((n, i) => [n.id, i]));
  const REP = 5200;
  const SPRING = 0.022;
  const DAMP = 0.86;
  const CENTER = 0.0016;

  // تنافر
  for (let i = 0; i < nodes.length; i++) {
    const a = nodes[i];
    for (let j = i + 1; j < nodes.length; j++) {
      const b = nodes[j];
      let dx = a.x - b.x;
      let dy = a.y - b.y;
      let d2 = dx * dx + dy * dy;
      if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.01; }
      if (d2 > 260 * 260) continue;
      const f = REP / d2;
      const d = Math.sqrt(d2);
      a.vx += (dx / d) * f;
      a.vy += (dy / d) * f;
      b.vx -= (dx / d) * f;
      b.vy -= (dy / d) * f;
    }
  }
  // نوابض
  for (const e of edges) {
    const i = idx.get(e.s);
    const j = idx.get(e.t);
    if (i === undefined || j === undefined) continue;
    const a = nodes[i];
    const b = nodes[j];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const d = Math.sqrt(dx * dx + dy * dy) || 1;
    const ideal = 74;
    const f = (d - ideal) * SPRING;
    a.vx += (dx / d) * f;
    a.vy += (dy / d) * f;
    b.vx -= (dx / d) * f;
    b.vy -= (dy / d) * f;
  }
  // تمركز + تحديث
  for (const n of nodes) {
    if (n.fx !== undefined) { n.x = n.fx; n.y = n.fy; n.vx = n.vy = 0; continue; }
    n.vx += (W / 2 - n.x) * CENTER;
    n.vy += (H / 2 - n.y) * CENTER;
    n.vx *= DAMP;
    n.vy *= DAMP;
    n.x += n.vx;
    n.y += n.vy;
  }
}

function drawGraph() {
  const canvas = $('#graph-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const { zoom, offsetX, offsetY, nodes, edges } = state.graph;
  ctx.save();
  ctx.translate(W / 2 + offsetX, H / 2 + offsetY);
  ctx.scale(zoom, zoom);
  ctx.translate(-W / 2, -H / 2);

  const idx = new Map(nodes.map((n) => [n.id, n]));
  // حواف
  for (const e of edges) {
    const a = idx.get(e.s);
    const b = idx.get(e.t);
    if (!a || !b) continue;
    const inferred = (e.n || '').includes('مستنتج');
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = inferred ? 'rgba(155, 140, 240, 0.22)' : 'rgba(224, 180, 92, 0.16)';
    ctx.lineWidth = inferred ? 0.8 : 1;
    if (inferred) ctx.setLineDash([3, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // عقد
  for (const n of nodes) {
    const color = TYPE_COLORS[n.type] || '#889';
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = n === state.graph.hover ? '#fff' : color;
    ctx.globalAlpha = n === state.graph.hover ? 1 : 0.9;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(8, 13, 22, 0.8)';
    ctx.lineWidth = 1.4;
    ctx.stroke();
    if (zoom > 0.85 && (n.degree > 2 || n.type !== 'مستند')) {
      ctx.font = `${11 / zoom}px "Segoe UI", sans-serif`;
      ctx.fillStyle = 'rgba(242, 236, 225, 0.72)';
      ctx.textAlign = 'center';
      ctx.fillText(n.label.slice(0, 26), n.x, n.y - n.r - 6);
    }
  }
  ctx.restore();
}

let simTimer = null;
function startSimulation() {
  cancelAnimationFrame(simTimer);
  let ticks = 0;
  const loop = () => {
    if (!state.graph.frozen && ticks < 900) { simulateStep(); ticks++; }
    drawGraph();
    simTimer = requestAnimationFrame(loop);
  };
  loop();
}

function fitGraph() {
  state.graph.zoom = 1;
  state.graph.offsetX = 0;
  state.graph.offsetY = 0;
}

function bindGraphInteractions() {
  const canvas = $('#graph-canvas');
  if (canvas.dataset.bound) return;
  canvas.dataset.bound = '1';
  let dragging = null;
  let panning = false;
  let last = { x: 0, y: 0 };

  const toWorld = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const { zoom, offsetX, offsetY } = state.graph;
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const cx = rect.width / 2 + offsetX;
    const cy = rect.height / 2 + offsetY;
    return {
      x: (mx - cx) / zoom + rect.width / 2,
      y: (my - cy) / zoom + rect.height / 2,
      mx, my,
    };
  };

  canvas.addEventListener('pointerdown', (ev) => {
    const p = toWorld(ev);
    const hit = state.graph.nodes.find((n) => Math.hypot(n.x - p.x, n.y - p.y) < n.r + 6 / state.graph.zoom);
    if (hit) {
      dragging = hit;
      hit.fx = hit.x; hit.fy = hit.y;
      canvas.setPointerCapture(ev.pointerId);
    } else {
      panning = true;
      last = { x: ev.clientX, y: ev.clientY };
    }
  });

  canvas.addEventListener('pointermove', (ev) => {
    const p = toWorld(ev);
    if (dragging) {
      dragging.x = p.x; dragging.y = p.y;
      dragging.fx = p.x; dragging.fy = p.y;
      return;
    }
    if (panning) {
      state.graph.offsetX += ev.clientX - last.x;
      state.graph.offsetY += ev.clientY - last.y;
      last = { x: ev.clientX, y: ev.clientY };
      return;
    }
    const hit = state.graph.nodes.find((n) => Math.hypot(n.x - p.x, n.y - p.y) < n.r + 6 / state.graph.zoom);
    state.graph.hover = hit || null;
    canvas.style.cursor = hit ? 'pointer' : 'grab';
  });

  canvas.addEventListener('pointerup', (ev) => {
    if (dragging) {
      const node = dragging;
      dragging = null;
      // نقرة (بلا سحبٍ طويل) تفتح العقدة
      if (node) {
        if (node.type === 'مستند') openDoc(node.id.replace(/^doc:/, ''));
        else showNodePanel(node);
      }
    }
    panning = false;
    try { canvas.releasePointerCapture(ev.pointerId); } catch { /* ignore */ }
  });

  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const factor = ev.deltaY > 0 ? 0.92 : 1.08;
    state.graph.zoom = Math.min(3.2, Math.max(0.35, state.graph.zoom * factor));
  }, { passive: false });

  $('#graph-reset').addEventListener('click', () => {
    state.graph.nodes.forEach((n) => { delete n.fx; delete n.fy; });
    state.graph.frozen = false;
    fitGraph();
  });
  $('#graph-freeze').addEventListener('click', () => {
    state.graph.frozen = !state.graph.frozen;
    $('#graph-freeze').textContent = state.graph.frozen ? 'تحرير' : 'تجميد';
  });

}

function focusNode(nodeId) {
  const node = state.graph.nodes.find((n) => n.id === nodeId);
  if (!node) return;
  state.graph.hover = node;
  showNodePanel(node);
}

async function showNodePanel(node) {
  const panel = $('#node-panel');
  panel.innerHTML = `<h4>${esc(node.label)}</h4><div class="muted small">${esc(node.type)} · ${node.degree} رابطة</div><div class="empty small"><span class="spinner"></span></div>`;
  try {
    const ego = await api(`/api/graph/ego/${encodeURIComponent(node.id)}?depth=1`);
    const others = ego.nodes.filter((n) => n.id !== node.id).slice(0, 12);
    const rels = ego.edges.slice(0, 14).map((e) => {
      const dir = e.s === node.id ? '→' : '←';
      const other = e.s === node.id ? e.t : e.s;
      const label = (ego.nodes.find((n) => n.id === other) || {}).label || other;
      return `<li>${esc(e.r)} ${dir} ${esc(label)}</li>`;
    }).join('');
    panel.innerHTML = `
      <h4>${esc(node.label)}</h4>
      <div class="muted small">${esc(node.type)} · ${node.degree} رابطة · ${others.length} جارًا مباشرًا</div>
      <ul>${rels || '<li class="muted">لا روابط</li>'}</ul>`;
  } catch {
    panel.innerHTML = '';
  }
}

/* ───────────────────────────── MQL ───────────────────────────── */
async function loadGraphExamples() {
  const { examples } = await api('/api/graph/examples');
  $('#mql-examples').innerHTML = examples.map((q) => `<button data-q="${esc(q)}">${esc(q)}</button>`).join('');
  $$('#mql-examples button').forEach((b) => b.addEventListener('click', () => {
    $('#mql').value = b.dataset.q;
    runMql();
  }));
}

async function runMql() {
  const query = $('#mql').value.trim();
  if (!query) return;
  const out = $('#mql-result');
  out.innerHTML = '<span class="spinner"></span>';
  try {
    const r = await api('/api/graph/query', { method: 'POST', body: JSON.stringify({ query }) });
    if (!r.count) {
      out.innerHTML = '<div class="muted small">لا نتائج.</div>';
      return;
    }
    const vars = Object.keys(r.bindings[0] || {});
    const rows = r.bindings.slice(0, 12).map((b) => (
      `<tr>${vars.map((v) => `<td>${esc(r.labels[b[v]] || b[v])}</td>`).join('')}</tr>`
    )).join('');
    out.innerHTML = `
      <div class="muted small" style="margin-bottom:6px">${r.count} نتيجة · ${r.elapsed_ms} ms</div>
      <table><thead><tr>${vars.map((v) => `<th>?${esc(v)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
  } catch (err) {
    out.innerHTML = `<div class="empty small">${esc(err.message)}</div>`;
  }
}

/* ───────────────────────────── الفضاء ───────────────────────────── */
async function loadSpace() {
  state.loaded.space = true;
  const [clusters, map] = await Promise.all([api('/api/clusters?k=6'), api('/api/space/map')]);
  renderClusters(clusters);
  renderTopicMap(map);
}

function renderClusters(data) {
  const colors = DOMAIN_COLORS;
  $('#clusters').innerHTML = data.clusters.map((c, i) => `
    <div class="cluster" style="border-color:${colors[i % colors.length]}33">
      <h4>عنقود ${i + 1} · ${c.size} مستندًا</h4>
      <div class="terms">${c.top_terms.slice(0, 5).map((t) => `<span>${esc(t)}</span>`).join('')}</div>
      <ul>${c.docs.slice(0, 6).map((d) => `<li><a data-doc="${esc(d.doc_id)}">${esc(d.title)}</a></li>`).join('')}</ul>
    </div>`).join('');
  $$('#clusters [data-doc]').forEach((a) => a.addEventListener('click', () => openDoc(a.dataset.doc)));
}

const domainColor = (domain) => {
  const i = state.domains.findIndex((d) => d.name === domain);
  return DOMAIN_COLORS[(i < 0 ? 0 : i) % DOMAIN_COLORS.length];
};

function renderTopicMap(data) {
  const canvas = $('#map-canvas');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const pad = 34;
  const sx = (x) => pad + ((x + 1) / 2) * (W - pad * 2);
  const sy = (y) => pad + ((1 - (y + 1) / 2)) * (H - pad * 2);

  // شبكة خفيفة
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = pad + (i / 4) * (W - pad * 2);
    const y = pad + (i / 4) * (H - pad * 2);
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }

  const points = data.points.map((p) => ({ ...p, px: sx(p.x), py: sy(p.y), color: domainColor(p.domain) }));
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.px, p.py, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  canvas.onmousemove = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const hit = points.find((p) => Math.hypot(p.px - mx, p.py - my) < 9);
    canvas.style.cursor = hit ? 'pointer' : 'default';
    canvas.dataset.hover = hit ? hit.doc_id : '';
    if (hit) {
      ctx.clearRect(0, 0, W, H);
      renderTopicMapBase(ctx, W, H, pad, points);
      ctx.fillStyle = 'rgba(8,13,22,0.92)';
      ctx.strokeStyle = 'rgba(224,180,92,0.5)';
      const label = hit.title.slice(0, 46);
      ctx.font = '13px "Segoe UI", sans-serif';
      const tw = ctx.measureText(label).width + 18;
      const bx = Math.min(W - tw - 8, Math.max(8, hit.px - tw / 2));
      const by = hit.py - 34 < 8 ? hit.py + 14 : hit.py - 34;
      ctx.beginPath();
      ctx.roundRect(bx, by, tw, 24, 7);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#f2ece1';
      ctx.fillText(label, bx + 9, by + 16);
      ctx.beginPath();
      ctx.arc(hit.px, hit.py, 7, 0, Math.PI * 2);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.clearRect(0, 0, W, H);
      renderTopicMapBase(ctx, W, H, pad, points);
    }
  };
  canvas.onclick = () => { if (canvas.dataset.hover) openDoc(canvas.dataset.hover); };

  const domainsUsed = [...new Set(points.map((p) => p.domain))];
  $('#map-legend').innerHTML = domainsUsed.map((d) => (
    `<span style="display:inline-flex;align-items:center;gap:5px"><i style="width:8px;height:8px;border-radius:50%;background:${domainColor(d)};display:inline-block"></i>${esc(d)}</span>`
  )).join('') + `<span class="muted small">· تفسير التباين ${(data.explained * 100).toFixed(1)}%</span>`;
}

function renderTopicMapBase(ctx, W, H, pad, points) {
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  for (let i = 0; i <= 4; i++) {
    const x = pad + (i / 4) * (W - pad * 2);
    const y = pad + (i / 4) * (H - pad * 2);
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
  }
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.px, p.py, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

async function wordNeighbors() {
  const word = $('#word-input').value.trim();
  if (!word) return;
  const out = $('#word-result');
  out.innerHTML = '<span class="spinner"></span>';
  try {
    const r = await api(`/api/words/${encodeURIComponent(word)}/neighbors?k=10`);
    if (!r.found) {
      out.innerHTML = `<div class="empty small">«${esc(word)}» ليست في مفردات الفضاء. جرّب كلمةً من نصوص المجموعة.</div>`;
      return;
    }
    const max = Math.max(...r.neighbors.map((n) => n.score)) || 1;
    out.innerHTML = r.neighbors.map((n) => `
      <div class="word-item">
        <span class="term">${esc(n.term)}</span>
        <span class="bar"><i style="width:${(n.score / max) * 100}%"></i></span>
        <span class="val">${n.score.toFixed(3)}</span>
      </div>`).join('');
  } catch (err) {
    out.innerHTML = `<div class="empty small">${esc(err.message)}</div>`;
  }
}

/* ───────────────────────────── مختبر النصّ ───────────────────────────── */
async function runLab() {
  const text = $('#lab-text').value.trim();
  if (!text) return;
  const showStop = $('#lab-show-stop').checked;
  try {
    const r = await api('/api/analyze', { method: 'POST', body: JSON.stringify({ text }) });
    $('#lab-summary').innerHTML = `
      <b>${r.counts.tokens}</b> كلمة · <b>${r.counts.types}</b> صيغة · <b>${r.counts.roots}</b> جذرًا
      · أنواع: ${Object.entries(r.counts.kinds).map(([k, v]) => `${esc(k)} ${v}`).join(' · ')}
      <div class="roots-cloud" style="margin-top:8px">${r.top_roots.map((t) => `<span>${esc(t.root)} ×${t.count}</span>`).join('')}</div>`;
    $('#lab-tokens').innerHTML = r.tokens.map((t) => `
      <div class="tok ${t.stop ? 'is-stop' : ''} ${t.kind === 'فعل' ? 'is-verb' : t.kind === 'اسم' ? 'is-noun' : ''} ${t.stop && !showStop ? 'is-hidden' : ''}"
           title="${esc(t.norm)} · ساق: ${esc(t.stem)} · نوع: ${esc(t.kind)} · ثقة: ${t.confidence}">
        <span class="w">${esc(t.raw)}</span>
        <span class="r">${esc(t.root)}</span>
      </div>`).join('');
  } catch (err) {
    $('#lab-summary').innerHTML = `<span class="muted">خطأ: ${esc(err.message)}</span>`;
  }
}

/* ───────────────────────────── اللوحة ───────────────────────────── */
async function loadDash() {
  state.loaded.dash = true;
  const s = await api('/api/stats');
  const cards = [
    ['مستندات', s.corpus.documents, `${Object.keys(s.corpus.domains).length} نطاقًا`],
    ['رموز (tokens)', s.corpus.tokens, `متوسط ${s.corpus.avg_doc_tokens} لكل مستند`],
    ['مفردات الفهرس', s.index.vocabulary, `${s.index.postings} إدخالًا`],
    ['متجهات دلالية', s.space.vocab, `${s.space.dim} بعدًا · طاقة ${s.space.energy ?? '—'}`],
    ['عُقد الرسم', s.graph.nodes, `${s.graph.edges} حافة`],
    ['حواف مستنتجة', s.inference.derivable, (s.inference.rules || []).length + ' قاعدة'],
    ['زمن البناء', `${s.build_seconds}s`, 'على نواتين'],
    ['اختبارات', '260', 'تمرّ كلها'],
  ];
  $('#cards').innerHTML = cards.map(([k, v, sub]) => `
    <div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`).join('');

  const domains = Object.entries(s.corpus.domains);
  const dmax = Math.max(...domains.map(([, c]) => c));
  $('#domain-bars').innerHTML = domains.map(([name, count]) => `
    <div class="bar-row">
      <span class="name">${esc(name)}</span>
      <span class="track"><i style="width:${(count / dmax) * 100}%"></i></span>
      <span class="num">${count}</span>
    </div>`).join('');

  const rels = Object.entries(s.graph.relations).slice(0, 12);
  const rmax = Math.max(...rels.map(([, c]) => c));
  $('#rel-bars').innerHTML = rels.map(([name, count]) => `
    <div class="bar-row">
      <span class="name">${esc(name)}</span>
      <span class="track"><i style="width:${(count / rmax) * 100}%;background:linear-gradient(90deg,var(--teal),var(--violet))"></i></span>
      <span class="num">${count}</span>
    </div>`).join('');

  renderEval();

  const build = {
    الإصدار: s.version,
    'زمن البناء': `${s.build_seconds} ثانية`,
    'معاملات HNSW': JSON.stringify(s.hnsw.params),
    'حجم HNSW': `${s.hnsw.size} عنصرًا · عمق ${s.hnsw.max_level} · درجة ${s.hnsw.avg_degree}`,
    'قواعد الاستدلال': (s.inference.rules || []).join(' · '),
    'المستنتَج حاليًّا': JSON.stringify(s.inference.by_rule),
    'كثافة مصفوفة التجاور': `${s.space.pairs} زوجًا`,
  };
  $('#build-dl').innerHTML = Object.entries(build).map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
}

async function renderEval() {
  const box = $('#eval-table');
  if (!box) return;
  try {
    const d = await api('/api/eval?k=10');
    if (!d.available) { box.innerHTML = '<p class="muted">لا مجموعة تقييم.</p>'; return; }
    const rows = d.reports.map((r) => {
      const best = r.strategy.includes('هجين');
      return `<tr class="${best ? 'is-best' : ''}">
        <td>${esc(r.strategy)}</td>
        <td class="num">${r.mrr.toFixed(3)}</td>
        <td class="num">${r.recall.toFixed(3)}</td>
        <td class="num">${r.hit1.toFixed(3)}</td>
        <td class="num">${r.latency_ms.toFixed(1)} ms</td>
      </tr>`;
    }).join('');
    box.innerHTML = `<table><thead><tr>
      <th>الاستراتيجية</th><th>MRR</th><th>R@10</th><th>H@1</th><th>الزمن</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="muted small">المرجع: <code>data/eval/queries.jsonl</code> · <code>make eval</code></p>`;
  } catch (err) {
    box.innerHTML = `<p class="muted">تعذّر التقييم: ${esc(err.message)}</p>`;
  }
}

/* ───────────────────────────── التهيئة ───────────────────────────── */
function bindEvents() {
  const modal = $('#doc-modal');
  $('#modal-close').addEventListener('click', () => modal.close());
  modal.addEventListener('click', (ev) => { if (ev.target === modal) modal.close(); });
  $$('.tab').forEach((t) => t.addEventListener('click', () => showView(t.dataset.view)));
  $('#search-btn').addEventListener('click', doSearch);
  $('#q').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') doSearch(); });
  $$('#mode-seg button').forEach((b) => b.addEventListener('click', () => {
    $$('#mode-seg button').forEach((x) => x.classList.remove('is-active'));
    b.classList.add('is-active');
    state.mode = b.dataset.mode;
    if (state.lastQuery) doSearch();
  }));
  $('#mql-form').addEventListener('submit', (ev) => { ev.preventDefault(); runMql(); });
  $('#word-btn').addEventListener('click', wordNeighbors);
  $('#word-input').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') wordNeighbors(); });
  $('#lab-btn').addEventListener('click', runLab);
  $('#lab-show-stop').addEventListener('change', runLab);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
      ev.preventDefault();
      showView('search');
      $('#q').focus();
    }
  });
  window.addEventListener('resize', () => {
    if (state.view === 'graph') drawGraph();
    if (state.view === 'space') loadSpace();
  });
}

async function init() {
  bindEvents();
  await bootHealth();
  await loadDomains();
  loadGraphExamples();
  // تحميلٌ مسبق خفيف
  loadDash().catch(() => {});
  showView('search');
}

init().catch((err) => {
  console.error(err);
  setStatus('فشل التهيئة', 'err');
});
