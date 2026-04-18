// LOADER
function setLoader(pct, msg) { document.getElementById('lbar').style.width = pct + '%'; document.getElementById('lsub').textContent = msg }

let RAW_DATA = [];
let map, layerGroup;

async function init() {
  setLoader(20, 'Fetching data from API…');
  try {
    const response = await fetch('https://landslide-final.onrender.com/api/data');
    if (!response.ok) throw new Error('API failed to respond');
    setLoader(80, 'Parsing records…');
    RAW_DATA = await response.json();
    setLoader(95, 'Initialising map…');
    await new Promise(r => setTimeout(r, 60));
    bootMap();
    setLoader(100, 'Ready!');
    await new Promise(r => setTimeout(r, 280));
    const ld = document.getElementById('loader'); ld.style.opacity = '0';
    setTimeout(() => ld.style.display = 'none', 400);
  } catch (error) {
    document.getElementById('lsub').textContent = "Error: Check if backend is running. " + error.message;
  }
}

function bootMap() {
  map = L.map('map', { zoomControl: true }).setView([22, 82], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap', maxZoom: 18 }).addTo(map);
  layerGroup = L.layerGroup().addTo(map);
  renderSidebar(); renderCurrentMap();
}

// ── COLORS ──────────────────────────────────────────────────────
function lerp(a, b, t) { return a + (b - a) * t }
function lc(c1, c2, t) { return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)] }
function rgb(r, g, b) { return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})` }
function t01(v, mn, mx) { return Math.max(0, Math.min(1, (v - mn) / (mx - mn))) }
function magColor(v) { const t = t01(v, 4, 7.8); return t < .5 ? rgb(...lc([30, 140, 255], [255, 210, 0], t * 2)) : rgb(...lc([255, 210, 0], [255, 30, 60], (t - .5) * 2)) }
function depthColor(v) { return rgb(...lc([0, 220, 190], [160, 20, 220], t01(v, 0, 300))) }
function pgaColor(v) { const t = t01(v, 0, 1); return t < .5 ? rgb(...lc([50, 220, 100], [255, 220, 0], t * 2)) : rgb(...lc([255, 220, 0], [255, 30, 60], (t - .5) * 2)) }
function mmiColor(v) { return rgb(...lc([80, 200, 255], [255, 40, 40], t01(v, 1, 10))) }
function rainColor(v) { return rgb(...lc([200, 235, 255], [0, 50, 220], t01(v, 0, 500))) }
function slopeColor(v) { return rgb(...lc([60, 200, 80], [230, 60, 20], t01(v, 0, 70))) }
function ndviColor(v) { return rgb(...lc([200, 155, 70], [20, 180, 55], t01(v, -0.2, 1))) }
function elevColor(v) { const t = t01(v, 0, 5000); if (t < .4) return rgb(...lc([80, 180, 80], [160, 220, 100], t / .4)); if (t < .7) return rgb(...lc([160, 220, 100], [200, 200, 180], (t - .4) / .3)); return rgb(...lc([200, 200, 180], [255, 255, 255], (t - .7) / .3)) }
function clayColor(v) { return rgb(...lc([240, 230, 200], [160, 90, 30], t01(v, 0, 60))) }
function sandColor(v) { return rgb(...lc([240, 200, 100], [180, 120, 30], t01(v, 0, 80))) }
function phColor(v) { return rgb(...lc([255, 80, 80], [80, 200, 255], t01(v, 4, 9))) }
function riverColor(v) { return rgb(...lc([255, 220, 0], [255, 30, 80], t01(v, 0, 10))) }

const LC_COL = { 'Grassland': '#7ec850', 'Water': '#2196f3', 'Bare/sparse': '#c8aa76', 'Tree cover': '#1a7a1a', 'Shrubland': '#a0c060', 'Cropland': '#f0d040', 'Herbaceous wetland': '#40c0c0', 'Built-up': '#e05030' };
const REG_COL = { 'Himalayan Belt': '#ff6b35', 'Central India': '#00d4ff', 'Peninsular India': '#a8e063', 'NE India': '#da70f5' };
const DSRC_COL = { 'None': '#3a4060', 'Combined': '#f5a623', 'USGS': '#00cfff', 'Seismic': '#ff3d6b', 'Rainfall': '#6ee87a', 'Other': '#aaa' };
const DSRC_FULL = { 'None': 'Below all thresholds', 'Combined': 'Combined-Kanungo2009', 'USGS': 'USGS-Direct', 'Seismic': 'Seismic-Newmark/Jibson2007', 'Rainfall': 'Rainfall-Guzzetti2008' };

// ── POPUP ────────────────────────────────────────────────────────
function popup(d) {
  const occColor = d.locc ? '#ff3d6b' : '#6ee87a';
  const dsrcBg = { 'None': '#1a1f30', 'Combined': '#3a2a08', 'USGS': '#0a2a3a', 'Seismic': '#3a0a18', 'Rainfall': '#0a2a18', 'Other': '#222' };
  return `<div class="popup">
    <div class="popup-head">
      <div class="popup-place">${d.place || '—'}</div>
      <div class="popup-region">${d.region} · ${d.year}</div>
    </div>
    <div class="popup-body">
      <div class="popup-row"><span class="popup-lbl">Magnitude</span><span class="popup-val" style="color:#f5a623">${d.mag} Mw</span></div>
      <div class="popup-row"><span class="popup-lbl">Depth</span><span class="popup-val">${d.depth} km</span></div>
      <div class="popup-row"><span class="popup-lbl">PGA</span><span class="popup-val">${d.pga} g</span></div>
      <div class="popup-row"><span class="popup-lbl">MMI</span><span class="popup-val">${d.mmi}</span></div>
      <div class="popup-row"><span class="popup-lbl">Landslide</span><span class="popup-val" style="color:${occColor}">${d.locc ? 'YES' : 'NO'}</span></div>
      <div class="popup-row"><span class="popup-lbl">Detection</span><span class="popup-val"><span class="dsrc-badge" style="background:${dsrcBg[d.dsrc] || '#222'};color:${DSRC_COL[d.dsrc] || '#aaa'}">${d.dsrc}</span></span></div>
      <div class="popup-row"><span class="popup-lbl">Rainfall 30d</span><span class="popup-val">${d.rain} mm</span></div>
      <div class="popup-row"><span class="popup-lbl">Peak Rain</span><span class="popup-val">${d.prain} mm</span></div>
      <div class="popup-row"><span class="popup-lbl">NDVI</span><span class="popup-val">${d.ndvi}</span></div>
      <div class="popup-row"><span class="popup-lbl">Elevation</span><span class="popup-val">${d.elev} m</span></div>
      <div class="popup-row"><span class="popup-lbl">Slope</span><span class="popup-val">${d.slope}°</span></div>
      <div class="popup-row"><span class="popup-lbl">Land Cover</span><span class="popup-val" style="color:${LC_COL[d.lc] || '#fff'}">${d.lc}</span></div>
      <div class="popup-row"><span class="popup-lbl">River Dist</span><span class="popup-val">${d.rdist} km</span></div>
      <div class="popup-row"><span class="popup-lbl">Clay/Sand</span><span class="popup-val">${d.clay}%/${d.sand}%</span></div>
      <div class="popup-row"><span class="popup-lbl">pH</span><span class="popup-val">${d.ph}</span></div>
      <div class="popup-row"><span class="popup-lbl">Steep</span><span class="popup-val">${d.steep ? '>25°' : ''} ${d.vsteep ? '>40°' : '' || '—'}</span></div>
    </div>
  </div>`;
}

// ── RENDER ───────────────────────────────────────────────────────
function clearLayers() { layerGroup.clearLayers() }
function dots(data, colorFn, sizeFn, opac = 0.75) {
  clearLayers();
  data.forEach(d => {
    L.circleMarker([d.lat, d.lon], { radius: sizeFn(d), fillColor: colorFn(d), color: 'rgba(0,0,0,0.2)', weight: .4, fillOpacity: opac })
      .bindPopup(popup(d), { maxWidth: 310 }).addTo(layerGroup);
  });
}
function gradBar(css, l, m, r) { return `<div class="grad-bar" style="background:${css}"></div><div class="grad-lbls"><span>${l}</span><span>${m}</span><span>${r}</span></div>` }
function mbarSet(items, mx) {
  return items.map(it => `<div class="mbar">
    <span class="mbar-lbl">${it.label}</span>
    <div class="mbar-track"><div class="mbar-fill" style="width:${Math.max(2, (it.val / mx * 100)).toFixed(1)}%;background:${it.color}"></div></div>
    <span class="mbar-val">${it.val.toLocaleString()}</span>
  </div>`).join('');
}

// ── FILTERS ───────────────────────────────────────────────────────
let F = { region: 'All', year: 'All', locc: 'All' };
function filtered() {
  return RAW_DATA.filter(d => {
    if (F.region !== 'All' && d.region !== F.region) return false;
    if (F.year !== 'All' && d.year !== parseInt(F.year)) return false;
    if (F.locc !== 'All' && d.locc !== parseInt(F.locc)) return false;
    return true;
  });
}
function applyFilter(k, v) { F[k] = v; renderCurrentMap() }
function buildFilters() {
  const regions = ['All', ...new Set(RAW_DATA.map(d => d.region))].sort();
  const years = ['All', ...new Set(RAW_DATA.map(d => d.year))].sort((a, b) => a === 'All' ? -1 : b === 'All' ? 1 : a - b);
  return `<div class="sb-section">
    <div class="sb-title">🔎 Filters</div>
    <div class="flt-label">Region</div>
    <select class="flt-select" onchange="applyFilter('region',this.value)">${regions.map(r => `<option ${F.region === r ? 'selected' : ''}>${r}</option>`).join('')}</select>
    <div class="flt-label">Year</div>
    <select class="flt-select" onchange="applyFilter('year',this.value)">${years.map(y => `<option ${F.year == y ? 'selected' : ''}>${y}</option>`).join('')}</select>
    <div class="flt-label">Landslide Occurred</div>
    <select class="flt-select" onchange="applyFilter('locc',this.value)">
      <option value="All" ${F.locc === 'All' ? 'selected' : ''}>All</option>
      <option value="1" ${F.locc === '1' ? 'selected' : ''}>YES — Occurred</option>
      <option value="0" ${F.locc === '0' ? 'selected' : ''}>NO — Not Occurred</option>
    </select>
    <div class="count-pill"><span id="shownCount">15140</span> <span>events shown</span></div>
  </div>`;
}

// ── MAP DEFINITIONS ────────────────────────────────────────────────
const MAPS = {

  earthquake: {
    title: 'Earthquake Magnitude Distribution',
    floatLegend() { return `<div class="lf-title">Magnitude (Mw)</div>${gradBar('linear-gradient(90deg,#1e8cff,#ffd200,#ff1e3c)', '4.0 Mw', '5.9 Mw', '7.8 Mw')}` },
    render(data) { dots(data, d => magColor(d.mag), d => Math.max(2.5, (d.mag - 3.5) * 2.8), 0.82) },
    sidebar() {
      const bins = [{ l: '4.0–4.9', f: d => d.mag < 5, c: '#1e8cff' }, { l: '5.0–5.9', f: d => d.mag >= 5 && d.mag < 6, c: '#ffd200' }, { l: '6.0–6.9', f: d => d.mag >= 6 && d.mag < 7, c: '#ff8800' }, { l: '7.0+', f: d => d.mag >= 7, c: '#ff1e3c' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">Magnitude Classes</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="note">Dot size & colour both encode Mw. M≥7 events: <b>${RAW_DATA.filter(d => d.mag >= 7).length}</b></div>`;
    }
  },

  depth: {
    title: 'Focal Depth Distribution',
    floatLegend() { return `<div class="lf-title">Depth (km)</div>${gradBar('linear-gradient(90deg,#00dcc8,#9060e0,#b414dc)', '0 km', '150 km', '300+ km')}` },
    render(data) { dots(data, d => depthColor(d.depth), d => 3.5, 0.72) },
    sidebar() {
      const bins = [{ l: 'Shallow 0–30km', f: d => d.depth < 30, c: '#00dcc8' }, { l: 'Interm. 30–70km', f: d => d.depth >= 30 && d.depth < 70, c: '#9060e0' }, { l: 'Deep >70km', f: d => d.depth >= 70, c: '#b414dc' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">Depth Classes</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="note">Shallow quakes (cyan) cause greatest surface damage. Deep events (purple) have wider but gentler shaking footprints.</div>`;
    }
  },

  pga: {
    title: 'Peak Ground Acceleration — PGA (g)',
    floatLegend() { return `<div class="lf-title">PGA (g)</div>${gradBar('linear-gradient(90deg,#32dc64,#ffdc00,#ff1e3c)', '0g', '0.5g', '1g+')}` },
    render(data) { dots(data, d => pgaColor(d.pga || 0), d => Math.max(2.5, (d.pga || 0) * 12 + 2.5), 0.78) },
    sidebar() {
      const bins = [{ l: 'Low <0.1g', f: d => d.pga < 0.1, c: '#32dc64' }, { l: 'Mod 0.1–0.3g', f: d => d.pga >= 0.1 && d.pga < 0.3, c: '#ffdc00' }, { l: 'High >0.3g', f: d => d.pga >= 0.3, c: '#ff1e3c' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">PGA Classes</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="note">Values >0.3g can cause significant structural damage and trigger slope failures.</div>`;
    }
  },

  mmi: {
    title: 'Modified Mercalli Intensity (MMI)',
    floatLegend() { return `<div class="lf-title">MMI Scale</div>${gradBar('linear-gradient(90deg,#50c8ff,#80e080,#f0c000,#ff2244)', 'I', 'V', 'X')}` },
    render(data) { dots(data, d => mmiColor(d.mmi || 1), d => Math.max(2.5, (d.mmi || 1) * 1.1 + 1), 0.78) },
    sidebar() {
      const levels = [[1, 'I Not felt', '#50c8ff'], [2, 'II Weak', '#40b0f0'], [3, 'III Weak', '#30a0e0'], [4, 'IV Light', '#80e080'], [5, 'V Moderate', '#d0e040'], [6, 'VI Strong', '#f0c000'], [7, 'VII V.Strong', '#f07000'], [8, 'VIII Severe', '#e04000'], [9, 'IX Violent', '#cc2020'], [10, 'X Extreme', '#ff2244']];
      return `<div class="sb-section"><div class="sb-title">MMI Levels</div>
        ${levels.map(l => { const c = RAW_DATA.filter(d => d.mmi === l[0]).length; return c > 0 ? `<div class="leg-item"><div class="leg-dot" style="background:${l[2]}"></div>${l[1]}<span class="leg-count">${c.toLocaleString()}</span></div>` : '' }).join('')}
      </div>`;
    }
  },

  landslide_occ: {
    title: 'Landslide Occurrence — Research Grade Labels',
    floatLegend() {
      return `<div class="lf-title">Landslide Occurred</div>
      <div class="leg-item"><div class="leg-dot" style="background:#ff3d6b"></div>YES — Occurred</div>
      <div class="leg-item"><div class="leg-dot" style="background:#3a4060;border:1px solid #556"></div>NO — Not Occurred</div>`},
    render(data) { dots(data, d => d.locc ? '#ff3d6b' : '#2a3050', d => d.locc ? 5 : 2.5, d => d.locc ? 0.9 : 0.45) },
    sidebar() {
      const occ = RAW_DATA.filter(d => d.locc === 1).length;
      const no = RAW_DATA.filter(d => d.locc === 0).length;
      const pct = (occ / (occ + no) * 100).toFixed(1);
      return `<div class="sb-section"><div class="sb-title">Occurrence Summary</div>
        <div class="mbar"><span class="mbar-lbl">YES Occurred</span><div class="mbar-track"><div class="mbar-fill" style="width:${pct}%;background:#ff3d6b"></div></div><span class="mbar-val">${occ.toLocaleString()}</span></div>
        <div class="mbar"><span class="mbar-lbl">NO Occurred</span><div class="mbar-track"><div class="mbar-fill" style="width:${(100 - parseFloat(pct)).toFixed(1)}%;background:#3a4080"></div></div><span class="mbar-val">${no.toLocaleString()}</span></div>
      </div>
      <div class="sb-section"><div class="sb-title">By Region (% YES)</div>
        ${['Himalayan Belt', 'Central India', 'Peninsular India', 'NE India'].map(r => {
        const rpts = RAW_DATA.filter(d => d.region === r);
        const ro = rpts.filter(d => d.locc === 1).length;
        const rpct = (ro / rpts.length * 100).toFixed(1);
        return `<div class="leg-item"><div class="leg-dot" style="background:${REG_COL[r]}"></div>${r}<span class="leg-count">${rpct}% (${ro})</span></div>`;
      }).join('')}
      </div>
      <div class="note">Bright red = confirmed landslide. Dark blue = no landslide. Only <b>${pct}%</b> of M≥4 events produced confirmed landslides in this dataset.</div>`;
    }
  },

  detection: {
    title: 'Landslide Detection Source',
    floatLegend() {
      return `<div class="lf-title">Detection Source</div>
      ${Object.entries(DSRC_COL).filter(([k]) => k !== 'Other').map(([k, c]) => `<div class="leg-item"><div class="leg-dot" style="background:${c}"></div>${k}<span class="leg-count" style="font-size:.54rem">${DSRC_FULL[k] || ''}</span></div>`).join('')}`
    },
    render(data) { dots(data, d => DSRC_COL[d.dsrc] || '#aaa', d => d.dsrc === 'None' ? 2 : 5, d => d.dsrc === 'None' ? 0.3 : 0.9) },
    sidebar() {
      const items = Object.keys(DSRC_COL).filter(k => k !== 'Other').map(k => ({ label: k, val: RAW_DATA.filter(d => d.dsrc === k).length, color: DSRC_COL[k] }));
      const mx = Math.max(...items.map(i => i.val));
      return `<div class="sb-section"><div class="sb-title">Detection Methods</div>${mbarSet(items, mx)}</div>
      <div class="sb-section"><div class="sb-title">Method Details</div>
        ${Object.entries(DSRC_FULL).map(([k, v]) => `<div class="leg-item"><div class="leg-dot" style="background:${DSRC_COL[k]}"></div><span style="font-size:.58rem">${v}</span></div>`).join('')}
      </div>
      <div class="note">Non-grey dots = confirmed landslide events. "None" (grey/dark) = below all research thresholds. Use the Detection Source filter to isolate a specific method.</div>`;
    }
  },

  rainfall: {
    title: '30-Day Cumulative Rainfall (mm)',
    floatLegend() { return `<div class="lf-title">Rainfall 30d (mm)</div>${gradBar('linear-gradient(90deg,#c8e4ff,#2060e0,#001880)', '0', '250', '500+ mm')}` },
    render(data) { dots(data, d => rainColor(d.rain || 0), d => 3.5, 0.72) },
    sidebar() {
      const bins = [{ l: '<50 mm', f: d => d.rain < 50, c: '#c8e4ff' }, { l: '50–150 mm', f: d => d.rain >= 50 && d.rain < 150, c: '#6ab0ff' }, { l: '150–300 mm', f: d => d.rain >= 150 && d.rain < 300, c: '#2060e0' }, { l: '>300 mm', f: d => d.rain >= 300, c: '#001880' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">Rainfall Classes</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="note">30-day rainfall is a primary landslide trigger. Compare with Landslide Occurred to see the correlation with high-rainfall zones.</div>`;
    }
  },

  slope: {
    title: 'Terrain Slope Angle (°)',
    floatLegend() { return `<div class="lf-title">Slope (°)</div>${gradBar('linear-gradient(90deg,#3cc850,#ffe040,#e63c14)', '0°', '35°', '70°+')}` },
    render(data) { dots(data, d => slopeColor(d.slope || 0), d => 3.5, 0.75) },
    sidebar() {
      const bins = [{ l: 'Gentle <15°', f: d => d.slope < 15, c: '#3cc850' }, { l: 'Mod 15–25°', f: d => d.slope >= 15 && d.slope < 25, c: '#ffe040' }, { l: 'Steep 25–40°', f: d => d.slope >= 25 && d.slope < 40, c: '#f07010' }, { l: 'Very Steep >40°', f: d => d.slope >= 40, c: '#e63c14' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">Slope Classes</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="sb-section"><div class="sb-title">Steep Terrain</div>
        <div class="leg-item">Steep >25°<span class="leg-count">${RAW_DATA.filter(d => d.steep === 1).length.toLocaleString()}</span></div>
        <div class="leg-item">Very Steep >40°<span class="leg-count">${RAW_DATA.filter(d => d.vsteep === 1).length.toLocaleString()}</span></div>
      </div>
      <div class="note">Steep slopes + heavy rainfall = high landslide probability. Himalayan Belt dominates in steep terrain.</div>`;
    }
  },

  ndvi: {
    title: 'NDVI — Vegetation Density Index',
    floatLegend() { return `<div class="lf-title">NDVI</div>${gradBar('linear-gradient(90deg,#c89650,#80c040,#14b430)', '-0.2 Bare', '0.4', '1.0 Dense')}` },
    render(data) { dots(data, d => ndviColor(d.ndvi || 0), d => 3.5, 0.75) },
    sidebar() {
      return `<div class="sb-section"><div class="sb-title">Scale</div>
        <div class="grad-bar" style="background:linear-gradient(90deg,#c89650,#80c040,#14b430)"></div>
        <div class="grad-lbls"><span>-0.2 Bare</span><span>0.4</span><span>1.0 Dense</span></div>
      </div>
      <div class="sb-section"><div class="sb-title">Avg NDVI by Land Cover</div>
        ${Object.entries(LC_COL).map(([lc, col]) => { const pts = RAW_DATA.filter(d => d.lc === lc && d.ndvi != null); const avg = pts.length ? (pts.reduce((s, d) => s + d.ndvi, 0) / pts.length).toFixed(2) : '—'; return `<div class="leg-item"><div class="leg-dot" style="background:${col}"></div>${lc}<span class="leg-count">${avg}</span></div>` }).join('')}
      </div>
      <div class="note">Lower NDVI (brown) = sparser vegetation, higher erosion risk. Dense vegetation (green) stabilises slopes.</div>`;
    }
  },

  elevation: {
    title: 'Terrain Elevation (m above sea level)',
    floatLegend() { return `<div class="lf-title">Elevation (m)</div>${gradBar('linear-gradient(90deg,#50b450,#a0d070,#e0e8d8,#ffffff)', '0 m', '2500 m', '5000+ m')}` },
    render(data) { dots(data, d => elevColor(d.elev || 0), d => 3.5, 0.72) },
    sidebar() {
      const bins = [{ l: 'Plains 0–200m', f: d => d.elev < 200, c: '#64b464' }, { l: 'Hills 200–1km', f: d => d.elev >= 200 && d.elev < 1000, c: '#98c878' }, { l: 'Mtns 1–3km', f: d => d.elev >= 1000 && d.elev < 3000, c: '#c0d8b8' }, { l: 'High >3km', f: d => d.elev >= 3000, c: '#e8eee8' }];
      const items = bins.map(b => ({ label: b.l, val: RAW_DATA.filter(b.f).length, color: b.c }));
      return `<div class="sb-section"><div class="sb-title">Elevation Zones</div>${mbarSet(items, Math.max(...items.map(i => i.val)))}</div>
      <div class="note">Himalayan Belt events cluster at high elevations. Peninsular India events mostly on plains.</div>`;
    }
  },

  landcover: {
    title: 'Land Cover Classification',
    floatLegend() { return `<div class="lf-title">Land Cover</div>${Object.entries(LC_COL).map(([lc, c]) => `<div class="leg-item"><div class="leg-sq" style="background:${c}"></div>${lc}</div>`).join('')}` },
    render(data) { dots(data, d => LC_COL[d.lc] || '#888', d => 3.5, 0.78) },
    sidebar() {
      const items = Object.entries(LC_COL).map(([lc, c]) => ({ label: lc, val: RAW_DATA.filter(d => d.lc === lc).length, color: c }));
      const mx = Math.max(...items.map(i => i.val));
      return `<div class="sb-section"><div class="sb-title">Cover Types</div>${mbarSet(items, mx)}</div>
      <div class="note">Tree cover stabilises slopes. Bare/sparse areas are most vulnerable to mass movement and erosion.</div>`;
    }
  },

  soil: {
    title: 'Soil Properties — Clay · Sand · pH',
    floatLegend() {
      return `<div class="lf-title">Soil: <span id="soilMode">Clay %</span></div>
      <div style="display:flex;gap:4px;margin-top:6px">
        <button onclick="setSoilMode('clay')" style="flex:1;padding:3px;background:var(--border2);border:none;color:var(--text);border-radius:3px;cursor:pointer;font-size:.56rem;font-family:var(--font-mono)">Clay</button>
        <button onclick="setSoilMode('sand')" style="flex:1;padding:3px;background:var(--border2);border:none;color:var(--text);border-radius:3px;cursor:pointer;font-size:.56rem;font-family:var(--font-mono)">Sand</button>
        <button onclick="setSoilMode('ph')" style="flex:1;padding:3px;background:var(--border2);border:none;color:var(--text);border-radius:3px;cursor:pointer;font-size:.56rem;font-family:var(--font-mono)">pH</button>
      </div>
      <div id="soilGrad" style="margin-top:6px"></div>`},
    render(data) {
      const m = window._soilMode || 'clay';
      if (m === 'clay') dots(data, d => clayColor(d.clay || 0), d => 3.5, 0.75);
      else if (m === 'sand') dots(data, d => sandColor(d.sand || 0), d => 3.5, 0.75);
      else dots(data, d => phColor(d.ph || 7), d => 3.5, 0.75);
      updateSoilGrad();
    },
    sidebar() {
      return `<div class="sb-section"><div class="sb-title">Sub-Layer</div>
        <div style="display:flex;gap:5px">
          <button onclick="setSoilMode('clay')" class="flt-select" style="cursor:pointer;text-align:center">🟤 Clay</button>
          <button onclick="setSoilMode('sand')" class="flt-select" style="cursor:pointer;text-align:center">🟡 Sand</button>
          <button onclick="setSoilMode('ph')" class="flt-select" style="cursor:pointer;text-align:center">🧪 pH</button>
        </div>
      </div>
      <div class="sb-section"><div class="sb-title">Stats</div>
        ${[['Clay (%)', 'clay'], ['Sand (%)', 'sand'], ['pH', 'ph']].map(([label, key]) => {
        const vals = RAW_DATA.map(d => d[key]).filter(v => v != null && !isNaN(v));
        const avg = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2);
        return `<div class="leg-item">${label}<span class="leg-count">avg ${avg}</span></div>`;
      }).join('')}
      </div>
      <div class="note">Clay retains water → liquefaction risk. Sandy soils drain faster. pH affects vegetation cover and erosion resistance.</div>`;
    }
  },

  river: {
    title: 'River Proximity & Stream Order',
    floatLegend() {
      return `<div class="lf-title">River Dist (km)</div>${gradBar('linear-gradient(90deg,#ffdc00,#ff1e50)', '0 km', '5 km', '10+ km')}
      <div class="leg-item" style="margin-top:6px"><div class="leg-dot" style="background:#ffdc00"></div>Near River (≤2km)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#ff1e50"></div>Distant</div>`},
    render(data) { dots(data, d => riverColor(Math.min(d.rdist || 0, 10)), d => 4.0, 0.95) },
    sidebar() {
      const near = RAW_DATA.filter(d => d.nriver === 1).length;
      const total = RAW_DATA.length;
      return `<div class="sb-section"><div class="sb-title">River Proximity</div>
        <div class="mbar"><span class="mbar-lbl">Near River</span><div class="mbar-track"><div class="mbar-fill" style="width:${(near / total * 100).toFixed(1)}%;background:#ffdc00"></div></div><span class="mbar-val">${near.toLocaleString()}</span></div>
        <div class="mbar"><span class="mbar-lbl">Far</span><div class="mbar-track"><div class="mbar-fill" style="width:${((total - near) / total * 100).toFixed(1)}%;background:#ff1e50"></div></div><span class="mbar-val">${(total - near).toLocaleString()}</span></div>
      </div>
      <div class="sb-section"><div class="sb-title">Stream Order</div>
        ${[1, 2, 3, 4, 5, 6].map(o => { const c = RAW_DATA.filter(d => d.stord === o).length; return c > 0 ? `<div class="leg-item">Order ${o}<span class="leg-count">${c.toLocaleString()}</span></div>` : '' }).join('')}
      </div>
      <div class="note">Events near rivers have higher landslide risk — erosion undercuts slopes and soils become waterlogged.</div>`;
    }
  },

  compare: {
    title: 'Regional Comparison — All 4 Seismic Zones',
    floatLegend() { return `<div class="lf-title">Regions</div>${Object.entries(REG_COL).map(([r, c]) => `<div class="leg-item"><div class="leg-dot" style="background:${c}"></div>${r}</div>`).join('')}` },
    render(data) { dots(data, d => REG_COL[d.region] || '#888', d => 3, 0.72) },
    sidebar() {
      const regions = ['Himalayan Belt', 'Central India', 'Peninsular India', 'NE India'];
      return `<div class="sb-section"><div class="sb-title">Region Stats</div>
        ${regions.map(r => {
        const pts = RAW_DATA.filter(d => d.region === r);
        const avg = k => (pts.reduce((s, d) => s + (d[k] || 0), 0) / pts.length).toFixed(2);
        const locc = ((pts.filter(d => d.locc === 1).length / pts.length) * 100).toFixed(1);
        const usgs = pts.filter(d => d.dsrc === 'USGS').length;
        const comb = pts.filter(d => d.dsrc === 'Combined').length;
        return `<div class="rcard" style="border-left-color:${REG_COL[r]}">
            <div class="rcard-name" style="color:${REG_COL[r]}">${r}</div>
            <div class="rcard-row">Events: <b>${pts.length.toLocaleString()}</b> · Landslide: <b>${locc}%</b></div>
            <div class="rcard-row">Avg Mag: <b>${avg('mag')} Mw</b> · Depth: <b>${avg('depth')} km</b></div>
            <div class="rcard-row">Avg PGA: <b>${avg('pga')} g</b> · Rain: <b>${avg('rain')} mm</b></div>
            <div class="rcard-row">USGS: <b>${usgs}</b> · Combined: <b>${comb}</b></div>
          </div>`;
      }).join('')}
      </div>`;
    }
  }
};

// ── SOIL SUB-MODE ────────────────────────────────────────────────
window._soilMode = 'clay';
function setSoilMode(m) {
  window._soilMode = m;
  const el = document.getElementById('soilMode'); if (el) el.textContent = { clay: 'Clay %', sand: 'Sand %', ph: 'pH' }[m];
  updateSoilGrad(); renderCurrentMap();
}
function updateSoilGrad() {
  const el = document.getElementById('soilGrad'); if (!el) return;
  const m = window._soilMode || 'clay';
  const g = { clay: 'linear-gradient(90deg,#f0e6c8,#a05a1e)', sand: 'linear-gradient(90deg,#f0c850,#b47820)', ph: 'linear-gradient(90deg,#ff5050,#50c8ff)' };
  const l = { clay: ['0%', '30%', '60%'], sand: ['0%', '40%', '80%'], ph: ['4 Acid', '6.5', '9 Base'] };
  el.innerHTML = `<div class="grad-bar" style="background:${g[m]}"></div><div class="grad-lbls"><span>${l[m][0]}</span><span>${l[m][1]}</span><span>${l[m][2]}</span></div>`;
}

// ── ORCHESTRATION ─────────────────────────────────────────────────
let currentMode = 'earthquake';
function renderCurrentMap() {
  if (!layerGroup) return;
  const data = filtered();
  const sc = document.getElementById('shownCount'); if (sc) sc.textContent = data.length.toLocaleString();
  document.getElementById('statShown').textContent = data.length.toLocaleString();
  MAPS[currentMode].render(data);
  const fl = document.getElementById('floatLegend');
  if (fl) { fl.style.display = 'block'; fl.innerHTML = MAPS[currentMode].floatLegend(); }
}
function renderSidebar() {
  const sb = document.getElementById('sidebar'); if (!sb) return;
  sb.innerHTML = buildFilters() + (MAPS[currentMode].sidebar() || '');
}
function switchMap(mode) {
  currentMode = mode;
  document.querySelectorAll('.tab').forEach((t, i) => {
    const modes = ['earthquake', 'depth', 'pga', 'mmi', 'landslide_occ', 'detection', 'rainfall', 'slope', 'ndvi', 'elevation', 'landcover', 'soil', 'river', 'compare'];
    t.classList.toggle('active', modes[i] === mode);
  });
  document.getElementById('mapTitle').textContent = MAPS[mode].title;
  renderSidebar(); renderCurrentMap();
}

init().catch(e => { console.error(e); document.getElementById('lsub').textContent = 'Error: ' + e.message });
