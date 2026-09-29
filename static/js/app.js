const TARGETS = ["rainfall","tmax","tmin","rh","wind_speed","solar_radiation"];
const PRETTY = {rainfall:"Rainfall",tmax:"Tmax",tmin:"Tmin",rh:"Humidity",wind_speed:"Wind",solar_radiation:"Solar"};
let map, lg, allReports = [];

function initMap() {
  const el = document.getElementById("map");
  if (typeof L === "undefined") {
    el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8b98a5;font-size:13px;text-align:center;padding:20px">Map tiles unavailable offline<br><small>All spatial data is shown in the contour plot below</small></div>';
    return;
  }
  try {
    map = L.map("map").setView([26.2, 91.0], 9);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      { attribution: '&copy; OpenStreetMap contributors', maxZoom: 18 }).addTo(map);
    lg = L.layerGroup().addTo(map);
  } catch (e) {
    el.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#8b98a5;font-size:13px">Map unavailable</div>';
  }
}

function riskColor(f) {
  if (Object.values(f).includes("critical")) return "#e74c3c";
  if (Object.values(f).includes("warning")) return "#f39c12";
  return "#2ecc71";
}

function displayName(r) {
  return r.panchayat_name && r.panchayat_name.trim() ? r.panchayat_name : r.panchayat_id;
}
function displayBlock(r) {
  return r.block_name && r.block_name.trim() ? r.block_name : r.block_id;
}

function drawMarkers(reports) {
  if (!lg) return;
  lg.clearLayers();
  if (!reports.length) return;
  const pts = [];
  reports.forEach(r => {
    const m = L.circleMarker([r.lat, r.lon], {
      radius: 6, color: "#0f1419", weight: 1,
      fillColor: riskColor(r.risk_flags || {}), fillOpacity: 0.9
    });
    m.bindPopup(popupHtml(r), { maxWidth: 320 });
    m.on("click", () => showDetail(r));
    lg.addLayer(m); pts.push([r.lat, r.lon]);
  });
  if (pts.length && map) map.fitBounds(pts, { padding: [30, 30] });
}

function popupHtml(r) {
  const w = r.weather;
  const advs = (r.advisories || []).slice(0, 3)
    .map(a => `<div class="adv ${a.severity}"><b>${a.category} · ${a.severity}</b><br>${a.message}</div>`).join("");
  return `<b>${displayName(r)}</b><br>
    <small style="color:#8b98a5">${displayBlock(r)} · LGD ${r.panchayat_id}</small><br>
    <small>Rain ${w.rainfall.toFixed(1)}mm · Tmax ${w.tmax.toFixed(1)}° · RH ${w.rh.toFixed(0)}%</small>${advs}`;
}

function renderReports(reports) {
  const el = document.getElementById("reports");
  el.innerHTML = "";
  if (!reports.length) {
    el.innerHTML = '<div class="loading">No matches</div>';
    return;
  }
  reports.forEach(r => {
    const c = document.createElement("div");
    c.className = "report-card";
    const chips = Object.entries(r.risk_flags || {})
      .map(([k,v]) => `<span class="chip ${v}">${k}:${v}</span>`).join("");
    const name = displayName(r);
    const block = displayBlock(r);
    c.innerHTML = `<h4>${name}<span style="color:var(--muted);font-size:10px">${block}</span></h4>
      <div style="font-size:10px;color:var(--muted);margin-bottom:4px">LGD: ${r.panchayat_id}</div>
      <div class="weather">Rain ${r.weather.rainfall.toFixed(1)}mm · Tmax ${r.weather.tmax.toFixed(1)}°C ·
      Tmin ${r.weather.tmin.toFixed(1)}°C · RH ${r.weather.rh.toFixed(0)}% · Wind ${r.weather.wind_speed.toFixed(1)}km/h</div>
      <div class="chips">${chips}</div>`;
    c.onclick = () => { showDetail(r); if (map) map.setView([r.lat, r.lon], 11); };
    el.appendChild(c);
  });
}
function showDetail(r) {
  const advs = (r.advisories || []).map(a =>
    `<div class="adv ${a.severity}"><b>${a.category} · ${a.severity} · ${a.rule_id}</b><br>${a.message}</div>`
  ).join("") || "<div class='loading'>No advisories triggered for current conditions.</div>";
  const name = displayName(r);
  const block = displayBlock(r);
  const el = document.getElementById("reports");
  el.innerHTML = `
    <div class="report-card" style="border-color:var(--accent)">
      <h4>${name}<button class="btn back-btn" style="padding:2px 8px;font-size:11px">← All</button></h4>
      <div style="font-size:10px;color:var(--muted);margin-bottom:4px">LGD: ${r.panchayat_id} · ${block}</div>
      <div class="weather">(${r.lat.toFixed(3)}, ${r.lon.toFixed(3)})</div>${advs}
    </div>`;

  const backBtn = el.querySelector(".back-btn");
  if (backBtn) {
    backBtn.onclick = () => {
      const search = document.getElementById("search-box")?.value.trim() || "";
      const filtered = filterReports(search);
      renderReports(filtered);
    };
  }
}


function renderSummary(reports, source) {
  const c = { critical:0, warning:0, info:0 };
  reports.forEach(r => (r.advisories || []).forEach(a => c[a.severity]++));
  document.getElementById("summary").innerHTML = `
    <div class="row"><span>Panchayats</span><span>${reports.length}</span></div>
    <div class="row"><span>Critical</span><span style="color:var(--crit)">${c.critical}</span></div>
    <div class="row"><span>Warnings</span><span style="color:var(--warn)">${c.warning}</span></div>
    <div class="row"><span>Info</span><span>${c.info}</span></div>`;
  document.getElementById("source-tag").textContent = `source: ${source}`;
}

function filterReports(q) {
  if (!q) return allReports;
  const s = q.toLowerCase();
  return allReports.filter(r =>
    (r.panchayat_name || "").toLowerCase().includes(s) ||
    (r.panchayat_id || "").toString().includes(s) ||
    (r.block_name || "").toLowerCase().includes(s) ||
    (r.block_id || "").toString().includes(s)
  );
}

function initPlotTabs() {
  const t = document.getElementById("plot-tabs");
  TARGETS.forEach((v, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === 0 ? " active" : "");
    b.textContent = PRETTY[v];
    b.onclick = () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      document.getElementById("plot-img").src = `/plots/contour_${v}.png?t=${Date.now()}`;
    };
    t.appendChild(b);
  });
  document.getElementById("plot-img").src = `/plots/contour_${TARGETS[0]}.png?t=${Date.now()}`;
}

async function loadReports(force = false) {
  const crop = document.getElementById("crop-select").value;
  const btn = document.getElementById("apply-btn");
  btn.disabled = true; btn.textContent = "Running…";
  try {
    if (force) {
      const r = await fetch("/api/refresh", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ crop: crop || null })
      });
      if (!r.ok) throw new Error(await r.text());
    }
    const res = await fetch(`/api/reports?crop=${encodeURIComponent(crop)}`);
    const data = await res.json();
    allReports = data.reports;
    const sb = document.getElementById("search-box");
    if (sb) sb.value = "";
    drawMarkers(allReports);
    renderReports(allReports);
    renderSummary(allReports, data.source);
    document.getElementById("plot-img").src =
      `/plots/contour_${TARGETS[0]}.png?t=${Date.now()}`;
  } catch (e) {
    document.getElementById("summary").innerHTML = `<div class="loading" style="color:var(--crit)">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Apply";
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initMap();
  initPlotTabs();

  const applyBtn = document.getElementById("apply-btn");
  const refreshBtn = document.getElementById("refresh-btn");
  const searchBox = document.getElementById("search-box");

  if (applyBtn) applyBtn.onclick = () => loadReports(false);
  if (refreshBtn) refreshBtn.onclick = () => loadReports(true);

  if (searchBox) {
    searchBox.addEventListener("input", (e) => {
      const filtered = filterReports(e.target.value.trim());
      renderReports(filtered);
      drawMarkers(filtered);
    });
  } else {
    console.warn("Search box element #search-box not found in DOM");
  }

  loadReports(true);
});