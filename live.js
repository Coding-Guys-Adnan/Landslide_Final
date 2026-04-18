const API_BASE = 'https://landslide-final.onrender.com';

// 1. Initialize Map
const map = L.map('liveMap', {
    center: [22.0, 80.0], // Center of India
    zoom: 5,
    zoomControl: false,
    preferCanvas: true
});

L.control.zoom({ position: 'bottomright' }).addTo(map);

// CartoDB Dark Matter setup
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Separate layers for earthquakes, rainfall, predictions
let eqLayer = L.featureGroup().addTo(map);
let rainLayer = L.featureGroup().addTo(map);
let predLayer = L.featureGroup().addTo(map);

async function safeFetch(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) return [];
        return await res.json();
    } catch (e) {
        return [];
    }
}

async function loadData() {
    const errorBanner = document.getElementById('apiErrorBanner');
    errorBanner.style.display = 'none';

    try {
        // Stats endpoint always works even if model not loaded
        const statsRes = await fetch(`${API_BASE}/api/stats`);
        if (!statsRes.ok) throw new Error('Backend not reachable');
        const statsData = await statsRes.json();

        // These may return 503 while model loads - that's OK
        const [eqData, rainData, predData, historyData] = await Promise.all([
            safeFetch(`${API_BASE}/api/earthquakes`),
            safeFetch(`${API_BASE}/api/rainfall`),
            safeFetch(`${API_BASE}/api/predictions`),
            safeFetch(`${API_BASE}/api/history`)
        ]);

        renderMap(eqData, rainData, predData);
        renderHistory(historyData);
        updateDebugPanel(statsData, predData);

    } catch (e) {
        console.error("Failed to fetch live data:", e);
        errorBanner.style.display = 'block';
    }
}

function renderMap(eqData, rainData, predData) {
    // Clear old layers
    eqLayer.clearLayers();
    rainLayer.clearLayers();
    predLayer.clearLayers();

    // 1. Plot Earthquakes
    eqData.forEach(eq => {
        if (!eq.lat || !eq.lon) return;
        const radius = Math.pow(1.5, eq.mag) * 2; // Exponential mapping
        const marker = L.circleMarker([eq.lat, eq.lon], {
            radius: radius,
            color: '#ff3d6b',
            weight: 2,
            fillColor: 'transparent',
            fillOpacity: 0
        });
        marker.bindPopup(`<b>Magnitude ${eq.mag}</b><br>${eq.place}`);
        eqLayer.addLayer(marker);
    });

    // 2. Plot Rainfall
    rainData.forEach(r => {
        if (!r.lat || !r.lon) return;
        const color = r.rain >= 100 ? '#ff3d6b' : '#00cfff';
        const marker = L.circleMarker([r.lat, r.lon], {
            radius: 5,
            color: color,
            weight: 1,
            fillColor: color,
            fillOpacity: r.rain >= 100 ? 0.8 : 0.4
        });
        marker.bindPopup(`<b>Rainfall (30d)</b><br>${r.rain.toFixed(1)} mm<br>${r.region}`);
        rainLayer.addLayer(marker);
    });

    // 3. Plot Predictions & Update List Panel
    const listDiv = document.getElementById('liveList');
    listDiv.innerHTML = ''; // reset list

    if (predData.length === 0) {
        listDiv.innerHTML = '<div style="text-align: center; color: #888; font-style: italic; margin-top: 20px;">No predictions...</div>';
    }

    const sortedData = [...predData].reverse();
    sortedData.forEach(event => {
        if (!event.lat || !event.lon) return;

        let riskClass = 'risk-low';
        let color = '#00cfff';
        let riskLabel = 'LOW';
        const risk = event.risk_level || event.risk || 'LOW';
        
        if (risk === "HIGH" || risk === "VERY HIGH") {
            riskClass = 'risk-high';
            color = '#ff3d6b';
            riskLabel = risk;
        } else if (risk === "MODERATE") {
            riskClass = 'risk-moderate';
            color = '#f5a623';
            riskLabel = risk;
        }

        // Plot to Map
        const marker = L.circleMarker([event.lat, event.lon], {
            radius: riskClass === 'risk-high' ? 8 : 4,
            fillColor: color,
            color: color,
            weight: riskClass === 'risk-high' ? 2 : 1,
            opacity: 0.8,
            fillOpacity: 0.5
        });
        
        marker.bindPopup(`
            <div style="font-family: var(--font-mono); font-size: 0.8rem;">
                <div style="font-weight: bold; color: ${color};">${riskLabel} - ${(event.probability || 0).toFixed(1)}%</div>
                <div>Trigger: ${event.trigger_type}</div>
                <div style="color: #aaa; margin-top: 4px;">Temp: ${event.timestamp}</div>
            </div>
        `);
        predLayer.addLayer(marker);

        // Plot to List
        const item = document.createElement('div');
        item.className = `live-item ${riskClass}`;
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: ${color}; font-weight: bold;">${riskLabel}</span>
                <span style="color: #aaa; font-size: 0.75rem;">${event.timestamp || ''}</span>
            </div>
            <div style="font-size: 0.95rem; margin-bottom: 8px;">${event.place}</div>
            <div style="font-size: 0.75rem; color: #888; display: grid; grid-template-columns: 1fr; gap: 5px;">
                <div>Trigger: <span style="color: #fff">${event.trigger_type} ${event.magnitude ? '(M'+event.magnitude+')' : ''}</span></div>
                <div>Prob: <span style="color: ${color}">${(event.probability || 0).toFixed(1)}%</span></div>
            </div>
        `;
        listDiv.appendChild(item);

        item.addEventListener('mouseenter', () => {
            map.flyTo([event.lat, event.lon], 8, { duration: 0.5 });
            marker.openPopup();
        });
    });
}

function renderHistory(historyData) {
    const histDiv = document.getElementById('historyList');
    histDiv.innerHTML = ''; // reset history

    if (historyData.length === 0) {
        histDiv.innerHTML = '<div style="text-align: center; color: #888; font-style: italic; margin-top: 20px;">No historical alerts...</div>';
        return;
    }

    historyData.forEach(event => {
        if (!event.lat || !event.lon) return;

        let riskClass = 'risk-low';
        let color = '#00cfff';
        const r = event.risk_level;
        
        if (r === "VERY HIGH" || r === "HIGH") {
            riskClass = "risk-high";
            color = "#ff3d6b";
        } else if (r === "MODERATE") {
            riskClass = "risk-moderate";
            color = "#f5a623";
        }

        const item = document.createElement('div');
        item.className = `live-item ${riskClass}`;
        item.style.cursor = 'pointer';
        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="color: ${color}; font-weight: bold;">${event.risk_level}</span>
                <span style="color: #aaa; font-size: 0.70rem;">${event.timestamp || ''}</span>
            </div>
            <div style="font-size: 0.90rem; margin-bottom: 8px;">${event.place}</div>
            <div style="font-size: 0.75rem; color: #888; display: grid; grid-template-columns: 1fr; gap: 5px;">
                <div>Trigger: <span style="color: #fff">${event.trigger_type}</span></div>
                <div>Prob: <span style="color: ${color}">${(event.probability || 0).toFixed(1)}%</span></div>
            </div>
        `;
        
        // Add click listener to fly the map back to this coordinate
        item.addEventListener('click', () => {
            map.flyTo([event.lat, event.lon], 9, { duration: 1.0 });
            
            // Generate a temporary marker on the map to visualize the historical location dynamically
            const tempMarker = L.circleMarker([event.lat, event.lon], {
                radius: 10, fillOpacity: 0.8, color: '#fff', fillColor: color, weight: 3
            }).addTo(map);
            
            tempMarker.bindPopup(`
                <div style="font-family: var(--font-mono); font-size: 0.8rem;">
                    <b style="color:${color}">${event.risk_level} RECORD</b><br>
                    Time: ${event.timestamp}<br>
                    Prob: ${(event.probability || 0).toFixed(1)}%
                </div>
            `).openPopup();
            
            // Remove after 10 seconds
            setTimeout(() => { map.removeLayer(tempMarker) }, 10000);
        });

        histDiv.appendChild(item);
    });
}

function updateDebugPanel(statsData, predData) {
    document.getElementById('debugEqCount').textContent = statsData.earthquakes_cached;
    document.getElementById('debugRainCount').textContent = statsData.rainfall_grids_cached;
    document.getElementById('debugPredCount').textContent = predData.length;
    document.getElementById('eventCount').textContent = `(${predData.length})`;
    document.getElementById('debugTime').textContent = statsData.last_updated;
}

// History Toggle Logic
document.getElementById('toggleHistoryBtn').addEventListener('click', (e) => {
    const histPanel = document.getElementById('historyPanel');
    if (histPanel.style.display === 'none' || histPanel.style.display === '') {
        histPanel.style.display = 'flex';
        e.target.textContent = 'Hide History 🕒';
        map.invalidateSize(); // Fix map render bounds
    } else {
        histPanel.style.display = 'none';
        e.target.textContent = 'Show History 🕒';
        map.invalidateSize();
    }
});

// Refresh Button Logic
document.getElementById('refreshDataBtn').addEventListener('click', async () => {
    const btn = document.getElementById('refreshDataBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = 'Searching... ⏳';
    try {
        await fetch(`${API_BASE}/api/force_refresh`);
    } catch (e) {
        console.error("Force refresh failed", e);
    }
    await loadData();
    setTimeout(() => { btn.innerHTML = originalText; }, 1000);
});

// 4. Auto Refresh Interval (15 minutes = 900000 ms)
setInterval(loadData, 900000);
loadData(); // Initial load

// 5. Subscription Logic
document.getElementById('subscribeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const emailStr = document.getElementById('subEmail').value;
    const btn = document.querySelector('.subscribe-btn');
    const msg = document.getElementById('subMsg');
    
    btn.textContent = 'Wait...';
    btn.disabled = true;
    msg.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/api/subscribe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email: emailStr })
        });

        if (!response.ok) throw new Error('Subscription API failed');
        msg.style.display = 'block';
        msg.textContent = 'Successfully Subscribed! ✓';
        msg.style.color = '#00cfff';
        document.getElementById('subEmail').value = '';
    } catch (err) {
        console.error(err);
        msg.style.display = 'block';
        msg.textContent = 'Error subscribing.';
        msg.style.color = '#ff3d6b';
    } finally {
        btn.textContent = 'Subscribe';
        btn.disabled = false;
        setTimeout(() => msg.style.display = 'none', 5000);
    }
});
